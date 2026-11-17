"""统一调度：按"任务×月"派发作业子进程，按实测内存放量，同组数据集对齐读盘，结束自动清理。

由 ``python -m trans_to_min run <运行配置>`` 启动。默认只打印计划，加 ``--execute`` 才派发。
中断后以同一配置重跑即可续接（已发布日期跳过）。在state_root下创建STOP文件，调度器
不再派发新作业，等已运行作业结束后退出。

资源规则：

- 内存总预算 = 总内存 × memory_fraction（上限0.95），其中含源文件暂存预算；
- 每个日期进程按"当日输入行数 × 每行字节数 × 1.15"预留；每行字节数初值200，
  每个作业结束后按实测峰值校准；
- 派发前同时检查剩余预算与当前MemAvailable，任一不足只暂停派发，不终止已运行作业；
- 每个日期进程的分块线程数 = CPU核数 ÷ 同时运行的日期进程数，限制在1~16。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import pyarrow.parquet as pq

from . import source_stage
from ._version import __version__
from .adapter import input_path, validate_matching_date_sets
from .plan import (
    LIGHT_REFERENCE_DAYS, RunPlan, TaskGroup, load_task_groups, metric_roots, task_datasets,
)
from .reference_check import CHECKPOINT_DIRECTORY
from .storage import month_path, published_dates


GB = 1024**3
DEFAULT_BYTES_PER_ROW = 200
SAFETY = 1.15
MAX_THREADS_PER_DATE = 16
MEMORY_RESERVE_BYTES = 8 * GB
POLL_SECONDS = 5
INTERMEDIATE_ENTRIES = (".group_staging", CHECKPOINT_DIRECTORY, ".task.lock")
LOGGER = logging.getLogger("trans_to_min.scheduler")


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024
    return values


def _tree_pss(pid: int) -> int:
    """进程树PSS之和；进程已退出时返回0。"""

    children = defaultdict(list)
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children[parent].append(int(entry.name))
    total, stack = 0, [pid]
    while stack:
        current = stack.pop()
        stack.extend(children.get(current, []))
        try:
            for line in Path(f"/proc/{current}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    total += int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
    return total


@dataclass
class TaskState:
    name: str
    items: list[TaskGroup]
    datasets: tuple[str, ...]
    days: list[date]
    pending: list[date]
    rows: dict[date, int] = field(default_factory=dict)
    queue: list[str] = field(default_factory=list)
    jobs: dict[str, list[date]] = field(default_factory=dict)
    reference_days: list[date] = field(default_factory=list)


@dataclass
class Job:
    task: str
    key: str
    days: list[date]
    workers: int
    threads: int
    reserve: int
    process: subprocess.Popen
    report_dir: Path
    started: float = field(default_factory=time.monotonic)
    peak: int = 0


def _rows(plan: RunPlan, datasets: tuple[str, ...], day: date) -> int:
    return sum(
        pq.ParquetFile(input_path(plan.input_root, dataset, day)).metadata.num_rows
        for dataset in datasets
    )


def _done_days(plan: RunPlan, spec, items: list[TaskGroup], days: list[date]) -> set[date]:
    """全部指标都已发布的日期（1.4.0起已发布即视为可信，不比对凭证）。"""

    months = sorted({day.strftime("%Y%m") for day in days})
    done: set[date] | None = None
    for item in items:
        for name, root in metric_roots(plan, spec, item).items():
            published: set[date] = set()
            for month in months:
                published |= published_dates(month_path(root, name, month))
            done = published if done is None else done & published
    return done or set()


def prepare(plan: RunPlan) -> dict[str, TaskState]:
    """加载任务、确定待生成日期、按月成组并选出light校验的代表日。"""

    states = {}
    for spec in plan.tasks:
        items = load_task_groups(spec)
        datasets = task_datasets(items)
        days = [
            datetime.strptime(str(value), "%Y%m%d").date()
            for value in validate_matching_date_sets(plan.input_root, datasets, plan.start, plan.end)
        ]
        if plan.mode == "repair":
            pending = list(days)
        else:
            done = _done_days(plan, spec, items, days)
            pending = [day for day in days if day not in done]
        state = TaskState(spec.name, items, datasets, days, pending)
        state.rows = {day: _rows(plan, datasets, day) for day in pending}
        months: dict[str, list[date]] = defaultdict(list)
        for day in pending:
            months[day.strftime("%Y%m")].append(day)
        keys = sorted(months)
        for start in range(0, len(keys), plan.months_per_job):
            chunk = keys[start:start + plan.months_per_job]
            key = chunk[0] if len(chunk) == 1 else f"{chunk[0]}-{chunk[-1]}"
            state.jobs[key] = [day for month in chunk for day in months[month]]
        state.queue = list(state.jobs)
        if pending:
            # 代表日：最大输入日、首日、末日，最多3个。
            candidates = [max(pending, key=state.rows.get), pending[0], pending[-1]]
            state.reference_days = sorted(dict.fromkeys(candidates))[:LIGHT_REFERENCE_DAYS]
        states[spec.name] = state
    return states


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def final_cleanup(plan: RunPlan, keep_checkpoints: bool) -> dict:
    """清除暂存与各任务输出目录下的中间数据；有失败时保留参考检查点供排查。"""

    report = {"stage": source_stage.clear(), "removed": [], "kept": []}
    for spec in plan.tasks:
        root = plan.task_output(spec)
        for name in INTERMEDIATE_ENTRIES:
            path = root / name
            if not path.exists():
                continue
            if name == CHECKPOINT_DIRECTORY and keep_checkpoints:
                report["kept"].append(str(path))
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            report["removed"].append(str(path))
    return report


def run(plan: RunPlan, *, execute: bool) -> int:
    plan.state_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(plan.state_root / "scheduler.log", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    states = prepare(plan)
    siblings = {
        name: [other for other in states if other != name
               and set(states[other].datasets) & set(state.datasets)]
        for name, state in states.items()
    }
    summary = {
        name: {"days": len(state.days), "pending_days": len(state.pending),
               "jobs": len(state.jobs), "reference_days": [str(d) for d in state.reference_days],
               "shares_reads_with": siblings[name]}
        for name, state in states.items()
    }
    if not execute:
        print(json.dumps({"status": "planned", "mode": plan.mode, "validation": plan.validation,
                          "tasks": summary}, ensure_ascii=False, indent=2))
        return 0
    if plan.mode == "full":
        from .bench import check_report
        check_report(plan)

    state_path = plan.state_root / "state.json"
    saved = _load(state_path)
    calibration = saved.get("calibration", {})
    for name in states:
        calibration.setdefault(name, DEFAULT_BYTES_PER_ROW)
    record = {
        "framework_version": __version__, "plan": str(plan.path), "mode": plan.mode,
        "validation": plan.validation, "started_at": datetime.now(timezone.utc).isoformat(),
        "tasks": summary, "jobs": saved.get("jobs", {}), "calibration": calibration,
    }
    total = _meminfo()["MemTotal"]
    compute_budget = plan.memory_fraction * total - plan.source_stage_budget_gb * GB
    cpu = os.cpu_count() or 1
    running: dict[str, Job] = {}
    failed: list[str] = []
    stopping = threading.Event()
    started = time.monotonic()

    def handle_stop(signum, _frame):
        LOGGER.warning("收到信号%s：停止派发并终止已派发作业", signum)
        stopping.set()
        for job in running.values():
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    def per_worker(name: str, key: str) -> int:
        state = states[name]
        largest = max(state.rows[day] for day in state.jobs[key])
        return int(largest * calibration[name] * SAFETY)

    def launch(name: str, key: str, workers: int, threads: int, reserve: int) -> None:
        state = states[name]
        days = state.jobs[key]
        report_dir = plan.state_root / name / key
        if report_dir.exists():
            report_dir.rename(report_dir.with_name(f"{key}.attempt_{datetime.now():%Y%m%d_%H%M%S}"))
        report_dir.mkdir(parents=True)
        dates_file = report_dir / "dates.json"
        dates_file.write_text(json.dumps([day.isoformat() for day in days]), encoding="utf-8")
        references = [day for day in state.reference_days if day in days]
        command = [
            sys.executable, "-m", "trans_to_min", "worker", "--plan", str(plan.path),
            "--task", name, "--dates-file", str(dates_file),
            "--result", str(report_dir / "result.json"),
            "--date-workers", str(workers), "--block-threads", str(threads),
            "--reference-days", ",".join(day.isoformat() for day in references),
        ]
        if siblings[name]:
            command.append("--retain")
        with (report_dir / "stdout.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        running[name] = Job(name, key, days, workers, threads, reserve, process, report_dir)
        record["jobs"][f"{name}/{key}"] = {
            "status": "running", "dates": len(days), "date_workers": workers,
            "block_threads": threads, "reserve_gib": round(reserve / GB, 1),
            "started_at": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
        }
        LOGGER.info("派发 %s/%s 日期%d 日期进程%d 每日期线程%d 预留%.1fG",
                    name, key, len(days), workers, threads, reserve / GB)

    def finish(job: Job) -> None:
        result = _load(job.report_dir / "result.json")
        ok = job.process.returncode == 0 and result.get("status") == "complete"
        entry = record["jobs"][f"{job.task}/{job.key}"]
        peaks = [
            generation.get("observed_peak_rss_bytes") or 0
            for group in result.get("groups", {}).values()
            for generation in group.get("generation", [])
        ]
        largest = max(states[job.task].rows[day] for day in job.days)
        rows = sum(states[job.task].rows[day] for day in job.days)
        elapsed = time.monotonic() - job.started
        entry.update(
            status="complete" if ok else "failed", exit_code=job.process.returncode,
            elapsed_seconds=round(elapsed, 1), rows=rows,
            peak_date_process_gib=round(max(peaks, default=0) / GB, 2),
            tree_peak_pss_gib=round(job.peak / GB, 1),
            errors=result.get("errors", [])[:20],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if max(peaks, default=0) and largest:
            # 每行字节数按本作业单日期进程实测峰值÷最大日行数校准，可上可下。
            calibration[job.task] = max(peaks) / largest
            entry["bytes_per_row"] = round(calibration[job.task])
        LOGGER.info("%s %s/%s 用时%.0fs 单日期进程峰值%.1fG", "完成" if ok else "失败",
                    job.task, job.key, elapsed, max(peaks, default=0) / GB)
        if not ok:
            failed.append(f"{job.task}/{job.key}")
            states[job.task].queue.clear()

    stop_file = plan.state_root / "STOP"
    while not stopping.is_set():
        for name, job in list(running.items()):
            job.peak = max(job.peak, _tree_pss(job.process.pid))
            if job.process.poll() is not None:
                finish(job)
                del running[name]
                if siblings[name] and not any(
                    other in running and running[other].key == job.key for other in siblings[name]
                ):
                    source_stage.sweep(unheld=True)
        if stop_file.exists():
            if not running:
                LOGGER.warning("检测到STOP文件：不再派发，已运行作业均已结束")
                stopping.set()
                break
            time.sleep(POLL_SECONDS)
            continue
        if not running and not any(state.queue for state in states.values()):
            break
        reserved = sum(max(job.reserve, job.peak) for job in running.values())
        unrealized = sum(max(0, job.reserve - _tree_pss(job.process.pid)) for job in running.values())
        budget = min(compute_budget - reserved,
                     _meminfo()["MemAvailable"] - MEMORY_RESERVE_BYTES - unrealized)
        idle = [name for name, state in states.items() if name not in running and state.queue]
        for name in list(idle):
            # 共用数据集的任务对齐到同一月份，使源文件只读一次。
            mine = states[name].queue[0]
            for other in siblings[name]:
                queue = states[other].queue
                if queue and (queue[0] < mine or (queue[0] == mine and other in running)):
                    idle.remove(name)
                    break
        idle.sort(key=lambda name: -sum(states[name].rows[d] for k in states[name].queue
                                        for d in states[name].jobs[k]))
        for position, name in enumerate(idle):
            key = states[name].queue[0]
            each = per_worker(name, key)
            # 给本轮排在后面的空闲任务各留一个日期进程的内存。
            room = budget - sum(per_worker(other, states[other].queue[0]) for other in idle[position + 1:])
            workers = min(len(states[name].jobs[key]), cpu, int(room // each))
            if workers < 1:
                continue
            concurrent = sum(job.workers for job in running.values()) + workers
            threads = max(1, min(MAX_THREADS_PER_DATE, cpu // concurrent))
            states[name].queue.pop(0)
            try:
                launch(name, key, workers, threads, workers * each)
            except Exception as exc:
                LOGGER.error("派发失败 %s/%s：%s", name, key, exc)
                record["jobs"][f"{name}/{key}"] = {"status": "not_dispatched", "error": repr(exc)}
                failed.append(f"{name}/{key}")
                states[name].queue.clear()
                continue
            budget -= workers * each
        record.update(
            updated_at=datetime.now(timezone.utc).isoformat(),
            running=sorted(f"{job.task}/{job.key}" for job in running.values()),
            remaining={name: len(state.queue) for name, state in states.items()},
            failed=failed,
        )
        _save(state_path, record)
        time.sleep(POLL_SECONDS)

    stopped = stopping.is_set()
    record["status"] = "failed" if failed else "stopped" if stopped else "complete"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["wall_seconds"] = round(time.monotonic() - started, 1)
    if not stopped:
        record["cleanup"] = final_cleanup(plan, keep_checkpoints=bool(failed))
    record["report"] = _run_report(record, states)
    _save(state_path, record)
    _save(plan.state_root / "run_report.json", record["report"])
    LOGGER.info("调度结束：%s；失败%s", record["status"], failed)
    return 0 if record["status"] == "complete" else 1


def _run_report(record: dict, states: dict[str, TaskState]) -> dict:
    """一份精简运行报告：日期数、耗时、峰值、每行字节数、吞吐与失败作业。"""

    tasks = {}
    for name, state in states.items():
        jobs = {key: value for key, value in record["jobs"].items() if key.startswith(f"{name}/")}
        done = [value for value in jobs.values() if value.get("status") == "complete"]
        rows = sum(value.get("rows", 0) for value in done)
        tasks[name] = {
            "pending_days": len(state.pending),
            "completed_days": sum(value.get("dates", 0) for value in done),
            "rows": rows,
            "max_date_process_peak_gib": max(
                (value.get("peak_date_process_gib", 0) for value in done), default=0),
            "max_bytes_per_row": max((value.get("bytes_per_row", 0) for value in done), default=0),
            "failed_jobs": sorted(key for key, value in jobs.items() if value.get("status") == "failed"),
            "reference_days": [str(day) for day in state.reference_days],
        }
    wall = record.get("wall_seconds") or 0
    return {
        "status": record.get("status"), "mode": record.get("mode"),
        "validation": record.get("validation"), "framework_version": record.get("framework_version"),
        "wall_seconds": wall,
        "rows_per_second": round(sum(task["rows"] for task in tasks.values()) / wall) if wall else None,
        "tasks": tasks, "cleanup": record.get("cleanup"),
    }
