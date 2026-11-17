"""效率验收：在隔离目录跑一个真实大日，报告耗时、峰值内存与每行字节数。

full模式启动前，调度器要求每个任务在bench_report目录下有``<任务名>.json``，且：
状态为passed（每行字节数不超过门槛）；报告晚于任务目录中最新修改的.py文件
（代码改过就要重跑bench）。只比较文件时间和数值，不做哈希。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq

from . import source_stage
from ._version import __version__
from .adapter import input_path
from .config import RunConfig
from .group_runner import MinuteMetricGroupRunner
from .plan import RunPlan, load_plan, load_task_groups, task_datasets
from .resources import process_peak_memory_bytes


# 每行字节数门槛：日期进程峰值RSS ÷ 当日该任务全部输入表的行数。
# 2026-11-17实测：分块示例组（成交表4指标）约124，pandas整天计算约196；
# 1.4.0四项基础任务约110~192。门槛取250，超过即视为写法有明显问题。
BYTES_PER_ROW_LIMIT = 250


def _newest_source(path: Path) -> float:
    times = [item.stat().st_mtime for item in path.rglob("*.py") if "__pycache__" not in item.parts]
    return max(times, default=0.0)


def check_report(plan: RunPlan) -> None:
    """full模式门禁：每个任务都要有达标且不早于其源码的bench报告。"""

    if plan.bench_report is None:
        raise ValueError("full模式须提供bench_report目录")
    problems = []
    for spec in plan.tasks:
        path = plan.bench_report / f"{spec.name}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            problems.append(f"{spec.name}缺少bench报告{path}")
            continue
        if report.get("status") != "passed":
            problems.append(
                f"{spec.name}的bench未达标：每行{report.get('bytes_per_row')}字节，"
                f"门槛{report.get('limit_bytes_per_row')}")
        if path.stat().st_mtime < _newest_source(spec.path):
            problems.append(f"{spec.name}的源码在bench之后修改过，须重跑bench")
    if problems:
        raise RuntimeError("full模式未通过效率门禁：" + "；".join(problems))


def run_bench(
    plan: RunPlan, task: str, day: date, work_root: Path, *, block_threads: int
) -> dict:
    spec = plan.task(task)
    items = load_task_groups(spec)
    datasets = task_datasets(items)
    rows = sum(
        pq.ParquetFile(input_path(plan.input_root, dataset, day)).metadata.num_rows
        for dataset in datasets
    )
    if work_root.exists():
        raise FileExistsError(f"隔离目录已存在：{work_root}")
    # bench用专属暂存目录，结束时整个删除，不影响其他作业保留的暂存副本。
    default = source_stage.default_root()
    stage_root = None if default is None else default.with_name(
        f"trans_to_min_bench_{os.getpid()}")
    config = RunConfig(
        input_root=plan.input_root, output_root=work_root, date_workers=1,
        adaptive_workers=False, block_threads=block_threads,
        source_stage_budget_gb=plan.source_stage_budget_gb,
        source_stage_root=None if stage_root is None else str(stage_root),
    )
    started = time.perf_counter()
    groups = {}
    try:
        for item in items:
            roots = {name: work_root / item.output_roots.get(name, "")
                     for name in item.group.metric_names}
            result = MinuteMetricGroupRunner(item.group, config, roots).run([day])
            groups[item.group.name] = {
                "status": result["status"],
                "block": bool(getattr(item.group, "stock_separable", False)),
                "whole_day_reason": getattr(item.group, "whole_day_reason", None),
            }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        if stage_root is not None:
            source_stage.clear(stage_root)
    elapsed = time.perf_counter() - started
    peak = process_peak_memory_bytes() or 0
    bytes_per_row = peak / rows if rows and peak else None
    passed = bytes_per_row is not None and bytes_per_row <= BYTES_PER_ROW_LIMIT
    return {
        "status": "passed" if passed else "failed",
        "task": task, "date": day.isoformat(), "datasets": list(datasets), "rows": rows,
        "elapsed_seconds": round(elapsed, 1),
        "rows_per_second": round(rows / elapsed) if elapsed else None,
        "block_threads": block_threads,
        "peak_rss_gib": round(peak / 2**30, 2),
        "bytes_per_row": round(bytes_per_row, 1) if bytes_per_row else None,
        "limit_bytes_per_row": BYTES_PER_ROW_LIMIT,
        "groups": groups, "framework_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main(arguments) -> int:
    plan = load_plan(arguments.plan)
    report = run_bench(
        plan, arguments.task, date.fromisoformat(arguments.date), Path(arguments.work_root),
        block_threads=arguments.block_threads,
    )
    directory = Path(arguments.report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arguments.task}.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1
