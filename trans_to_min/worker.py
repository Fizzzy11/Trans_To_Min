"""调度器派发的单个作业：一个任务在一组日期上的生成、抽样独立复算与结构检查。

由 ``python -m trans_to_min worker`` 启动，业务任务不直接调用。
"""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import time
import traceback

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import MINUTE_COUNT, RunConfig
from .group_runner import MinuteMetricGroupRunner
from .plan import RunPlan, load_task_groups, metric_roots
from .reference_check import validate_published
from .storage import month_path


def _config(plan: RunPlan, spec, date_workers: int, block_threads: int, retain: bool) -> RunConfig:
    return RunConfig(
        input_root=plan.input_root,
        output_root=plan.task_output(spec),
        date_workers=date_workers,
        adaptive_workers=False,
        block_threads=block_threads,
        resume=plan.mode != "repair",
        overwrite_dates=plan.mode == "repair",
        publish_while_computing=True,
        auto_memory_fraction=plan.memory_fraction,
        source_stage_budget_gb=plan.source_stage_budget_gb,
        source_stage_retain=retain,
    )


def structure_check(roots: dict[str, Path], days: list[date]) -> list[str]:
    """已发布月文件中每个日期都存在，且行数恰为股票数×242、无重复股票分钟。"""

    errors = []
    wanted = {day.isoformat() for day in days}
    for name, root in roots.items():
        for month in sorted({day.strftime("%Y%m") for day in days}):
            path = month_path(root, name, month)
            if not path.is_file():
                errors.append(f"{name}缺少月文件{path}")
                continue
            table = pq.read_table(path, columns=["trade_date", "symbol"])
            dates = pc.cast(table["trade_date"], "string")
            for day in sorted(wanted):
                if day[:7].replace("-", "") != month:
                    continue
                mask = pc.equal(dates, day)
                rows = pc.sum(mask).as_py() or 0
                symbols = pc.count_distinct(pc.filter(table["symbol"], mask)).as_py()
                if rows == 0:
                    errors.append(f"{name} {day}未发布")
                elif rows != symbols * MINUTE_COUNT:
                    errors.append(f"{name} {day}行数{rows}≠股票数{symbols}×242")
    return errors


def run_job(
    plan: RunPlan,
    task: str,
    days: list[date],
    reference_days: list[date],
    *,
    date_workers: int,
    block_threads: int,
    retain: bool,
) -> dict:
    spec = plan.task(task)
    items = load_task_groups(spec)
    full_validation = plan.validation == "full"
    checked = sorted(days) if full_validation else sorted(set(reference_days) & set(days))
    others = sorted(set(days) - set(checked))
    config = _config(plan, spec, date_workers, block_threads, retain or len(items) > 1)
    report = {
        "task": task, "dates": [day.isoformat() for day in sorted(days)],
        "validation": {"level": plan.validation, "dates": [d.isoformat() for d in checked]},
        "groups": {}, "errors": [],
    }
    for item in items:
        roots = metric_roots(plan, spec, item)
        group_report = {"generation": [], "validation": None, "structure_errors": []}
        for subset, reference in ((checked, item.reference), (others, None)):
            if not subset:
                continue
            result = MinuteMetricGroupRunner(item.group, config, roots, reference).run(subset)
            group_report["generation"].append({
                "dates": len(subset),
                "status": result["status"],
                "elapsed_seconds": result.get("elapsed_seconds"),
                "observed_peak_rss_bytes": max(
                    (entry.get("observed_peak_rss_bytes") or 0
                     for entry in result["parallelism"].get("history", [])), default=0),
                "block_stats": _block_stats(result),
            })
            if result["status"] != "complete":
                report["errors"].append(f"{item.group.name}生成状态{result['status']}")
        if checked:
            validation = validate_published(
                item.group, config, roots, checked, item.reference, workers=1)
            group_report["validation"] = {
                key: validation.get(key) for key in ("status", "level", "dates", "errors")}
            if validation.get("status") != "passed":
                report["errors"].append(f"{item.group.name}独立复算未通过：{validation.get('errors')}")
        group_report["structure_errors"] = structure_check(roots, sorted(days))
        report["errors"].extend(group_report["structure_errors"])
        report["groups"][item.group.name] = group_report
    report["status"] = "failed" if report["errors"] else "complete"
    return report


def _block_stats(result: dict) -> dict:
    """汇总各日期的分块统计；出现重排说明上游排序与约定不符。"""

    blocks = threads = minute_reordered = 0
    stock_reordered: set[str] = set()
    for metric in result.get("metrics", {}).values():
        for day in metric.get("days", {}).values():
            stats = day.get("block_stats")
            if stats:
                blocks = max(blocks, stats["blocks"])
                threads = max(threads, stats["threads"])
                minute_reordered += stats["minute_reordered_blocks"]
                stock_reordered.update(stats["stock_reordered_tables"])
        break
    return {"max_blocks": blocks, "max_threads": threads,
            "minute_reordered_blocks": minute_reordered,
            "stock_reordered_tables": sorted(stock_reordered)}


def main(arguments) -> int:
    from .plan import load_plan

    started = time.perf_counter()
    plan = load_plan(arguments.plan)
    days = [date.fromisoformat(text) for text in json.loads(Path(arguments.dates_file).read_text())]
    reference_days = [date.fromisoformat(text) for text in arguments.reference_days.split(",") if text]
    try:
        report = run_job(
            plan, arguments.task, days, reference_days,
            date_workers=arguments.date_workers, block_threads=arguments.block_threads,
            retain=arguments.retain,
        )
    except Exception as exc:
        report = {"task": arguments.task, "status": "failed", "errors": [repr(exc)],
                  "traceback": traceback.format_exc()}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    report["pid"] = os.getpid()
    path = Path(arguments.result)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return 0 if report["status"] == "complete" else 1
