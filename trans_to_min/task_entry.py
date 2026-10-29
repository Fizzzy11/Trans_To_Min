"""全量与日更共用的单指标任务协议；业务就绪和校验逻辑由调用方提供。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Sequence
import uuid

from ._version import __version__
from .config import RunConfig
from .metric import MinuteMetric
from .provenance import source_contract, validate_resume
from .runner import MinuteConversionRunner
from .storage import metric_root, month_path


class InputsNotReady(RuntimeError):
    """业务确认上游尚未完成发布或质检，允许外部调度稍后重试。"""


class _TaskLocked(RuntimeError):
    """同一输出指标已有标准入口任务占用。"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="全量、每日增量和指定日期修订共用入口；默认仅预演")
    parser.add_argument("--mode", choices=("full", "daily", "repair"), required=True,
                        help="历史全量、日更补缺或显式修订")
    dates = parser.add_mutually_exclusive_group(required=True)
    dates.add_argument("--date", help="明确的单个交易日，格式YYYY-MM-DD")
    dates.add_argument("--dates-file", type=Path, help="可信调度提供的JSON日期数组")
    parser.add_argument("--input-root", type=Path, required=True, help="逐笔输入根目录")
    parser.add_argument("--output-root", type=Path, required=True, help="指标所属类别目录")
    parser.add_argument("--result-json", type=Path, required=True, help="每次唯一、不可覆盖的任务报告")
    parser.add_argument("--execute", action="store_true", help="显式执行；省略时不读取源数据、不生成结果")
    parser.add_argument("--date-workers", type=int, default=0, help="日期进程数；0为框架自动建议")
    parser.add_argument("--arrow-threads-per-worker", type=int, default=4, help="每日期进程Arrow线程数")
    parser.add_argument("--auto-memory-fraction", type=float, default=0.70, help="自动建议使用的可用内存比例")
    return parser


def _dates(args: argparse.Namespace) -> list[date]:
    values = ([args.date] if args.date is not None
              else json.loads(args.dates_file.read_text(encoding="utf-8")))
    if not isinstance(values, list):
        raise ValueError("日期文件必须是YYYY-MM-DD字符串数组")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("交易日必须是YYYY-MM-DD字符串")
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("交易日必须采用YYYY-MM-DD格式")
        result.append(parsed)
    if len(result) != len(set(result)):
        raise ValueError("日期清单不得重复，须由调用方确认授权范围")
    if not result and args.mode != "daily":
        raise ValueError("全量和修订模式不能使用空日期清单")
    return sorted(result)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _report_path(args: argparse.Namespace, config: RunConfig) -> Path:
    source, output = config.input_root.resolve(), config.output_root.resolve()
    if _inside(output, source) or _inside(source, output):
        raise ValueError("输入与输出目录不得相同或相互包含")
    report = args.result_json.resolve()
    if report.suffix.lower() != ".json":
        raise ValueError("任务报告必须使用.json后缀")
    if _inside(report, source) or _inside(report, output):
        raise ValueError("任务报告必须保存在输入及生产输出之外的业务工作区")
    if report.exists():
        raise ValueError("任务报告已存在，请使用本次唯一的新路径")
    return report


def _save_report(path: Path, payload: dict) -> None:
    """报告每次原子替换；路径已由本次调用独占创建。"""
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _task_lock(root: Path, invocation_id: str):
    """仅协调使用同一协议的入口，异常退出留下的锁须人工核实后处理。"""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".task.lock"
    owner = json.dumps({"invocation_id": invocation_id, "pid": os.getpid()}, sort_keys=True)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise _TaskLocked(f"指标正在运行或存在待核实的残留锁：{path}") from exc
    try:
        with stream:
            stream.write(owner)
        yield
    finally:
        if path.read_text(encoding="utf-8") != owner:
            raise RuntimeError("运行锁身份改变，拒绝删除其他任务的锁")
        path.unlink()


def _validation_identity(metric: MinuteMetric, config: RunConfig, dates: list[date]) -> dict:
    """校验前后检查来源及月文件身份；不替代业务校验器内部的稳定快照读取。"""
    months = sorted({day.strftime("%Y%m") for day in dates})
    outputs = {}
    for month in months:
        path = month_path(config.output_root, metric.name, month)
        requested = [day for day in dates if day.strftime("%Y%m") == month]
        validate_resume(path, metric, requested, config.input_root)
        stat = path.stat()
        outputs[month] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return {
        "outputs": outputs,
        "inputs": {day.isoformat(): source_contract(metric, day, config.input_root) for day in dates},
    }


def _check_validation(result: dict, dates: list[date]) -> None:
    expected = [day.isoformat() for day in dates]
    if not isinstance(result, dict) or result.get("status") != "passed":
        raise ValueError("业务校验必须显式返回status=passed")
    actual = result.get("dates")
    if (not isinstance(actual, list) or any(not isinstance(value, str) for value in actual)
            or sorted(actual) != expected):
        raise ValueError("业务校验日期必须精确覆盖本次请求，不得漏日、重复或扩大范围")
    if not isinstance(result.get("level"), str) or not result["level"].strip():
        raise ValueError("业务校验必须声明非空的校验等级level")
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def run_task_cli(
    metric: MinuteMetric,
    *,
    check_ready: Callable | None = None,
    validate_output: Callable | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """执行统一单指标入口，返回可由主脚本SystemExit使用的退出码。

    两个业务钩子均接收(metric, config, dates)。就绪检查未通过应抛InputsNotReady；
    校验须返回包含status、dates、level的JSON对象。此入口不实现日历、定时器、总
    调度器或交错校验，也不改变公式、分钟轴和按月保存格式。
    """
    try:
        args = _parser().parse_args(argv)
        dates = _dates(args)
        config = RunConfig(
            input_root=args.input_root.resolve(), output_root=args.output_root.resolve(),
            date_workers=args.date_workers, arrow_threads_per_worker=args.arrow_threads_per_worker,
            auto_memory_fraction=args.auto_memory_fraction,
            resume=args.mode != "repair", overwrite_dates=args.mode == "repair",
            # 全量保留上月发布与下月计算重叠；业务校验始终等本次生成完成。
            publish_while_computing=args.mode == "full",
        )
        root = metric_root(config.output_root, metric.name)
        path = _report_path(args, config)
        if args.execute and dates and (not callable(check_ready) or not callable(validate_output)):
            raise ValueError("执行前必须接入业务上游就绪检查和独立校验，不能以空钩子冒充通过")
        path.parent.mkdir(parents=True, exist_ok=True)
        # 排它创建阻止两个调用抢用相同报告路径；之后只更新本次拥有的报告。
        with path.open("x", encoding="utf-8") as stream:
            stream.write("{}\n")
    except SystemExit as exc:
        return int(exc.code)
    except (ValueError, OSError) as exc:
        print(f"任务参数错误：{exc}", file=sys.stderr)
        return 2

    payload = {
        "protocol_version": 1, "framework_version": __version__,
        "invocation_id": uuid.uuid4().hex, "mode": args.mode, "metric": metric.name,
        "requested_dates": [day.isoformat() for day in dates],
        "input_root": str(config.input_root), "output_root": str(config.output_root),
        "result_json": str(path), "execute": args.execute,
        "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
        "status": "running", "generation": None, "validation": None,
        "error": None, "exit_code": None,
    }
    code = 1
    try:
        _save_report(path, payload)
        if not args.execute:
            payload["status"], code = "planned", 0
        elif not dates:
            payload["status"], code = "no_work", 0
        else:
            # 父进程业务日志转stderr；子进程仍可能输出日志，机器接口只认独立报告文件。
            with redirect_stdout(sys.stderr), _task_lock(root, payload["invocation_id"]):
                if check_ready(metric, config, list(dates)) is not None:
                    raise ValueError("就绪检查成功须正常返回None；未就绪须抛InputsNotReady")
                payload["generation"] = MinuteConversionRunner(metric, config).run(dates)
                _save_report(path, payload)
                before = _validation_identity(metric, config, dates)
                validation = validate_output(metric, config, list(dates))
                _check_validation(validation, dates)
                if _validation_identity(metric, config, dates) != before:
                    raise RuntimeError("校验期间输入或月份文件版本改变，校验结论失效")
                payload["validation"] = validation
            payload["status"], code = "complete", 0
    except InputsNotReady as exc:
        payload["status"], payload["error"], code = "not_ready", str(exc), 3
    except _TaskLocked as exc:
        payload["status"], payload["error"], code = "locked", str(exc), 4
    except Exception as exc:
        payload["status"], payload["error"], code = "failed", f"{type(exc).__name__}: {exc}", 1
    finally:
        payload["exit_code"] = code
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _save_report(path, payload)
        except Exception as exc:
            payload["status"], payload["exit_code"], code = "failed", 1, 1
            payload["error"] = f"任务报告落盘失败：{exc}"
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return code
