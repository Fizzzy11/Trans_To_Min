"""独立参考与生成共用一次源读取：日参考检查点与已发布月文件的逐值比对。

业务任务只提供一个“独立参考函数”（:class:`DayReference`）。框架在日期进程里
把同一份源字节先后交给参考函数和生产计算，把参考结果写成检查点；校验阶段只读
检查点和已发布的月文件，不再扫描源文件。检查点缺失或绑定不符（来源或参考代码
变化、续跑跳过的日期）时，框架自动回退为重新扫描来源，并在报告里标明来源。

参考函数必须与生产公式相互独立；框架只负责调度、绑定和比对，不参与计算。
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .adapter import input_path
from .config import MINUTE_COUNT, RunConfig
from .group import MinuteMetricGroup
from .provenance import source_contract
from .source_stage import acquire, config_options, file_identity
from .storage import month_path, output_schema


CHECKPOINT_DIRECTORY = ".reference_checkpoints"
CHECKPOINT_FORMAT = 1
ORIGIN_GENERATION = "generation"
ORIGIN_RESCAN = "validation_rescan"


@dataclass(frozen=True)
class DayReference:
    """业务提供的单日独立参考。

    ``function(trade_date, sources)`` 必须是模块级函数；``sources`` 为
    ``{数据集: SourceHandle}``，用 ``pq.ParquetFile(sources[数据集].open())`` 读取，
    不得自行按路径打开源文件。返回
    ``{"symbols": 升序股票列表, "values": {指标名: 股票×242数组}, "info": {...}}``；
    期望为NULL的位置：浮点指标用NaN，或任意类型用 ``numpy.ma.MaskedArray`` 的掩码
    （整数指标只能用掩码）；掩码随检查点保存并参与NULL位置比对。``tolerances`` 为 ``{指标名: (atol, rtol)}``，
    未列出的指标按位精确比较。
    """

    function: Callable
    identity: Mapping[str, object] = field(default_factory=dict)
    tolerances: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    level: str = "全部目标股票分钟指标独立公式逐值校验"

    def __post_init__(self) -> None:
        if not callable(self.function) or "<locals>" in getattr(self.function, "__qualname__", "<locals>"):
            raise ValueError("独立参考必须是可被子进程导入的模块级函数")
        json.dumps(dict(self.identity), sort_keys=True, allow_nan=False)
        for name, pair in self.tolerances.items():
            if len(pair) != 2 or min(pair) < 0:
                raise ValueError(f"{name}的容差必须是非负的(atol, rtol)")
        if not isinstance(self.level, str) or not self.level.strip():
            raise ValueError("独立参考必须声明非空的校验等级")


def reference_digest(reference: DayReference) -> dict:
    """参考实现身份：函数位置、同包全部源码摘要和业务声明的版本信息。"""

    module = sys.modules[reference.function.__module__]
    path = Path(module.__file__)
    files = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.parent.glob("*.py"))
    }
    return {
        "function": f"{reference.function.__module__}:{reference.function.__qualname__}",
        "package_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "identity": json.loads(json.dumps(dict(reference.identity), sort_keys=True)),
    }


def checkpoint_path(config: RunConfig, scope: str, trade_date: date) -> Path:
    return (
        Path(config.output_root) / CHECKPOINT_DIRECTORY / scope
        / f"{trade_date:%Y%m}" / f"{trade_date:%Y%m%d}.parquet"
    )


def _binding(reference, scope, trade_date, source_identities, value_types) -> dict:
    return json.loads(json.dumps({
        "format": CHECKPOINT_FORMAT,
        "scope": scope,
        "date": trade_date.isoformat(),
        "sources": {name: dict(identity) for name, identity in source_identities.items()},
        "reference": reference_digest(reference),
        "metrics": dict(value_types),
    }, sort_keys=True))


def compute_day_reference(
    reference: DayReference,
    scope: str,
    value_types: Mapping[str, str],
    trade_date: date,
    sources: Mapping,
    config: RunConfig,
    *,
    origin: str,
) -> dict:
    """运行独立参考并原子写入检查点；返回可写入报告的摘要。"""

    started = time.perf_counter()
    result = reference.function(trade_date, sources)
    symbols = list(result["symbols"])
    if any(not isinstance(value, str) for value in symbols) or any(
        left >= right for left, right in zip(symbols, symbols[1:])
    ):
        raise ValueError("独立参考的股票列表必须是严格升序的字符串")
    missing = sorted(set(value_types) - set(result["values"]))
    if missing:
        raise ValueError(f"独立参考缺少指标：{missing}")
    columns = {"symbol": pa.array(symbols, type=pa.string())}
    for name, value_type in value_types.items():
        values = result["values"][name]
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype(value_type)
                or values.shape != (len(symbols), MINUTE_COUNT)):
            raise ValueError(f"独立参考{name}必须是{value_type}的股票×{MINUTE_COUNT}数组")
        # 掩码数组的掩码即期望NULL位置，写成Arrow空值随检查点保存，不能丢。
        mask = (np.ma.getmaskarray(values).reshape(-1)
                if isinstance(values, np.ma.MaskedArray) else None)
        columns[name] = pa.FixedSizeListArray.from_arrays(
            pa.array(np.ascontiguousarray(np.ma.getdata(values)).reshape(-1), mask=mask),
            MINUTE_COUNT,
        )
    identities = {name: handle.identity for name, handle in sources.items()}
    info = json.loads(json.dumps(result.get("info", {}), allow_nan=False, default=str))
    info["reference_seconds"] = round(time.perf_counter() - started, 3)
    metadata = {
        b"reference_binding": json.dumps(
            _binding(reference, scope, trade_date, identities, value_types), sort_keys=True
        ).encode(),
        b"reference_origin": origin.encode(),
        b"reference_info": json.dumps(info, sort_keys=True).encode(),
    }
    table = pa.table(columns).replace_schema_metadata(metadata)
    path = checkpoint_path(config, scope, trade_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd", compression_level=1,
                       use_dictionary=False, write_statistics=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"origin": origin, "elapsed_seconds": round(time.perf_counter() - started, 3), "symbols": len(symbols),
            "bytes": path.stat().st_size, "info": info}


def _checkpoint_header(path: Path, binding: dict) -> dict | None:
    """检查点存在且绑定与当前来源、参考实现一致时返回其来源信息。"""

    try:
        metadata = pq.read_schema(path).metadata or {}
        if json.loads(metadata[b"reference_binding"]) != binding:
            return None
        return {"origin": metadata[b"reference_origin"].decode(),
                "info": json.loads(metadata[b"reference_info"])}
    except (OSError, KeyError, ValueError, pa.ArrowException):
        return None


def _load_checkpoint(path: Path, binding: dict, value_types: Mapping[str, str]) -> dict:
    header = _checkpoint_header(path, binding)
    if header is None:
        raise ValueError(f"参考检查点缺失或绑定不符：{path}")
    table = pq.read_table(path, use_threads=True)
    symbols = table["symbol"].to_pylist()
    values, nulls = {}, {}
    for name, value_type in value_types.items():
        stored = table[name].combine_chunks().flatten()
        if stored.null_count:
            # 参考声明的期望NULL位置；先取出掩码再填零，避免整数列被转成浮点。
            nulls[name] = stored.is_null().to_numpy(zero_copy_only=False).reshape(
                len(symbols), MINUTE_COUNT)
            stored = stored.fill_null(0)
        flat = stored.to_numpy(zero_copy_only=False)
        if flat.dtype != np.dtype(value_type) or len(flat) != len(symbols) * MINUTE_COUNT:
            raise ValueError(f"参考检查点内容损坏：{path}")
        values[name] = flat.reshape(len(symbols), MINUTE_COUNT)
    return {"symbols": symbols, "values": values, "nulls": nulls, **header}


def _source_identities(group: MinuteMetricGroup, config: RunConfig, trade_date: date) -> dict:
    return {
        dataset: file_identity(input_path(Path(config.input_root), dataset, trade_date))
        for dataset in group.inputs
    }


def _value_types(group: MinuteMetricGroup) -> dict[str, str]:
    return {metric.name: metric.value_type for metric in group.metrics}


def _rescan_day(arguments) -> tuple[date, dict]:
    """回退路径：没有可用检查点的日期重新读取来源（仍经共享暂存）。"""

    group, reference, config, trade_date = arguments
    options = config_options(config)
    sources = {}
    try:
        for dataset in group.inputs:
            sources[dataset] = acquire(
                input_path(Path(config.input_root), dataset, trade_date), **options
            )
        return trade_date, compute_day_reference(
            reference, group.name, _value_types(group), trade_date, sources, config,
            origin=ORIGIN_RESCAN,
        )
    finally:
        for handle in sources.values():
            handle.close()


def ensure_checkpoints(
    group: MinuteMetricGroup,
    config: RunConfig,
    dates: Sequence[date],
    reference: DayReference,
    *,
    workers: int = 1,
) -> dict[date, dict]:
    """保证每个日期都有绑定当前来源的检查点，返回逐日绑定。"""

    value_types = _value_types(group)
    bindings = {
        day: _binding(reference, group.name, day, _source_identities(group, config, day), value_types)
        for day in dates
    }
    missing = [
        day for day in dates
        if _checkpoint_header(checkpoint_path(config, group.name, day), bindings[day]) is None
    ]
    jobs = [(group, reference, config, day) for day in missing]
    if len(jobs) <= 1 or workers <= 1:
        for job in jobs:
            _rescan_day(job)
    else:
        # spawn避免继承生成阶段的大块内存；子进程只做流式参考。
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                 mp_context=mp.get_context("spawn")) as executor:
            list(executor.map(_rescan_day, jobs))
    return bindings


def clear_checkpoints(config: RunConfig, scope: str, dates: Sequence[date]) -> int:
    """删除指定日期的检查点，并清掉因此变空的目录。"""

    removed = 0
    parents = set()
    for day in dates:
        path = checkpoint_path(config, scope, day)
        if path.is_file():
            path.unlink()
            removed += 1
        parents.add(path.parent)
    root = Path(config.output_root) / CHECKPOINT_DIRECTORY
    for directory in sorted(parents) + [root / scope, root]:
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def _minute_labels(trade_date: date) -> np.ndarray:
    """242个分钟结束标签的UTC毫秒；独立于生产分钟工具。"""

    import pandas as pd

    axis = np.r_[np.arange(570, 691), np.arange(780, 901)]
    midnight = int(pd.Timestamp(trade_date, tz="Asia/Shanghai").value // 1_000_000)
    return midnight + axis.astype(np.int64) * 60_000


def _compare_metric_month(
    group: MinuteMetricGroup,
    name: str,
    path: Path,
    days: Sequence[date],
    expected: Mapping[date, dict],
    expected_keys: Mapping[date, dict],
    input_root: Path,
    tolerance: tuple[float, float] | None,
) -> tuple[list[dict], list[str], dict]:
    """一个指标的一个月文件只读一次，逐日按期望键序列直接比对。"""

    metric = group.metric(name)
    coverage: list[dict] = []
    errors: list[str] = []
    before = file_identity(path)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if not schema.equals(output_schema(name, metric.value_type), check_metadata=False):
        raise ValueError(f"{name}四列物理Schema不符")
    table = parquet.read(use_threads=True)
    table = table.filter(pc.is_in(table["trade_date"], value_set=pa.array(list(days), type=pa.date32())))
    if any(table[column].null_count for column in ("trade_date", "minute", "symbol")):
        raise ValueError(f"{name}输出键含NULL")
    date_keys = pc.cast(table["trade_date"], pa.int32()).to_numpy(zero_copy_only=False)
    if len(date_keys) > 1 and (np.diff(date_keys) < 0).any():
        raise ValueError(f"{name}请求日期输出未按日期、分钟、股票升序")
    is_float = metric.value_type == "float64"
    for day in days:
        ordinal = (day - date(1970, 1, 1)).days
        start = int(np.searchsorted(date_keys, ordinal, "left"))
        end = int(np.searchsorted(date_keys, ordinal, "right"))
        day_table = table.slice(start, end - start)
        wanted_keys = expected_keys[day]
        count = len(expected[day]["symbols"])
        item = dict(metric=name, date=day.isoformat(), expected_rows=count * MINUTE_COUNT,
                    actual_rows=end - start, missing_keys=0, extra_keys=0, null_mismatches=0,
                    nonfinite_values=0, different_values=0, max_abs_error=0.0, max_rel_error=0.0,
                    error_statistics_overflow=False)
        minute = pc.cast(day_table["minute"], pa.int64()).to_numpy(zero_copy_only=False)
        # 键序列与期望逐位相同，等价于“键集合相同、无重复且按分钟、股票升序”。
        key_match = (
            day_table.num_rows == count * MINUTE_COUNT
            and np.array_equal(minute, wanted_keys["minutes"])
            and (count == 0 or pc.all(pc.equal(
                day_table["symbol"].combine_chunks(), wanted_keys["symbols"])).as_py())
        )
        if not key_match:
            symbols = day_table["symbol"].to_numpy(zero_copy_only=False)
            actual_keys = set(zip(symbols, minute))
            desired_keys = set(zip(
                wanted_keys["symbols"].to_numpy(zero_copy_only=False), wanted_keys["minutes"]))
            if actual_keys == desired_keys and len(actual_keys) == len(symbols):
                errors.append(f"{name}/{day}输出未按分钟、股票升序")
            item.update(missing_keys=len(desired_keys - actual_keys),
                        extra_keys=len(actual_keys - desired_keys),
                        duplicate_keys=len(symbols) - len(actual_keys))
            errors.append(f"{name}/{day}股票分钟键不一致")
        else:
            # 参考值为股票×分钟；输出为分钟×股票，转置后逐位对应。
            wanted = expected[day]["values"][name].T.reshape(-1)
            column = day_table[name]
            nulls = column.is_null().to_numpy(zero_copy_only=False)
            desired_nulls = np.isnan(wanted) if is_float else np.zeros(len(wanted), dtype=bool)
            declared_nulls = expected[day]["nulls"].get(name)
            if declared_nulls is not None:
                desired_nulls = desired_nulls | declared_nulls.T.reshape(-1)
            item["null_mismatches"] = int(np.count_nonzero(nulls != desired_nulls))
            actual = column.fill_null(0).to_numpy(zero_copy_only=False)
            # Arrow有效位区分合法NULL和物理非空NaN；全部非空值先做有限性检查。
            nonfinite = ~nulls & ~np.isfinite(actual)
            item["nonfinite_values"] = int(np.count_nonzero(nonfinite))
            valid = ~nulls & ~desired_nulls & ~nonfinite
            a, b = actual[valid], wanted[valid]
            if tolerance is None or not is_float:
                different = a != b
            else:
                different = ~np.isclose(a, b, atol=tolerance[0], rtol=tolerance[1])
            item["different_values"] = int(np.count_nonzero(different)) + item["nonfinite_values"]
            if len(a):
                # 极端有限输出也可能使绝对／相对误差超float64；用JSON NULL明确记录。
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    error = np.abs(a.astype(np.longdouble) - b.astype(np.longdouble))
                    nonzero = b != 0
                    statistics = {
                        "max_abs_error": error.max(),
                        "max_rel_error": (np.max(error[nonzero] / np.abs(b[nonzero]))
                                          if nonzero.any() else 0.0),
                    }
                for key, statistic in statistics.items():
                    if np.isfinite(statistic) and statistic <= np.finfo(np.float64).max:
                        item[key] = float(statistic)
                    else:
                        item[key] = None
                        item["error_statistics_overflow"] = True
            if item["nonfinite_values"]:
                errors.append(f"{name}/{day}含物理非空NaN或Inf输出")
            if item["null_mismatches"] or item["different_values"]:
                errors.append(f"{name}/{day}NULL或公式值不一致")
        coverage.append(item)
    if file_identity(path) != before:
        raise ValueError("校验期间输出快照变化")
    output = dict(metric=name, path=str(Path(path).resolve()), identity=before, body_reads=1)
    return coverage, errors, output


def validate_published(
    group: MinuteMetricGroup,
    config: RunConfig,
    output_roots: Mapping[str, Path],
    dates: Sequence[date],
    reference: DayReference,
    *,
    metric_names: Sequence[str] | None = None,
    workers: int = 1,
    clear_on_pass: bool = True,
) -> dict:
    """用日参考检查点逐值核对已发布月文件；返回可直接写入任务报告的结果。

    ``workers`` 只用于缺检查点日期的回退重扫。全部通过且 ``clear_on_pass`` 时删除
    本次日期的检查点；失败时保留检查点以便排查和重跑。
    """

    started = time.monotonic()
    days = sorted(dates)
    selected = [metric.name for metric in group.select_metrics(metric_names)]
    value_types = _value_types(group)
    report = dict(
        status="running", level=reference.level, dates=[day.isoformat() for day in days],
        metrics=selected, reference=reference_digest(reference), validation_workers=max(1, int(workers)),
        sources=[], coverage=[], outputs=[], errors=[], reference_origins={},
    )
    try:
        if not days or len(set(days)) != len(days):
            raise ValueError("独立校验需要非空且不重复的明确日期")
        if not selected:
            raise ValueError("独立校验需要至少一个指标")
        # 只执行指标声明的报价版本门禁；返回的来源身份不保存、不比对。
        for metric in group.select_metrics(metric_names):
            if getattr(metric, "required_price_policy_version", None) is not None:
                for day in days:
                    source_contract(metric, day, config.input_root)
        bindings = ensure_checkpoints(group, config, days, reference, workers=max(1, int(workers)))
        input_root = Path(config.input_root)
        for month in sorted({day.strftime("%Y%m") for day in days}):
            month_days = [day for day in days if day.strftime("%Y%m") == month]
            expected, expected_keys = {}, {}
            for day in month_days:
                loaded = _load_checkpoint(
                    checkpoint_path(config, group.name, day), bindings[day], value_types)
                expected[day] = loaded
                symbols = loaded["symbols"]
                expected_keys[day] = dict(
                    minutes=np.repeat(_minute_labels(day), len(symbols)),
                    symbols=(pa.array(np.tile(np.array(symbols, dtype=object), MINUTE_COUNT),
                                      type=pa.string())
                             if symbols else pa.array([], type=pa.string())),
                )
                report["sources"].append(dict(
                    date=day.isoformat(), origin=loaded["origin"],
                    sources=bindings[day]["sources"], symbols=len(symbols), info=loaded["info"]))
                report["reference_origins"][loaded["origin"]] = (
                    report["reference_origins"].get(loaded["origin"], 0) + 1)

            def check(name):
                return _compare_metric_month(
                    group, name, month_path(Path(output_roots[name]), name, month),
                    month_days, expected, expected_keys, input_root,
                    reference.tolerances.get(name))

            # 指标间互不依赖，线程并行（Arrow读取与NumPy比较大多释放GIL）；结果按声明顺序汇总。
            with ThreadPoolExecutor(max_workers=max(1, min(8, len(selected)))) as executor:
                for coverage, errors, output in executor.map(check, selected):
                    report["coverage"].extend(coverage)
                    report["errors"].extend(errors)
                    report["outputs"].append(output)
        for day in days:
            if _source_identities(group, config, day) != bindings[day]["sources"]:
                raise ValueError("校验前后来源身份变化")
        for output in report["outputs"]:
            if file_identity(output["path"]) != output["identity"]:
                raise ValueError("校验前后输出身份变化")
        report["status"] = "failed" if report["errors"] else "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    if report["status"] == "passed" and clear_on_pass:
        report["checkpoints_removed"] = clear_checkpoints(config, group.name, days)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report
