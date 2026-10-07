"""隔离三表到分钟的小样本验收，不查询数据库、不触碰正式输出。

订单字典与业务参考公式不调用生产订单视图或聚合函数，但仍共享L2InputAdapter
和框架分钟映射，因此不是完全独立的raw到分钟全链验证。实际运行会生成隔离分钟
文件；只读契约函数和合成单测不生成真实数据。不得把小样本通过当作历史重建授权。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from datetime import date
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
# 只有直接运行隔离入口才引导源码路径；作为helper导入时必须保留wheel验收环境。
if __name__ == "__main__" and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trans_to_min.adapter import L2InputAdapter, input_path  # noqa: E402
from trans_to_min.minute import (  # noqa: E402
    assign_minute_codes,
    assign_timestamp_minute_codes,
    minute_labels,
)


VERIFY_DATES = (date(2021, 6, 9), date(2025, 1, 9), date(2026, 9, 24))
VERIFY_SYMBOLS = (
    "000001.XSHE", "000002.XSHE", "300750.XSHE",
    "600000.XSHG", "600519.XSHG", "601318.XSHG",
)
VALUE_TYPES = {"float64": pa.float64(), "int64": pa.int64(), "uint64": pa.uint64()}
REFERENCE_BOUNDARY = "订单及业务公式独立；共享L2InputAdapter和框架分钟映射，非完全独立全链验证"


def validate_output_schema(schema: pa.Schema, metric_name: str, value_type: str) -> None:
    """按明确的四列物理契约检查类型与可空标记，不调用生产Schema生成器。"""

    if value_type not in VALUE_TYPES:
        raise ValueError(f"验收不支持指标类型：{value_type}")
    expected = pa.schema([
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("minute", pa.timestamp("ms", tz="Asia/Shanghai"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field(metric_name, VALUE_TYPES[value_type], nullable=True),
    ])
    if schema.names != expected.names:
        raise ValueError(f"分钟输出必须依次为四列：{expected.names}，实际={schema.names}")
    for actual, wanted in zip(schema, expected):
        if actual.type != wanted.type or actual.nullable != wanted.nullable:
            raise ValueError(f"分钟物理Schema不一致：字段{wanted.name}，期望={wanted}，实际={actual}")


def validate_day_output(
    table: pa.Table,
    trade_date: date,
    expected_symbols: Iterable[str],
    metric_name: str,
    value_type: str,
    expected_values: Mapping[tuple[str, int], float | int | None],
    empty_value: float | int | None,
) -> dict:
    """只读核验单日物理表、完整股票分钟网格、逐键参考值及真实NULL位图。

    expected_values是独立公式的稀疏结果，键为股票与0至241分钟编码；没有事件的键
    才按empty_value补齐。参考中显式NULL不得与已知零或真正无事件混淆。
    """

    validate_output_schema(table.schema, metric_name, value_type)
    for key in ("trade_date", "minute", "symbol"):
        if table.column(key).null_count:
            raise ValueError(f"分钟唯一键字段不允许物理NULL：{key}")
    symbols = set(expected_symbols)
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError("期望股票集合必须是非空字符串，不能把空身份转换成股票")
    actual_dates = set(table.column("trade_date").to_pylist())
    if table.num_rows and actual_dates != {trade_date}:
        raise ValueError(f"实际trade_date与请求交易日不一致：{actual_dates}，期望={trade_date}")
    actual_symbols = set(table.column("symbol").to_pylist())
    if actual_symbols != symbols:
        raise ValueError(f"实际股票集合不一致：缺失={sorted(symbols - actual_symbols)}，额外={sorted(actual_symbols - symbols)}")

    frame = table.to_pandas()
    if str(frame["minute"].dt.tz) != "Asia/Shanghai":
        raise ValueError("分钟结果必须使用Asia/Shanghai时区")
    if frame.duplicated(["trade_date", "minute", "symbol"]).any():
        raise ValueError("分钟结果存在重复唯一键：trade_date + minute + symbol")
    if table.num_rows != 242 * len(symbols):
        raise ValueError(f"分钟行数不完整：期望={242 * len(symbols)}，实际={table.num_rows}")
    labels = minute_labels(trade_date)
    if len(labels) != 242 or str(labels.tz) != "Asia/Shanghai":
        raise ValueError("参考分钟轴必须是242个Asia/Shanghai结束标签")
    codes = labels.get_indexer(frame["minute"])
    if (codes < 0).any():
        raise ValueError("分钟结果包含请求交易日242标签之外的时间")
    for symbol in symbols:
        symbol_codes = codes[frame["symbol"].eq(symbol).to_numpy()]
        if not np.array_equal(np.sort(symbol_codes), np.arange(242)):
            raise ValueError(f"股票分钟标签不完整：{symbol}必须恰好包含全部242标签")
    for key in expected_values:
        if (not isinstance(key, tuple) or len(key) != 2 or key[0] not in symbols
                or not isinstance(key[1], (int, np.integer)) or not 0 <= key[1] < 242):
            raise ValueError(f"独立参考包含股票分钟轴之外的键：{key!r}")

    fill = np.nan if empty_value is None else empty_value
    values = [expected_values.get((symbol, int(code)), fill)
              for symbol, code in zip(frame["symbol"], codes)]
    expected_nulls = np.asarray([pd.isna(value) for value in values], dtype=bool)
    actual_nulls = table.column(metric_name).is_null().to_numpy(zero_copy_only=False)
    if not np.array_equal(actual_nulls, expected_nulls):
        raise ValueError("指标物理NULL位图与独立参考不一致，禁止将未知金额填零或把已知零置空")
    known = ~expected_nulls
    actual_values = table.column(metric_name).to_pylist()
    if value_type in ("int64", "uint64"):
        # 整数逐值精确比较，不能因转float64或浮点容差漏掉大整数差异。
        for position in np.flatnonzero(known):
            wanted = values[position]
            if not np.isfinite(wanted) or wanted != int(wanted) or actual_values[position] != int(wanted):
                raise ValueError("整数指标与独立公式不完全一致")
    else:
        wanted = np.asarray([values[position] for position in np.flatnonzero(known)], dtype=float)
        actual = np.asarray([actual_values[position] for position in np.flatnonzero(known)], dtype=float)
        if not np.isfinite(wanted).all() or not np.isfinite(actual).all():
            raise ValueError("非空指标或独立参考包含NaN/Inf，不能冒充物理NULL")
        try:
            np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-8)
        except AssertionError as exc:
            raise ValueError("指标值与独立公式不一致") from exc
    return {
        "rows": table.num_rows,
        "symbols": len(symbols),
        "null_values": int(actual_nulls.sum()),
        "output_contract": "passed",
        "physical_nulls": "passed",
    }


def validate_isolated_paths(
    input_root: Path, output_root: Path, report: Path, *, root: Path = ROOT,
) -> tuple[Path, Path, Path]:
    """限制隔离路径、不覆盖已有报告或输出，并禁止写入输入树。"""

    root = Path(root).resolve()
    paths = tuple(Path(path).resolve() for path in (input_root, output_root, report))
    for path in paths:
        if path == root or not path.is_relative_to(root):
            raise ValueError("只允许当前候选内的隔离输入、输出与报告")
    source, destination, summary = paths
    if (source == destination or destination.is_relative_to(source)
            or source.is_relative_to(destination) or summary.is_relative_to(source)):
        raise ValueError("隔离输出和报告不得覆盖或包含输入树")
    if (summary == destination or destination.is_relative_to(summary)
            or source.is_relative_to(summary)):
        raise ValueError("隔离报告必须是独立文件路径，不能作为输入或输出目录")
    if not source.is_dir():
        raise ValueError(f"隔离输入目录不存在：{source}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("隔离输出必须是全新或空目录，不覆盖既有分钟结果")
    if summary.exists():
        raise ValueError("隔离报告已存在，必须选择新的报告文件")
    return source, destination, summary


def preflight_verify_inputs(input_root: Path) -> dict:
    """生成前只读预检固定三日六股的九份输入，仅分批投影symbol/date。

    不扫描其他日期、不输出明细，不凭metadata代替实际日期和股票集合检查；全池或
    错日期输入必须在任何业务导入、MinuteMetricGroupRunner.run之前拒绝。
    """

    expected_symbols = set(VERIFY_SYMBOLS)
    checked = 0
    for day in VERIFY_DATES:
        for dataset in ("weituo", "chengjiao", "chedan"):
            path = input_path(input_root, dataset, day)
            if not path.is_file():
                raise ValueError(f"隔离样本输入文件不存在：{path}")
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            metadata = schema.metadata or {}
            for key, wanted in ((b"price_policy_version", b"3"), (b"format_version", b"1")):
                if metadata.get(key) != wanted:
                    raise ValueError(f"隔离样本{key.decode()}必须为{wanted.decode()}：{path}")
            if not {"symbol", "date"}.issubset(schema.names):
                raise ValueError(f"隔离样本缺少物理symbol/date：{path}")
            if schema.field("symbol").type != pa.string() or schema.field("date").type != pa.date32():
                raise ValueError(f"隔离样本symbol/date物理类型不符合string/date32：{path}")
            seen_symbols = set()
            seen_dates = set()
            for batch in parquet.iter_batches(batch_size=65536, columns=["symbol", "date"], use_threads=False):
                symbols = batch.column(batch.schema.get_field_index("symbol"))
                dates = batch.column(batch.schema.get_field_index("date"))
                if symbols.null_count or dates.null_count:
                    raise ValueError(f"隔离样本实际股票或日期含NULL：{path}")
                batch_symbols = set(pc.unique(symbols).to_pylist())
                batch_dates = set(pc.unique(dates).to_pylist())
                if batch_symbols - expected_symbols:
                    raise ValueError(f"隔离样本不是固定六股，存在额外股票：{path}")
                if batch_dates != {day}:
                    raise ValueError(f"隔离样本实际日期与请求交易日不一致：{path}")
                seen_symbols.update(batch_symbols)
                seen_dates.update(batch_dates)
            if seen_symbols != expected_symbols or seen_dates != {day}:
                raise ValueError(f"隔离样本股票集合或实际日期不完整：{path}")
            checked += 1
    return {"status": "passed", "files": checked, "trade_dates": len(VERIFY_DATES),
            "symbols_per_day": len(VERIFY_SYMBOLS), "checked_columns": ["symbol", "date"]}


def independent_orders(frame):
    """逐行字典构造完整日订单参考，不调用生产订单视图。"""

    groups = {}
    timing = "time_ms" if "time_ms" in frame else "timestamp"
    for position, row in enumerate(frame.to_dict("records")):
        key = tuple(row[field] for field in ("symbol", "channel", "side", "order_num"))
        if key not in groups:
            groups[key] = {**row, "_position": position,
                           "money": np.nan if "money" in row else None, "volume": 0}
        target = groups[key]
        target[timing] = min(target[timing], row[timing])
        target["volume"] += row.get("volume", 0)
        if "money" in row and pd.notna(row["money"]):
            target["money"] = row["money"] if pd.isna(target["money"]) else target["money"] + row["money"]
    result = pd.DataFrame(groups.values())
    if result.empty:
        return frame.copy()
    return result.sort_values([timing, "_position"], kind="stable")


def reference(metric, original, day, theme):
    """保留已验收候选的订单字典、概率、加权时间和累计路径参考公式。"""

    frame = independent_orders(original) if metric.inputs == ("weituo",) else original.copy()
    if theme == "maxdiv_min":
        t = frame["time_ms"]
        frame = frame.loc[((t >= 34_200_000) & (t < 41_400_000))
                          | ((t >= 46_800_000) & (t < 53_820_000))].copy()
    if "time_ms" in frame:
        frame["_code"] = assign_minute_codes(frame["time_ms"]).to_numpy()
    else:
        frame["_code"] = assign_timestamp_minute_codes(frame["timestamp"]).to_numpy()
    frame = frame.loc[frame["_code"].ge(0)]
    sparse = {}
    for (symbol, code), events in frame.groupby(["symbol", "_code"], observed=True, sort=False):
        if theme == "maxdiv_min":
            events = events.sort_values("time_ms", kind="stable")
            kind = metric.weight_kind
            weights = np.ones(len(events)) if kind == "d" else events[
                "volume" if kind == "v" else "money"
            ].to_numpy(dtype=float)
            known = ~pd.isna(weights)
            if kind == "m" and not known.any():
                value = np.nan
            else:
                sign = np.where(events["side"].eq(49), 1, np.where(events["side"].eq(50), -1, 0))
                path = np.cumsum(np.where(known, weights, 0) * sign)
                buy = max(0., path.max())
                sell = max(0., -path.min())
                value = {"b": buy, "s": sell, "a": buy + sell}[metric.direction_kind]
        elif theme == "entropy_min":
            volume = events.loc[events["volume"].gt(0), "volume"].to_numpy(dtype=float)
            probability = volume / volume.sum() if len(volume) else volume
            value = max(0., float(-(probability * np.log(probability)).sum())) if len(volume) else np.nan
        elif metric.name == "cd_bv_min":
            value = events.loc[events["side"].eq(49), "volume"].sum()
        else:
            if metric.side_code is not None:
                events = events.loc[events["side"].eq(metric.side_code)]
            weights = events["volume"].to_numpy(dtype=float) if metric.weight_kind == "volume" else np.ones(len(events))
            seconds = (events["time_ms"].to_numpy(dtype=np.int64) % 60_000) / 1000.
            value = float(np.dot(weights, seconds) / weights.sum()) if weights.sum() else np.nan
        sparse[(str(symbol), int(code))] = value
    return sparse


def _price_quality(adapter, dataset, day, input_root):
    """额外取数包只在真实价格对照时导入，普通合成契约测试不依赖它。"""

    from research_toolkit.core.record_source import L2SchemaAdapter

    quote = adapter.read(dataset, day, ("symbol", "price", "money", "volume"))
    raw_path = input_path(input_root, dataset, day)
    schema_adapter = L2SchemaAdapter.inspect(raw_path)
    batches = [schema_adapter.normalize_batch(batch, ("price",)) for batch in
               pq.ParquetFile(raw_path).iter_batches(columns=["price"])]
    toolkit_prices = np.concatenate([batch.column(0).to_numpy(zero_copy_only=False) for batch in batches])
    np.testing.assert_allclose(toolkit_prices, quote["price"], equal_nan=True, rtol=0, atol=0)
    return {
        "events": len(quote), "unknown_price": int(quote["price"].isna().sum()),
        "unknown_money": int(quote["money"].isna().sum()),
        "unknown_price_volume": int(quote.loc[quote["price"].isna(), "volume"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("maxdiv_min", "entropy_min", "time_center_min"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--business-root", type=Path, default=ROOT / "business",
                        help="仅供导入业务源码的任务父目录；输入、输出和报告仍限制在当前框架内")
    args = parser.parse_args()
    input_root, output_root, report_path = validate_isolated_paths(
        args.input_root, args.output_root, args.report, root=ROOT,
    )
    input_preflight = preflight_verify_inputs(input_root)
    business_project = (args.business_root / args.theme).resolve()
    if not business_project.is_dir():
        raise ValueError(f"业务源码目录不存在，请显式指定--business-root：{business_project}")
    sys.path.insert(0, str(business_project))
    import minute_metrics as business
    from trans_to_min import MinuteMetricGroup, MinuteMetricGroupRunner, RunConfig
    from trans_to_min.provenance import read_contracts
    from trans_to_min.storage import month_path

    if args.theme == "maxdiv_min":
        groups = business.build_maxdiv_groups(business.METRIC_CLASSES)
    elif args.theme == "time_center_min":
        groups = business.build_time_center_groups(business.ALL_METRIC_CLASSES)
    else:
        groups = tuple(MinuteMetricGroup(f"{cls.inputs[0]}_entropy_check", (cls(),)) for cls in
                       (business.TradeVolumeEntropy, business.OrderVolumeEntropy, business.CancelVolumeEntropy))
    started = time.perf_counter()
    report = {
        "theme": args.theme, "status": "running",
        "scope": "固定三日沪深各三只股票，不是全市场或全历史验收",
        "reference_boundary": REFERENCE_BOUNDARY,
        "input_preflight": input_preflight,
        "research_toolkit_version": importlib.metadata.version("research-toolkit"),
        "business_source": str(business_project), "results": [],
    }
    adapter = L2InputAdapter(input_root)
    for group in groups:
        roots = {metric.name: output_root for metric in group.metrics}
        run = MinuteMetricGroupRunner(group, RunConfig(
            input_root=input_root, output_root=output_root, date_workers=2,
            publish_while_computing=False), roots).run(VERIFY_DATES)
        if run["status"] != "complete":
            raise ValueError("隔离分钟生成未成功，不能进入参考验收")
        for day in VERIFY_DATES:
            dataset = group.inputs[0]
            original = adapter.read(dataset, day, group.input_columns[dataset])
            symbols = set(str(symbol) for symbol in original["symbol"].unique())
            if symbols != set(VERIFY_SYMBOLS):
                raise ValueError(f"输入不是固定六股样本：{day} {dataset} {sorted(symbols)}")
            price_quality = _price_quality(adapter, dataset, day, input_root) if args.theme == "maxdiv_min" else None
            for metric in group.metrics:
                expected = reference(metric, original, day, args.theme)
                path = month_path(output_root, metric.name, day.strftime("%Y%m"))
                validate_output_schema(pq.ParquetFile(path).schema_arrow, metric.name, metric.value_type)
                table = pq.read_table(path, filters=[("trade_date", "==", day)])
                checks = validate_day_output(table, day, VERIFY_SYMBOLS, metric.name,
                                             metric.value_type, expected, metric.empty_value)
                if day.isoformat() not in read_contracts(path):
                    raise ValueError(f"分钟输出缺少该日来源凭证：{day} {metric.name}")
                report["results"].append({"metric": metric.name, "date": day.isoformat(), **checks,
                                          "price_quality": price_quality, "independent_reference": "passed"})
        # 来源未变化时只能按契约跳过，不能仅凭文件存在认定已完成。
        resumed = MinuteMetricGroupRunner(group, RunConfig(
            input_root=input_root, output_root=output_root, date_workers=2,
            publish_while_computing=False, resume=True), roots).run(VERIFY_DATES)
        if not all(state["progress"]["skipped"] == len(VERIFY_DATES) for state in resumed["metrics"].values()):
            raise ValueError("同源断点复核未全部跳过，不能标记验收完成")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["status"] = "complete"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建也保护检查后并发出现的报告，不能覆盖既有验收证据。
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"theme": args.theme, "checks": len(report["results"]), "status": "complete",
                      "seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
