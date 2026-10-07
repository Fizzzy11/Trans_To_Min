"""隔离验收门禁的合成测试；不运行真实生成或调用生产订单聚合构造预期。"""

from datetime import date, timedelta
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

def load_validation_script():
    """按绝对文件载入helper，不把源码根加入wheel或spawn进程的搜索路径。"""

    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_policy_v3_chain.py"
    spec = importlib.util.spec_from_file_location("_minute_chain_validation_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法载入隔离验收helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = load_validation_script()
independent_orders = validation.independent_orders
reference = validation.reference
validate_day_output = validation.validate_day_output
validate_isolated_paths = validation.validate_isolated_paths
validate_output_schema = validation.validate_output_schema
preflight_verify_inputs = validation.preflight_verify_inputs


DAY = date(2025, 1, 9)
SYMBOLS = ("000001.XSHE", "600000.XSHG")
METRIC = "chain_amount_min"
SPARSE = {(SYMBOLS[0], 0): 0, (SYMBOLS[0], 1): None,
          (SYMBOLS[0], 2): 10004, (SYMBOLS[1], 3): 4004}


def sample_table(value_type="float64"):
    """独立构造242标签与物理字段，不复用生产分钟轴或Schema生成器。"""

    types = {"float64": pa.float64(), "int64": pa.int64(), "uint64": pa.uint64()}
    labels = pd.date_range(f"{DAY} 09:30", f"{DAY} 11:30", freq="min", tz="Asia/Shanghai").append(
        pd.date_range(f"{DAY} 13:00", f"{DAY} 15:00", freq="min", tz="Asia/Shanghai")
    )
    schema = pa.schema([
        pa.field("trade_date", pa.date32(), nullable=False),
        pa.field("minute", pa.timestamp("ms", tz="Asia/Shanghai"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field(METRIC, types[value_type], nullable=True),
    ])
    symbols = [symbol for symbol in SYMBOLS for _ in labels]
    values = [SPARSE.get((symbol, code), 0) for symbol in SYMBOLS for code in range(242)]
    return pa.Table.from_arrays([
        pa.array([DAY] * len(symbols), type=pa.date32()),
        pa.array(list(labels) * len(SYMBOLS), type=schema.field("minute").type),
        pa.array(symbols, type=pa.string()),
        pa.array(values, type=types[value_type], from_pandas=True),
    ], schema=schema)


def check(table, *, symbols=SYMBOLS, sparse=SPARSE, value_type="float64", empty_value=0):
    return validate_day_output(table, DAY, symbols, METRIC, value_type, sparse, empty_value)


def replace_column(table, name, values):
    """保留字段物理契约，只替换测试数据内容。"""

    position = table.schema.get_field_index(name)
    field = table.schema.field(name)
    return table.set_column(position, field, pa.array(values, type=field.type, from_pandas=True))


@pytest.mark.parametrize("value_type", ["float64", "int64", "uint64"])
def test_complete_physical_output_preserves_null_zero_and_known_amount(value_type):
    table = sample_table(value_type)
    result = check(table, value_type=value_type)
    assert result == {"rows": 484, "symbols": 2, "null_values": 1,
                      "output_contract": "passed", "physical_nulls": "passed"}


def test_contract_accepts_shuffled_rows_but_compares_values_by_key():
    table = sample_table().take(pa.array(np.arange(483, -1, -1)))
    assert check(table)["rows"] == 484


def test_loading_helper_does_not_change_wheel_module_search_path():
    import trans_to_min

    search_path = list(sys.path)
    imported_core = trans_to_min.__file__
    load_validation_script()
    assert sys.path == search_path
    assert trans_to_min.__file__ == imported_core


@pytest.mark.parametrize("name,physical_type,nullable", [
    ("trade_date", pa.date64(), False),
    ("minute", pa.timestamp("us", tz="Asia/Shanghai"), False),
    ("minute", pa.timestamp("ms", tz="UTC"), False),
    ("minute", pa.timestamp("ms"), False),
    ("symbol", pa.large_string(), False),
    (METRIC, pa.float32(), True),
    ("trade_date", pa.date32(), True),
    (METRIC, pa.float64(), False),
])
def test_wrong_physical_type_timezone_or_nullability_is_rejected(name, physical_type, nullable):
    schema = sample_table().schema
    changed = schema.set(schema.get_field_index(name), pa.field(name, physical_type, nullable=nullable))
    with pytest.raises(ValueError, match="物理Schema"):
        validate_output_schema(changed, METRIC, "float64")


@pytest.mark.parametrize("mode", ["missing", "extra", "reordered"])
def test_four_columns_must_match_names_and_order(mode):
    schema = sample_table().schema
    if mode == "missing":
        schema = schema.remove(3)
    elif mode == "extra":
        schema = schema.append(pa.field("price_source", pa.string()))
    else:
        schema = pa.schema([schema.field(1), schema.field(0), schema.field(2), schema.field(3)])
    with pytest.raises(ValueError, match="四列"):
        validate_output_schema(schema, METRIC, "float64")


@pytest.mark.parametrize("key", ["trade_date", "minute", "symbol"])
def test_identity_columns_must_not_contain_physical_null(key):
    table = sample_table()
    values = table.column(key).to_pylist()
    values[0] = None
    with pytest.raises(ValueError, match="唯一键字段不允许物理NULL"):
        check(replace_column(table, key, values))


def test_actual_trade_date_must_match_requested_day():
    table = sample_table()
    dates = table.column("trade_date").to_pylist()
    dates[0] = DAY + timedelta(days=1)
    with pytest.raises(ValueError, match="实际trade_date"):
        check(replace_column(table, "trade_date", dates))


@pytest.mark.parametrize("symbols", [(SYMBOLS[0],), (*SYMBOLS, "000002.XSHE")])
def test_exact_stock_set_not_just_row_count_is_required(symbols):
    with pytest.raises(ValueError, match="股票集合"):
        check(sample_table(), symbols=symbols)


def test_balanced_duplicate_and_missing_key_cannot_pass_total_row_count():
    table = sample_table()
    table = pa.concat_tables([table.slice(0, 483), table.slice(0, 1)])
    assert table.num_rows == 484
    with pytest.raises(ValueError, match="重复唯一键"):
        check(table)


def test_missing_minute_is_rejected():
    with pytest.raises(ValueError, match="分钟行数不完整"):
        check(sample_table().slice(0, 483))


@pytest.mark.parametrize("label", [f"{DAY} 12:00", f"{DAY} 09:30:30", f"{DAY + timedelta(days=1)} 09:30"])
def test_off_axis_label_and_minute_date_are_rejected(label):
    table = sample_table()
    minutes = table.column("minute").to_pylist()
    minutes[0] = pd.Timestamp(label, tz="Asia/Shanghai")
    with pytest.raises(ValueError, match="242标签之外"):
        check(replace_column(table, "minute", minutes))


@pytest.mark.parametrize("position,value", [(1, 0), (0, None)])
def test_unknown_amount_and_known_zero_must_not_exchange_null_state(position, value):
    table = sample_table()
    values = table.column(METRIC).to_pylist()
    values[position] = value
    with pytest.raises(ValueError, match="物理NULL位图"):
        check(replace_column(table, METRIC, values))


def test_nonnull_nan_cannot_impersonate_parquet_null():
    table = sample_table()
    values = table.column(METRIC).to_pylist()
    values[1] = np.nan
    table = table.set_column(3, table.schema.field(3), pa.array(values, type=pa.float64(), from_pandas=False))
    assert table.column(METRIC).null_count == 0
    with pytest.raises(ValueError, match="物理NULL位图"):
        check(table)


def test_known_value_corruption_is_rejected():
    table = sample_table()
    values = table.column(METRIC).to_pylist()
    values[2] = 10000
    with pytest.raises(ValueError, match="指标值与独立公式"):
        check(replace_column(table, METRIC, values))


def test_large_integer_comparison_does_not_use_float_tolerance():
    table = sample_table("uint64")
    values = table.column(METRIC).to_pylist()
    values[2] = 2**53 + 1
    expected = {**SPARSE, (SYMBOLS[0], 2): 2**53}
    with pytest.raises(ValueError, match="整数指标"):
        check(replace_column(table, METRIC, values), sparse=expected, value_type="uint64")


@pytest.mark.parametrize("key", [("000002.XSHE", 0), (SYMBOLS[0], 242)])
def test_reference_keys_outside_expected_axis_are_not_silently_ignored(key):
    with pytest.raises(ValueError, match="独立参考包含"):
        check(sample_table(), sparse={**SPARSE, key: 1})


def test_true_empty_null_semantics_are_checked_separately_from_explicit_unknown():
    table = sample_table()
    values = [SPARSE.get((symbol, code)) for symbol in SYMBOLS for code in range(242)]
    result = check(replace_column(table, METRIC, values), empty_value=None)
    assert result["null_values"] == 481


def test_independent_order_dictionary_keeps_anchor_volume_and_known_fragment_amount():
    frame = pd.DataFrame({
        "symbol": [SYMBOLS[0]] * 4, "channel": [1] * 4, "side": [49] * 4,
        "order_num": [10, 10, 20, 20], "time_ms": [34200010, 34260010, 34200020, 34260020],
        "volume": [600, 400, 200, 300], "money": [6000, 4004, None, 3000],
    })
    original = frame.copy(deep=True)
    result = independent_orders(frame).set_index("order_num")
    assert result.loc[10, "volume"] == 1000
    assert result.loc[10, "money"] == 10004
    assert result.loc[20, "money"] == 3000
    assert result.loc[20, "time_ms"] == 34200020
    pd.testing.assert_frame_equal(frame, original)


def test_reference_preserves_unknown_zero_and_order_amount_without_production_order_view():
    frame = pd.DataFrame({
        "symbol": [SYMBOLS[0]] * 4, "channel": [1] * 4, "side": [49] * 4,
        "order_num": [10, 10, 20, 30], "time_ms": [34200010, 34260010, 34320010, 34380010],
        "volume": [600, 400, 100, 100], "money": [6000, 4004, None, 0],
    })
    metric = SimpleNamespace(inputs=("weituo",), weight_kind="m", direction_kind="b")
    expected = reference(metric, frame, DAY, "maxdiv_min")
    assert expected[(SYMBOLS[0], 1)] == 10004
    assert (SYMBOLS[0], 2) not in expected
    assert pd.isna(expected[(SYMBOLS[0], 3)])
    assert expected[(SYMBOLS[0], 4)] == 0


@pytest.mark.parametrize("dataset", ["chengjiao", "chedan"])
def test_trade_and_cancel_reference_keeps_actual_event_minutes(dataset):
    frame = pd.DataFrame({
        "symbol": [SYMBOLS[0]] * 2, "channel": [1, 1], "side": [49, 49],
        "order_num": [10, 10], "time_ms": [34200010, 34260010], "volume": [600, 400],
    })
    metric = SimpleNamespace(inputs=(dataset,), weight_kind="v", direction_kind="b")
    expected = reference(metric, frame, DAY, "maxdiv_min")
    assert expected == {(SYMBOLS[0], 1): 600, (SYMBOLS[0], 2): 400}


@pytest.mark.parametrize("outside", ["input", "output", "report"])
def test_paths_outside_isolated_root_are_rejected(tmp_path, outside):
    root = tmp_path / "candidate"
    source = root / "input"
    source.mkdir(parents=True)
    paths = {"input": source, "output": root / "output", "report": root / "report.json"}
    paths[outside] = tmp_path / outside
    with pytest.raises(ValueError, match="只允许当前候选内"):
        validate_isolated_paths(paths["input"], paths["output"], paths["report"], root=root)


def test_paths_keep_input_readonly_and_refuse_existing_output_or_report(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    report = tmp_path / "report.json"
    assert validate_isolated_paths(source, output, report, root=tmp_path) == (source, output, report)
    with pytest.raises(ValueError, match="输入树"):
        validate_isolated_paths(source, source, report, root=tmp_path)
    output.mkdir()
    (output / "retained.txt").write_text("既有隔离证据", encoding="utf-8")
    with pytest.raises(ValueError, match="不覆盖既有"):
        validate_isolated_paths(source, output, report, root=tmp_path)
    report.write_text("既有报告", encoding="utf-8")
    with pytest.raises(ValueError, match="报告已存在"):
        validate_isolated_paths(source, tmp_path / "new_output", report, root=tmp_path)


def test_report_file_cannot_be_an_output_directory(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(ValueError, match="独立文件路径"):
        validate_isolated_paths(source, tmp_path / "output", tmp_path / "output", root=tmp_path)


def write_preflight_inputs(input_root, *, problem=None, changed_metadata=None):
    """仅构造九份symbol/date合成投影，不运行raw或分钟生成。"""

    metadata = {b"price_policy_version": b"3", b"format_version": b"1"}
    if changed_metadata:
        metadata.update(changed_metadata)
    for day in validation.VERIFY_DATES:
        for dataset in ("weituo", "chengjiao", "chedan"):
            symbols = list(validation.VERIFY_SYMBOLS)
            dates = [day] * len(symbols)
            if day == validation.VERIFY_DATES[0] and dataset == "weituo":
                if problem == "full_pool":
                    symbols.append("000003.XSHE")
                    dates.append(day)
                elif problem == "wrong_date":
                    dates[0] = day + timedelta(days=1)
            text = day.strftime("%Y%m%d")
            path = input_root / dataset / text[:4] / text[:6] / f"{text}_{dataset}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table({"symbol": pa.array(symbols, type=pa.string()),
                              "date": pa.array(dates, type=pa.date32())})
            pq.write_table(table.replace_schema_metadata(metadata), path)


def test_legal_three_day_six_stock_input_preflight_passes_without_output(tmp_path):
    source = tmp_path / "input"
    write_preflight_inputs(source)
    assert preflight_verify_inputs(source) == {
        "status": "passed", "files": 9, "trade_dates": 3,
        "symbols_per_day": 6, "checked_columns": ["symbol", "date"],
    }
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("problem, message", [
    ("full_pool", "不是固定六股"), ("wrong_date", "实际日期与请求交易日不一致"),
])
def test_invalid_sample_is_rejected_before_business_or_runner(tmp_path, monkeypatch, problem, message):
    from trans_to_min import MinuteMetricGroupRunner

    source = tmp_path / "input"
    write_preflight_inputs(source, problem=problem)
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_policy_v3_chain.py", "--theme", "maxdiv_min",
                                     "--input-root", str(source), "--output-root", str(tmp_path / "output"),
                                     "--report", str(tmp_path / "report.json")])
    calls = []

    def forbidden_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("不应触发分钟生成")

    monkeypatch.setattr(MinuteMetricGroupRunner, "run", forbidden_run)
    # 业务目录未构造；预检应在业务导入前失败，也绝不能触发真实runner。
    with pytest.raises(ValueError, match=message):
        validation.main()
    assert calls == []
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("metadata, message", [
    ({b"price_policy_version": b"2"}, "price_policy_version必须为3"),
    ({b"format_version": b"2"}, "format_version必须为1"),
])
def test_input_preflight_rejects_wrong_policy_or_format(tmp_path, metadata, message):
    source = tmp_path / "input"
    write_preflight_inputs(source, changed_metadata=metadata)
    with pytest.raises(ValueError, match=message):
        preflight_verify_inputs(source)
