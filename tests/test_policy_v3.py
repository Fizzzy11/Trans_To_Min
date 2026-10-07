"""新版三表的订单、双侧计数、空价与来源断点验收。"""
from datetime import date
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trans_to_min import aggregate_order_fragments, trade_order_participation, validate_amounts
from trans_to_min import (
    MinuteMetric,
    MinuteConversionRunner,
    MinuteMetricGroup,
    MinuteMetricGroupRunner,
    RunConfig,
    input_path,
)
from trans_to_min.minute import minute_labels
from trans_to_min.storage import month_path
from trans_to_min.provenance import read_contracts, source_contract

DAY = date(2025, 1, 9)


def orders():
    return pd.DataFrame({
        "symbol": ["000001.XSHE"]*4, "channel": [1]*4, "side": [49,49,50,50],
        "order_num": [10,10,20,21], "time_ms": [34200001,34261000,34200002,34200003],
        "volume": [600,400,200,300], "money": [6000,4004,None,3000],
    })


def test_orders_use_full_identity_and_earliest_anchor_without_price_filter():
    frame=orders(); frame.loc[0,"money"]=np.nan
    before=frame.copy(deep=True)
    result=aggregate_order_fragments(frame,DAY).set_index("order_num")
    assert result.loc[10,"time_ms"]==34200001
    assert result.loc[10,"volume"]==1000 and result.loc[10,"money"]==4004
    assert pd.isna(result.loc[20,"money"])
    pd.testing.assert_frame_equal(frame,before)


def test_split_600_400_amount_is_sum_of_fragments_not_first_price():
    result=aggregate_order_fragments(orders(),DAY).set_index("order_num")
    assert result.loc[10,"money"]==10004 and result.loc[10,"volume"]==1000


@pytest.mark.parametrize("different",["symbol","channel","side"])
def test_order_identity_does_not_cross_stock_channel_direction(different):
    frame=orders().iloc[:2].copy()
    frame.loc[1,different]={"symbol":"600000.XSHG","channel":2,"side":50}[different]
    assert len(aggregate_order_fragments(frame,DAY))==2


@pytest.mark.parametrize("missing",["channel","order_num","side"])
def test_order_missing_keys_fail_instead_of_silent_row_count(missing):
    with pytest.raises(ValueError,match="缺少字段"):
        aggregate_order_fragments(orders().drop(columns=missing),DAY)


def test_trade_two_legs_and_cross_minute_do_not_change_original_events():
    frame=pd.DataFrame({"symbol":["000001.XSHE"]*3,"channel":[1]*3,
        "time_ms":[34200001,34261000,34261001],"buy_order_num":[10]*3,
        "sell_order_num":[20,21,21],"side":[49,50,50],"volume":[600,200,200]})
    before=frame.copy()
    assert len(trade_order_participation(frame,DAY,order_side="buy",time_scope="first_event"))==1
    assert len(trade_order_participation(frame,DAY,order_side="sell",time_scope="first_event"))==2
    assert len(trade_order_participation(frame,DAY,order_side="both",time_scope="first_event"))==3
    assert len(trade_order_participation(frame,DAY,order_side="buy",time_scope="per_minute"))==2
    pd.testing.assert_frame_equal(frame,before)


@pytest.mark.parametrize("value",[-1,float("inf"),float("-inf"),"错误"])
def test_nonempty_bad_money_not_swallowed(value):
    with pytest.raises((ValueError,TypeError)):
        validate_amounts(pd.Series([None,0,value]))


class AmountSum(MinuteMetric):
    name="known_money_min"
    inputs=("weituo",)
    input_columns={"weituo":("symbol","time_ms","money")}
    empty_value=0.0
    def prepare_events(self,day,inputs):
        return inputs["weituo"]
    def aggregate_minutes(self,events):
        return events.groupby(["symbol","minute"],observed=True)["money"].sum(min_count=1).rename(self.name).reset_index()


def write_input(root, *, policy=b"3", day=DAY, priced=100000):
    path=input_path(root,"weituo",day);path.parent.mkdir(parents=True,exist_ok=True)
    table=pa.table({"symbol":["000001.XSHE"]*3,"time":pa.array([34200001,34200002,34261000],pa.uint32()),
        "price":pa.array([None,priced,None],pa.int64()),"volume":pa.array([200,300,100],pa.uint32())})
    pq.write_table(table.replace_schema_metadata({b"price_scale":b"10000",b"price_policy_version":policy}),path)
    return path


def test_all_unknown_group_remains_null_but_true_empty_minute_zero():
    frame=orders().iloc[:3].copy()
    frame["time_ms"]=[34200001,34200002,34261000]
    frame["money"]=[None,0,None]
    result=AmountSum().compute_day(DAY,{"weituo":frame}).set_index("minute")
    labels=minute_labels(DAY)
    assert result.loc[labels[1],"known_money_min"]==0
    assert pd.isna(result.loc[labels[2],"known_money_min"])
    assert result.loc[labels[3],"known_money_min"]==0


def test_resume_requires_same_source_and_preserves_contract(tmp_path):
    raw=tmp_path/"raw";out=tmp_path/"output";write_input(raw)
    config=RunConfig(input_root=raw,output_root=out,date_workers=1,publish_while_computing=False)
    MinuteConversionRunner(AmountSum(),config).run([DAY])
    path=month_path(out,"known_money_min","202501")
    assert DAY.isoformat() in read_contracts(path)
    resume=RunConfig(input_root=raw,output_root=out,date_workers=1,publish_while_computing=False,resume=True)
    assert MinuteConversionRunner(AmountSum(),resume).run([DAY])["pending_dates"]==[]
    write_input(raw,policy=b"2")
    with pytest.raises(ValueError,match="不能resume"):
        MinuteConversionRunner(AmountSum(),resume).run([DAY])


def test_old_output_without_contract_never_silently_skipped(tmp_path):
    raw=tmp_path/"raw";out=tmp_path/"output";write_input(raw)
    config=RunConfig(input_root=raw,output_root=out,date_workers=1,publish_while_computing=False)
    MinuteConversionRunner(AmountSum(),config).run([DAY])
    path=month_path(out,"known_money_min","202501")
    table=pq.read_table(path)
    metadata=dict(table.schema.metadata);metadata.pop(b"source_contracts")
    pq.write_table(table.replace_schema_metadata(metadata),path)
    with pytest.raises(ValueError,match="不能resume"):
        MinuteConversionRunner(AmountSum(),RunConfig(input_root=raw,output_root=out,date_workers=1,resume=True)).run([DAY])


def test_required_policy_blocks_old_input_before_computation(tmp_path):
    raw = tmp_path / "raw"
    write_input(raw, policy=b"2")
    metric = AmountSum()
    metric.required_price_policy_version = 3
    with pytest.raises(ValueError, match="要求报价版本3"):
        MinuteConversionRunner(metric, RunConfig(input_root=raw, output_root=tmp_path/"output", date_workers=1)).run([DAY])
    assert not (tmp_path/"output"/"known_money_min"/"2025"/"202501.parquet").exists()


def test_algorithm_change_blocks_resume(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "output"
    write_input(raw)
    MinuteConversionRunner(AmountSum(), RunConfig(input_root=raw,output_root=out,date_workers=1)).run([DAY])
    changed = AmountSum()
    changed.algorithm_version = "changed"
    with pytest.raises(ValueError, match="不能resume"):
        MinuteConversionRunner(changed, RunConfig(input_root=raw,output_root=out,date_workers=1,resume=True)).run([DAY])


def test_explicit_null_integer_and_true_empty_zero_do_not_mix():
    metric = AmountSum()
    metric.value_type = "uint64"
    aggregated = pd.DataFrame({"symbol":["000001.XSHE"], "minute":[minute_labels(DAY)[1]], metric.name:[None]})
    result = metric.finalize_aggregated_day(DAY, {"weituo":orders()}, aggregated)
    assert str(result[metric.name].dtype) == "UInt64"
    assert pd.isna(result.loc[1, metric.name])
    assert result.loc[2, metric.name] == 0


def test_quality_has_missing_volume_and_mixed_orders():
    frame = orders()
    frame.loc[1,"money"] = np.nan
    quality = aggregate_order_fragments(frame, DAY).attrs["price_quality"]
    assert quality["unknown_money_fragments"] == 2
    assert quality["unknown_money_volume"] == 600
    assert quality["mixed_money_orders"] == 1


def _config(raw, out, **changes):
    """关闭发布重叠，使故障注入和逐日契约检查可确定复现。"""
    return RunConfig(
        input_root=raw,
        output_root=out,
        date_workers=1,
        publish_while_computing=False,
        **changes,
    )


def test_month_increment_retains_audited_day_contracts(tmp_path):
    """同月新增日期保留旧日凭证，并允许整个新月份按同源恢复。"""
    raw, out = tmp_path / "raw", tmp_path / "output"
    next_day = date(2025, 1, 10)
    write_input(raw)
    write_input(raw, day=next_day)
    metric = AmountSum()
    MinuteConversionRunner(metric, _config(raw, out)).run([DAY])
    path = month_path(out, metric.name, "202501")
    original = read_contracts(path)
    MinuteConversionRunner(metric, _config(raw, out)).run([next_day])
    updated = read_contracts(path)
    assert updated[DAY.isoformat()] == original[DAY.isoformat()]
    assert updated[next_day.isoformat()] == source_contract(metric, next_day, raw)
    assert len(pq.read_table(path)) == 484
    resumed = MinuteConversionRunner(
        metric, _config(raw, out, resume=True)
    ).run([DAY, next_day])
    assert resumed["pending_dates"] == []


def test_explicit_day_replacement_changes_only_that_contract(tmp_path):
    """显式重建只替换目标日期的值和凭证，不冒充同月其他日期重算。"""
    raw, out = tmp_path / "raw", tmp_path / "output"
    next_day = date(2025, 1, 10)
    write_input(raw)
    write_input(raw, day=next_day)
    metric = AmountSum()
    MinuteConversionRunner(metric, _config(raw, out)).run([DAY, next_day])
    path = month_path(out, metric.name, "202501")
    original_contracts = read_contracts(path)
    original_frame = pq.read_table(path).to_pandas()
    write_input(raw, priced=200000)
    MinuteConversionRunner(
        metric, _config(raw, out, overwrite_dates=True)
    ).run([DAY])
    updated_contracts = read_contracts(path)
    assert updated_contracts[DAY.isoformat()] != original_contracts[DAY.isoformat()]
    assert updated_contracts[next_day.isoformat()] == original_contracts[next_day.isoformat()]
    frame = pq.read_table(path).to_pandas()
    pd.testing.assert_frame_equal(
        original_frame.loc[original_frame["trade_date"].eq(next_day)].reset_index(drop=True),
        frame.loc[frame["trade_date"].eq(next_day)].reset_index(drop=True),
    )
    assert frame.loc[frame["trade_date"].eq(DAY), metric.name].max() == 6000
    assert MinuteConversionRunner(
        metric, _config(raw, out, resume=True)
    ).run([DAY, next_day])["pending_dates"] == []


class SharedAmountGroup(MinuteMetricGroup):
    """代表具有自定义共享计算入口的指标组。"""

    def iter_compute_day(self, trade_date, inputs, metric_names=None):
        yield from super().iter_compute_day(trade_date, inputs, metric_names)


def test_default_group_contract_matches_single_metric_and_resumes(tmp_path):
    """默认组仅委派单指标计算，名字不同也不应改变数值契约。"""
    raw, out = tmp_path / "raw", tmp_path / "output"
    write_input(raw)
    metric = AmountSum()
    group = MinuteMetricGroup("default_amount_group", (metric,))
    assert source_contract(metric, DAY, raw) == source_contract(
        metric, DAY, raw, group=group
    )
    MinuteConversionRunner(metric, _config(raw, out)).run([DAY])
    resumed = MinuteMetricGroupRunner(
        group, _config(raw, out, resume=True)
    ).run([DAY])
    assert resumed["metrics"][metric.name]["progress"]["pending"] == 0
    assert resumed["metrics"][metric.name]["progress"]["skipped"] == 1


def test_custom_group_contract_is_not_silently_mixed_with_single(tmp_path):
    """自定义组源码及组名参与凭证；同一自定义组可恢复，换入口须阻断。"""
    raw, out = tmp_path / "raw", tmp_path / "output"
    write_input(raw)
    metric = AmountSum()
    group = SharedAmountGroup("shared_amount_group", (metric,))
    assert source_contract(metric, DAY, raw) != source_contract(
        metric, DAY, raw, group=group
    )
    assert source_contract(metric, DAY, raw, group=group) != source_contract(
        metric, DAY, raw,
        group=SharedAmountGroup("another_shared_group", (metric,)),
    )
    MinuteMetricGroupRunner(group, _config(raw, out)).run([DAY])
    assert MinuteMetricGroupRunner(
        group, _config(raw, out, resume=True)
    ).run([DAY])["metrics"][metric.name]["progress"]["pending"] == 0
    with pytest.raises(ValueError, match="不能resume"):
        MinuteConversionRunner(metric, _config(raw, out, resume=True)).run([DAY])
    with pytest.raises(ValueError, match="不能resume"):
        MinuteMetricGroupRunner(
            MinuteMetricGroup("default_group", (metric,)),
            _config(raw, out, resume=True),
        ).run([DAY])


@pytest.mark.parametrize("grouped", [False, True])
def test_source_change_during_compute_refuses_month_publication(tmp_path, grouped):
    """计算后源替换时单/组入口都失败，不留下可误当成功的正式月文件。"""
    raw, out = tmp_path / "raw", tmp_path / "output"
    write_input(raw)
    original_compute = AmountSum.compute_day

    def replace_source_after_compute(metric, trade_date, inputs):
        frame = original_compute(metric, trade_date, inputs)
        write_input(raw, priced=200000)
        return frame

    with patch.object(AmountSum, "compute_day", replace_source_after_compute):
        if grouped:
            runner = MinuteMetricGroupRunner(
                MinuteMetricGroup("changing_source_group", (AmountSum(),)),
                _config(raw, out),
            )
        else:
            runner = MinuteConversionRunner(AmountSum(), _config(raw, out))
        with pytest.raises(RuntimeError, match="计算期间来源或算法发生变化"):
            runner.run([DAY])
    assert not month_path(out, "known_money_min", "202501").exists()


def test_declared_business_parameters_participate_in_resume_contract(tmp_path):
    """业务实例的显式参数变化也必须阻止旧结果被跳过。"""
    raw, out = tmp_path / "raw", tmp_path / "output"
    write_input(raw)
    metric = AmountSum()
    metric.contract_parameters = {"统计单位": "事件", "阈值": 1}
    MinuteConversionRunner(metric, _config(raw, out)).run([DAY])
    changed = AmountSum()
    changed.contract_parameters = {"统计单位": "事件", "阈值": 2}
    with pytest.raises(ValueError, match="不能resume"):
        MinuteConversionRunner(changed, _config(raw, out, resume=True)).run([DAY])


@pytest.mark.parametrize("use_timestamp", [False, True])
def test_complete_stock_partitions_match_whole_day_order_view(use_timestamp):
    """按整股分区保留完整订单，与整日聚合逐值相同，不允许跨分钟切碎。"""
    first = orders()
    first.loc[0, "money"] = np.nan
    second = orders()
    second["symbol"] = "600000.XSHG"
    second["channel"] = 2
    second["time_ms"] += 120000
    frame = pd.concat([first, second], ignore_index=True)
    frame["trade_date"] = DAY
    timing = "time_ms"
    if use_timestamp:
        frame["timestamp"] = (
            pd.Timestamp(DAY).tz_localize("Asia/Shanghai")
            + pd.to_timedelta(frame["time_ms"], unit="ms")
        )
        frame = frame.drop(columns="time_ms")
        timing = "timestamp"
    keys = ["symbol", "channel", "side", "order_num"]
    whole = aggregate_order_fragments(frame, DAY)
    parts = pd.concat([
        aggregate_order_fragments(stock, DAY)
        for _, stock in frame.groupby("symbol", sort=False)
    ], ignore_index=True)
    whole.attrs.clear()
    parts.attrs.clear()
    pd.testing.assert_frame_equal(
        whole.sort_values(keys).reset_index(drop=True),
        parts.sort_values(keys).reset_index(drop=True),
    )
    order = whole.loc[whole["symbol"].eq("000001.XSHE") & whole["order_num"].eq(10)].iloc[0]
    assert order["volume"] == 1000 and order["money"] == 4004
    expected = (
        pd.Timestamp(f"{DAY.isoformat()} 09:30:00.001", tz="Asia/Shanghai")
        if use_timestamp else 34200001
    )
    if use_timestamp:
        assert pd.Timestamp(order[timing]) == expected
    else:
        assert int(order[timing]) == expected


def test_trade_missing_physical_money_cannot_be_recomputed(tmp_path):
    """成交物理额缺失必须失败，不通过价乘量伪造沪市真实成交额。"""
    path = input_path(tmp_path, "chengjiao", DAY)
    path.parent.mkdir(parents=True)
    table = pa.table({"price": [10.0], "volume": [100]})
    pq.write_table(table, path)
    from trans_to_min import L2InputAdapter
    with pytest.raises(ValueError, match="成交必须读取原money"):
        L2InputAdapter(tmp_path).read("chengjiao", DAY, ("money",))


def test_real_trade_money_validation_does_not_allow_missing():
    """真实成交额严格校验与委托/撤单允许缺失的分支不能混用。"""
    with pytest.raises(ValueError, match="真实成交金额不允许为空"):
        validate_amounts(pd.Series([1000.0, None]), allow_missing=False)
