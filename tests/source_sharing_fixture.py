"""源共享回归使用的指标与独立参考；放在可导入模块里供日期子进程使用。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import MinuteMetric, MinuteMetricGroup, input_path


class OrderCount(MinuteMetric):
    name = "share_count_min"
    inputs = ("weituo",)
    input_columns = {"weituo": ("symbol", "time_ms")}
    value_type = "uint64"
    empty_value = 0

    def prepare_events(self, trade_date, inputs):
        return inputs["weituo"].loc[:, ["symbol", "time_ms"]]

    def aggregate_minutes(self, events):
        return (
            events.groupby(["symbol", "minute"], observed=True)
            .size().astype("uint64").rename(self.name).reset_index()
        )


class OrderVolume(MinuteMetric):
    name = "share_volume_min"
    inputs = ("weituo",)
    input_columns = {"weituo": ("symbol", "time_ms", "volume")}
    value_type = "uint64"
    empty_value = 0

    def prepare_events(self, trade_date, inputs):
        return inputs["weituo"].loc[:, ["symbol", "time_ms", "volume"]]

    def aggregate_minutes(self, events):
        return (
            events.groupby(["symbol", "minute"], observed=True)["volume"]
            .sum().astype("uint64").rename(self.name).reset_index()
        )


def build_group() -> MinuteMetricGroup:
    return MinuteMetricGroup("share_group", (OrderCount(), OrderVolume()))


class NullableOrderVolume(OrderVolume):
    """没有委托的分钟输出NULL而不是0。"""

    empty_value = None


def build_nullable_group() -> MinuteMetricGroup:
    return MinuteMetricGroup("share_group", (OrderCount(), NullableOrderVolume()))


def masked_reference(trade_date, sources):
    """无委托分钟的委托量期望为NULL，用掩码声明。"""

    result = day_reference(trade_date, sources)
    result["values"]["share_volume_min"] = np.ma.masked_array(
        result["values"]["share_volume_min"], mask=result["values"]["share_count_min"] == 0)
    return result


def write_input(root: Path, day: date, *, extra_volume: int = 0) -> Path:
    path = input_path(root, "weituo", day)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "symbol": pa.array(["000001.XSHE", "000001.XSHE", "600000.XSHG"], type=pa.string()),
        "date": pa.array([day] * 3, type=pa.date32()),
        # 09:30:00.001、09:31:00.000（同属09:31分钟）与10:00:00.000
        "time": pa.array([34_200_001, 34_260_000, 36_000_000], type=pa.uint32()),
        "volume": pa.array([100, 200 + extra_volume, 300], type=pa.uint32()),
    })
    pq.write_table(table, path)
    return path


def day_reference(trade_date, sources):
    """不经过框架适配器和分钟工具的独立复算。"""

    table = pq.ParquetFile(sources["weituo"].open()).read()
    symbols = sorted(set(table["symbol"].to_pylist()))
    index = {symbol: position for position, symbol in enumerate(symbols)}
    count = np.zeros((len(symbols), 242), dtype=np.uint64)
    volume = np.zeros((len(symbols), 242), dtype=np.uint64)
    for symbol, time_ms, size in zip(
        table["symbol"].to_pylist(), table["time"].to_pylist(), table["volume"].to_pylist()
    ):
        minute = 0 if time_ms <= 34_200_000 else (time_ms + 59_999) // 60_000 - 570
        count[index[symbol], minute] += 1
        volume[index[symbol], minute] += size
    return {
        "symbols": symbols,
        "values": {"share_count_min": count, "share_volume_min": volume},
        "info": {"rows": table.num_rows},
    }


def wrong_reference(trade_date, sources):
    result = day_reference(trade_date, sources)
    result["values"]["share_volume_min"] = result["values"]["share_volume_min"] + np.uint64(1)
    return result
