"""三表新价格口径的物理编码、空值和金额分支回归测试。"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trans_to_min.adapter import L2InputAdapter, input_path


def _write(root, dataset, prices, *, physical_type=pa.int64(), scale=b"10000"):
    metadata = {b"price_policy_version": b"2"}
    if scale is not None:
        metadata[b"price_scale"] = scale
    schema = pa.schema([
        pa.field("symbol", pa.string()), pa.field("date", pa.date32()),
        pa.field("time", pa.uint32()), pa.field("price", physical_type),
        pa.field("volume", pa.uint32()),
    ], metadata=metadata)
    count = len(prices)
    table = pa.table({
        "symbol": ["000001.XSHE"] * count, "date": [date(2021, 6, 9)] * count,
        "time": [34_200_001] * count, "price": prices, "volume": [100] * count,
    }, schema=schema)
    path = input_path(Path(root), dataset, 20210609)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.mark.parametrize("dataset", ["weituo", "chedan"])
@pytest.mark.parametrize("prices", [[100_000, None], [None, None]])
def test_nullable_scaled_quote_keeps_counts_volume_and_only_filters_money_branch(tmp_path, dataset, prices):
    _write(tmp_path, dataset, prices)
    frame = L2InputAdapter(tmp_path).read(dataset, 20210609, ("symbol", "time_ms", "price", "volume", "money"))
    assert len(frame) == 2
    assert frame["volume"].sum() == 200
    assert frame["money"].isna().tolist() == [value is None for value in prices]
    priced = frame.loc[frame["money"].notna()]
    assert len(priced) == sum(value is not None for value in prices)
    if not priced.empty:
        assert priced["price"].tolist() == [10.0]
        assert priced["money"].tolist() == [1000.0]
        assert np.isfinite(priced["money"]).all()
    assert len(frame) == 2


def test_nullable_integer_quote_still_requires_scale(tmp_path):
    _write(tmp_path, "weituo", [100_000, None], scale=None)
    with pytest.raises(ValueError, match="price_scale"):
        L2InputAdapter(tmp_path).read("weituo", 20210609, ("price",))


def test_real_float_quote_with_null_is_not_rescaled(tmp_path):
    _write(tmp_path, "weituo", [10.0, None], physical_type=pa.float64())
    frame = L2InputAdapter(tmp_path).read("weituo", 20210609, ("price", "money"))
    assert frame["price"].iloc[0] == 10.0
    assert frame["money"].iloc[0] == 1000.0
    assert pd.isna(frame["money"].iloc[1])


def test_nullable_integer_physical_money_is_rescaled_and_fills_not_merged(tmp_path):
    path = input_path(tmp_path, "chengjiao", 20210609)
    path.parent.mkdir(parents=True)
    schema = pa.schema([pa.field("money", pa.int64()), pa.field("buy_order_num", pa.uint32()), pa.field("sell_order_num", pa.uint32())], metadata={b"money_scale": b"10000"})
    pq.write_table(pa.table({"money": [10_000_000, 10_000_000, None], "buy_order_num": [10, 10, 10], "sell_order_num": [20, 21, 21]}, schema=schema), path)
    frame = L2InputAdapter(tmp_path).read("chengjiao", 20210609, ("money", "buy_order_num", "sell_order_num"))
    assert len(frame) == 3
    assert frame["money"].iloc[:2].tolist() == [1000.0, 1000.0]
    assert pd.isna(frame["money"].iloc[2])
    assert frame["sell_order_num"].tolist() == [20, 21, 21]
