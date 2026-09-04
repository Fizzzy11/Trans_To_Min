from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min.adapter import L2InputAdapter, input_path


METADATA = {
    b"format_version": b"1",
    b"price_scale": b"10000",
    b"money_scale": b"10000",
    b"time_encoding": b"milliseconds_since_midnight",
    b"type_encoding": b"1=1,2=2,3=U,50=50,85=85",
}


class AdapterTest(unittest.TestCase):
    def test_adapter_normalizes_current_trade_parquet(self) -> None:
        with TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            self._assert_trade_adapter(tmp_path)

    def _assert_trade_adapter(self, tmp_path: Path) -> None:
        schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("date", pa.date32(), nullable=False),
            pa.field("time", pa.uint32(), nullable=False),
            pa.field("buy_order_num", pa.uint32(), nullable=False),
            pa.field("sell_order_num", pa.uint32(), nullable=False),
            pa.field("channel", pa.uint16(), nullable=False),
            pa.field("price", pa.int64(), nullable=False),
            pa.field("volume", pa.uint32(), nullable=False),
            pa.field("money", pa.int64(), nullable=False),
            pa.field("side", pa.uint8(), nullable=False),
        ],
        metadata=METADATA,
    )
        table = pa.table(
        {
            "symbol": ["000001.XSHE"],
            "date": [date(2025, 1, 2)],
            "time": [34_200_001],
            "buy_order_num": [1],
            "sell_order_num": [2],
            "channel": [2011],
            "price": [100_500],
            "volume": [100],
            "money": [10_050_000],
            "side": [49],
        },
        schema=schema,
    )
        path = input_path(tmp_path, "chengjiao", 20250102)
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)

        result = L2InputAdapter(tmp_path).read(
            "chengjiao",
            20250102,
            ("symbol", "timestamp", "price", "volume", "money"),
        )
        self.assertEqual(result["symbol"].tolist(), ["000001.XSHE"])
        self.assertIsInstance(result["symbol"].dtype, pd.CategoricalDtype)
        self.assertEqual(
            str(result["timestamp"].iloc[0]),
            "2025-01-02 09:30:00.001000+08:00",
        )
        self.assertEqual(result["price"].iloc[0], 10.05)
        self.assertEqual(result["volume"].iloc[0], 100)
        self.assertEqual(result["money"].iloc[0], 1005.0)


    def test_adapter_derives_order_money(self) -> None:
        with TemporaryDirectory() as temporary:
            self._assert_order_money(Path(temporary))

    def _assert_order_money(self, tmp_path: Path) -> None:
        schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("date", pa.date32(), nullable=False),
            pa.field("time", pa.uint32(), nullable=False),
            pa.field("order_num", pa.uint32(), nullable=False),
            pa.field("channel", pa.uint16(), nullable=False),
            pa.field("price", pa.int64(), nullable=False),
            pa.field("side", pa.uint8(), nullable=False),
            pa.field("volume", pa.uint32(), nullable=False),
            pa.field("type", pa.uint8(), nullable=False),
        ],
        metadata=METADATA,
    )
        table = pa.table(
        {
            "symbol": ["600000.XSHG"],
            "date": [date(2025, 1, 2)],
            "time": [34_200_001],
            "order_num": [1],
            "channel": [1],
            "price": [94_300],
            "side": [49],
            "volume": [100],
            "type": [50],
        },
        schema=schema,
    )
        path = input_path(tmp_path, "weituo", 20250102)
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)

        result = L2InputAdapter(tmp_path).read(
            "weituo",
            20250102,
            ("symbol", "timestamp", "price", "volume", "money"),
        )
        self.assertEqual(result["price"].iloc[0], 9.43)
        self.assertEqual(result["money"].iloc[0], 943.0)
