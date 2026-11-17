"""字段编码信息（field_info）与原始整数无损还原（kernels.raw_integers）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import MinuteMetric, input_path, kernels
from trans_to_min.adapter import L2InputAdapter
from trans_to_min.universe import symbol_universe_scope

from block_demo_fixture import FieldInfoProbe


DAY = date(2025, 1, 2)


def _write(root: Path, dataset: str, *, integer_price: bool, with_money: bool = True) -> None:
    rows = 6
    price_raw = np.array([1, 1_479_999, 105_000, 105_100, 99_990, 100_000], dtype=np.int64)
    columns = {
        "symbol": pa.array(["000001.XSHE"] * 3 + ["600000.XSHG"] * 3),
        "date": pa.array([DAY] * rows, pa.date32()),
        "time": pa.array(np.arange(34_200_001, 34_200_001 + rows * 1000, 1000, dtype=np.uint32)),
        "volume": pa.array(np.full(rows, 100, dtype=np.uint32)),
        "price": pa.array(price_raw if integer_price else price_raw / 37),
    }
    metadata = {b"price_scale": b"37", b"money_scale": b"10000"} if integer_price else {}
    if with_money:
        columns["money"] = pa.array(np.full(rows, 1_000_000, dtype=np.int64))
        metadata.setdefault(b"money_scale", b"10000")
    table = pa.table(columns).replace_schema_metadata(metadata)
    path = input_path(root, dataset, DAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


class FieldInfoTest(unittest.TestCase):
    def test_adapter_reports_physical_type_and_scale(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root, "chengjiao", integer_price=True)
            frame = L2InputAdapter(root).read(
                "chengjiao", DAY, ("symbol", "time_ms", "price", "volume", "money"))
            info = frame.attrs["field_info"]
            self.assertEqual(info["price"], {"physical_type": "int64", "scale": 37, "normalized": True})
            self.assertEqual(info["money"]["scale"], 10000)
            self.assertEqual(info["volume"], {"physical_type": "uint32", "scale": None, "normalized": False})
            self.assertEqual(info["time_ms"]["physical_type"], "uint32")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root, "chengjiao", integer_price=False)
            info = L2InputAdapter(root).read("chengjiao", DAY, ("symbol", "price")).attrs["field_info"]
            self.assertEqual(info["price"], {"physical_type": "double", "scale": None, "normalized": False})
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root, "weituo", integer_price=True, with_money=False)
            info = L2InputAdapter(root).read("weituo", DAY, ("symbol", "money")).attrs["field_info"]
            self.assertEqual(info["money"]["physical_type"], "derived")

    def test_block_exposes_read_only_field_info(self) -> None:
        class Price(MinuteMetric):
            name = "probe_price_min"
            inputs = ("chengjiao",)
            input_columns = {"chengjiao": ("symbol", "time_ms", "price")}
            value_type = "uint64"
            empty_value = 0

            def prepare_events(self, trade_date, inputs):
                raise NotImplementedError

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root, "chengjiao", integer_price=True)
            inputs = {"chengjiao": L2InputAdapter(root).read(
                "chengjiao", DAY, Price.input_columns["chengjiao"])}
            FieldInfoProbe.seen.clear()
            with symbol_universe_scope(inputs):
                frames = dict(FieldInfoProbe("probe_group", [Price()]).iter_compute_day(DAY, inputs))
            info = FieldInfoProbe.seen[0]["chengjiao"]["price"]
            self.assertEqual(dict(info), {"physical_type": "int64", "scale": 37, "normalized": True})
            with self.assertRaises(TypeError):
                info["scale"] = 1
            frame = next(iter(frames.values()))
            self.assertEqual(int(frame["probe_price_min"].sum()), 1 + 1_479_999 + 105_000
                             + 105_100 + 99_990 + 100_000)


class RawIntegersTest(unittest.TestCase):
    INFO = {"physical_type": "int64", "scale": 10000, "normalized": True}

    def test_random_round_trip_is_exact(self) -> None:
        raw = np.random.default_rng(3).integers(0, 2**50 // 10000, 200_000)
        np.testing.assert_array_equal(kernels.raw_integers(raw.astype(np.float64) / 10000, self.INFO), raw)

    def test_reported_counterexample(self) -> None:
        """同一浮点数组：按整数scale=37还原得到[1, 1479999]；物理为浮点时拒绝还原。"""

        values = np.array([1, 1_479_999], dtype=np.int64).astype(np.float64) / 37
        integer = {"physical_type": "int64", "scale": 37, "normalized": True}
        self.assertEqual(kernels.raw_integers(values, integer).tolist(), [1, 1_479_999])
        with self.assertRaisesRegex(ValueError, "不是整数缩放编码"):
            kernels.raw_integers(values, {"physical_type": "double", "scale": None, "normalized": False})

    def test_unsafe_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不一致"):
            kernels.raw_integers(np.array([0.1]), {"physical_type": "int64", "scale": 1, "normalized": True})
        with self.assertRaises(OverflowError):
            kernels.raw_integers(np.array([2.0**51]), {"physical_type": "int64", "scale": 1, "normalized": True})
        with self.assertRaisesRegex(ValueError, "NaN"):
            kernels.raw_integers(np.array([np.nan]), self.INFO)
        with self.assertRaisesRegex(ValueError, "不是整数缩放编码"):
            kernels.raw_integers(np.array([1.0]), None)


if __name__ == "__main__":
    unittest.main()
