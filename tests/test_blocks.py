"""分块指标组：真实Parquet链路、与整天计算一致、分块数与线程数不影响结果。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import types
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import RunConfig, input_path
from trans_to_min.adapter import L2InputAdapter
from trans_to_min.blocks import BlockMetricGroup, CellValues, _forbid_pandas

from block_demo_fixture import DemoBlockGroup, DuplicateKeys, ListResult, MissingMetric
from trans_to_min.group import MinuteMetricGroup
from trans_to_min.group_runner import MinuteMetricGroupRunner
from trans_to_min.metric import MinuteMetric
from trans_to_min.storage import month_path
from trans_to_min.universe import symbol_universe_scope


TRADE_DATE = date(2025, 1, 2)
COLUMNS = ("symbol", "time_ms", "volume", "price", "order_num")


class _Base:
    inputs = ("chengjiao",)
    input_columns = {"chengjiao": COLUMNS}

    def prepare_events(self, trade_date, inputs):
        return inputs["chengjiao"]


class EventCount(_Base, MinuteMetric):
    name = "block_count_min"
    value_type = "uint64"
    empty_value = 0

    def aggregate_minutes(self, events):
        return events.groupby(["symbol", "minute"], observed=True).size().astype(
            "uint64").rename(self.name).reset_index()


class VolumeSum(_Base, MinuteMetric):
    name = "block_volume_min"
    value_type = "uint64"
    empty_value = 0

    def aggregate_minutes(self, events):
        return events.groupby(["symbol", "minute"], observed=True)["volume"].sum().astype(
            "uint64").rename(self.name).reset_index()


class OrderCount(_Base, MinuteMetric):
    name = "block_orders_min"
    value_type = "uint64"
    empty_value = 0

    def aggregate_minutes(self, events):
        return events.groupby(["symbol", "minute"], observed=True)["order_num"].nunique().astype(
            "uint64").rename(self.name).reset_index()


class KnownMoney(_Base, MinuteMetric):
    """已知价的价×量之和；有事件但全缺价为NULL，无事件也为NULL。"""

    name = "block_money_min"
    value_type = "float64"
    empty_value = None

    def aggregate_minutes(self, events):
        known = events.loc[events["price"].notna()]
        money = (known["price"] * known["volume"].astype("float64")).groupby(
            [known["symbol"], known["minute"]], observed=True).sum()
        return money.rename(self.name).reset_index()


METRICS = (EventCount, VolumeSum, OrderCount, KnownMoney)


def _write_input(root: Path, shuffle: bool) -> None:
    random = np.random.default_rng(7)
    rows = 6000
    symbols = np.array([f"{600000 + i:06d}.XSHG" for i in range(23)])
    symbol = np.sort(random.integers(0, len(symbols), rows))
    # 覆盖盘前、午休、15:00整点及盘后。
    time = random.choice(
        np.array([33_000_000, 34_200_001, 36_000_000, 41_400_000, 46_800_000,
                  50_000_123, 54_000_000, 54_000_001, 55_000_000]), rows)
    order = np.lexsort((time, symbol))
    price = random.random(rows) * 20
    price[random.random(rows) < 0.2] = np.nan
    table = pd.DataFrame({
        "symbol": symbols[symbol][order],
        "time": time[order].astype(np.uint32),
        "volume": random.integers(0, 5000, rows).astype(np.uint32),
        "price": price,
        "order_num": random.integers(1, 40, rows).astype(np.uint64),
    })
    # 一只只有盘后记录的股票，以及一只全缺价的股票。
    table.loc[table["symbol"] == symbols[3], "time"] = np.uint32(55_000_000)
    table.loc[table["symbol"] == symbols[5], "price"] = np.nan
    if shuffle:
        table = table.sample(frac=1, random_state=3).reset_index(drop=True)
    path = input_path(root, "chengjiao", TRADE_DATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    arrow = arrow.append_column("date", pa.array([TRADE_DATE] * len(table), pa.date32()))
    pq.write_table(arrow, path)


def _day_frames(group, input_root, **kwargs):
    inputs = {"chengjiao": L2InputAdapter(input_root).read(
        "chengjiao", TRADE_DATE, COLUMNS)}
    with symbol_universe_scope(inputs):
        return {metric.name: frame for metric, frame in group.iter_compute_day(
            TRADE_DATE, inputs, **kwargs)}


class BlockGroupTest(unittest.TestCase):
    def test_blocks_match_whole_day_metrics_for_sorted_and_shuffled_input(self) -> None:
        """分块结果与普通整天指标一致；分块数、线程数、源顺序都不改变结果。"""

        for shuffle in (False, True):
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                _write_input(root, shuffle)
                plain = _day_frames(
                    MinuteMetricGroup("plain_group", [cls() for cls in METRICS]), root)
                group = DemoBlockGroup("demo_block_group", [cls() for cls in METRICS])
                single = _day_frames(group, root)
                many = _day_frames(group, root, config=RunConfig(
                    input_root=root, block_row_budget=500, block_threads=3))
                for name, expected in plain.items():
                    pd.testing.assert_frame_equal(single[name], many[name])
                    if name == "block_money_min":
                        pd.testing.assert_frame_equal(
                            single[name], expected, check_exact=False, rtol=1e-12)
                    else:
                        pd.testing.assert_frame_equal(single[name], expected)
                money = single["block_money_min"]
                self.assertTrue(money.loc[money["symbol"] == "600005.XSHG",
                                          "block_money_min"].isna().all())

    def test_runner_publishes_block_group_from_real_parquet(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_input(root / "input", shuffle=True)
            group = DemoBlockGroup("demo_block_group", [cls() for cls in METRICS])
            config = RunConfig(input_root=root / "input", output_root=root / "output",
                               date_workers=1, block_row_budget=700, block_threads=2)
            result = MinuteMetricGroupRunner(group, config).run([TRADE_DATE])
            self.assertEqual(result["status"], "complete")
            restored = pd.read_parquet(month_path(
                root / "output", "block_volume_min", "202501"))
            self.assertEqual(len(restored), 23 * 242)

    def test_invalid_block_results_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_input(root, shuffle=False)
            for cls, error, message in (
                (DuplicateKeys, ValueError, "格键重复"),
                (MissingMetric, RuntimeError, "恰好返回"),
                (ListResult, TypeError, "numpy数组"),
            ):
                with self.assertRaisesRegex(error, message):
                    _day_frames(cls("bad_group", [VolumeSum()]), root)

    def test_block_group_rejects_pandas(self) -> None:
        """分块指标组所在模块导入pandas，或compute_block引用pandas，定义时即报错。"""

        with self.assertRaisesRegex(TypeError, "导入了pandas"):
            class InPandasModule(BlockMetricGroup):
                def compute_block(self, block, metric_names):
                    return {}

        def compute_block(self, block, metric_names):
            return pd.DataFrame()

        module = types.ModuleType("pandas_free_fixture")
        sys.modules[module.__name__] = module
        try:
            namespace = {"compute_block": compute_block, "__module__": module.__name__}
            with self.assertRaisesRegex(TypeError, "不得使用pandas"):
                _forbid_pandas(type("UsesPd", (), namespace))
            _forbid_pandas(type("Clean", (), {"__module__": module.__name__}))
        finally:
            del sys.modules[module.__name__]

    def test_group_declaration_rules(self) -> None:
        class NoTime(MinuteMetric):
            name = "no_time_min"
            inputs = ("chengjiao",)
            input_columns = {"chengjiao": ("symbol", "timestamp")}

            def prepare_events(self, trade_date, inputs):
                return inputs["chengjiao"]

        class PostMarket(VolumeSum):
            name = "post_volume_min"
            include_post_market = True

        with self.assertRaisesRegex(ValueError, "time_ms"):
            DemoBlockGroup("bad_group", [NoTime()])
        with self.assertRaisesRegex(ValueError, "盘前/盘后"):
            DemoBlockGroup("bad_group", [VolumeSum(), PostMarket()])


if __name__ == "__main__":
    unittest.main()
