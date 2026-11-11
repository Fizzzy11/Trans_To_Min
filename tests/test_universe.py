"""1.2.4单日股票全集缓存：框架自动只求一次，结果与逐次计算一致。"""

from __future__ import annotations

from datetime import date
import gc
import warnings
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from trans_to_min import universe
from trans_to_min.group import MinuteMetricGroup
from trans_to_min.metric import MinuteMetric
from trans_to_min.minute import minute_labels
from trans_to_min.universe import frame_symbols, symbol_universe_scope


DAY = date(2025, 1, 2)


class TradeCount(MinuteMetric):
    """只用成交的测试指标。"""

    name = "universe_trade_count_min"
    inputs = ("chengjiao",)
    input_columns = {"chengjiao": ("symbol", "time_ms")}
    value_type = "uint64"

    def prepare_events(self, trade_date, inputs):
        return inputs["chengjiao"]

    def aggregate_minutes(self, events):
        return (
            events.groupby(["symbol", "minute"], observed=True)
            .size()
            .astype("uint64")
            .rename(self.name)
            .reset_index()
        )


def numbered(index: int) -> type[TradeCount]:
    return type(f"TradeCount{index}", (TradeCount,), {"name": f"universe_trade_count_{index}_min"})


class OrderAndTradeCount(TradeCount):
    """声明委托+成交两张输入的测试指标，全集取两表并集。"""

    name = "universe_order_trade_count_min"
    inputs = ("weituo", "chengjiao")
    input_columns = {"weituo": ("symbol", "time_ms"), "chengjiao": ("symbol", "time_ms")}


def frame(symbols, times=None) -> pd.DataFrame:
    count = len(symbols)
    return pd.DataFrame(
        {
            "symbol": pd.Categorical(symbols),
            "time_ms": pd.Series(times or [34_200_001] * count, dtype="uint32"),
        }
    )


def codes_of(data: pd.DataFrame) -> np.ndarray:
    return data["symbol"].array._ndarray


class FrameSymbolsTest(unittest.TestCase):
    def test_categorical_fast_path_matches_dropna_unique(self) -> None:
        rng = np.random.default_rng(1)
        pool = np.array([f"{index:06d}.XSHE" for index in range(300)], dtype=object)
        for case in range(20):
            values = rng.choice(pool, size=int(rng.integers(0, 2000))).astype(object)
            if len(values):
                values[rng.random(len(values)) < 0.1] = None
            # 分类表含未出现的值，模拟适配器整日分类表大于子集实际出现股票。
            column = pd.Categorical(values, categories=pool[: 200 + case])
            data = pd.DataFrame({"symbol": column})
            expected = {str(value) for value in data["symbol"].dropna().unique().tolist()}
            self.assertEqual(frame_symbols(data, "chengjiao"), expected)

    def test_non_categorical_column_uses_original_path(self) -> None:
        data = pd.DataFrame({"symbol": ["600000.XSHG", None, "000001.XSHE", "600000.XSHG"]})
        self.assertEqual(frame_symbols(data, "chengjiao"), {"600000.XSHG", "000001.XSHE"})

    def test_empty_and_all_null(self) -> None:
        self.assertEqual(frame_symbols(frame([]), "chengjiao"), frozenset())
        data = pd.DataFrame({"symbol": pd.Categorical([None, None], categories=["000001.XSHE"])})
        self.assertEqual(frame_symbols(data, "chengjiao"), frozenset())

    def test_missing_symbol_column_reports_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "weituo 缺少symbol字段"):
            frame_symbols(pd.DataFrame({"time_ms": [1]}), "weituo")


class ScopeCacheTest(unittest.TestCase):
    def counting(self):
        return mock.patch.object(universe, "_compute_symbols", wraps=universe._compute_symbols)

    def test_group_of_thirteen_scans_each_input_once(self) -> None:
        group = MinuteMetricGroup("universe_group", [numbered(index)() for index in range(13)])
        inputs = {"chengjiao": frame(["600000.XSHG", "000001.XSHE"], [34_200_001, 54_000_001])}
        with self.counting() as counter:
            uncached = list(group.iter_compute_day(DAY, inputs))
            self.assertEqual(counter.call_count, 13)
            counter.reset_mock()
            with symbol_universe_scope(inputs):
                cached = list(group.iter_compute_day(DAY, inputs))
            self.assertEqual(counter.call_count, 1)
        for (_, left), (_, right) in zip(uncached, cached):
            pd.testing.assert_frame_equal(left, right)
            # 仅盘后出现的股票仍在全集内。
            self.assertEqual(sorted(right["symbol"].unique()), ["000001.XSHE", "600000.XSHG"])

    def test_metrics_with_different_inputs_union_their_own_tables(self) -> None:
        inputs = {"weituo": frame(["000001.XSHE", "600000.XSHG"]), "chengjiao": frame(["600000.XSHG"])}
        with self.counting() as counter, symbol_universe_scope(inputs):
            self.assertEqual(TradeCount().symbol_union(inputs), ["600000.XSHG"])
            self.assertEqual(OrderAndTradeCount().symbol_union(inputs), ["000001.XSHE", "600000.XSHG"])
            self.assertEqual(TradeCount().symbol_union(inputs), ["600000.XSHG"])
        self.assertEqual(counter.call_count, 2)

    def test_unregistered_frames_are_neither_cached_nor_protected(self) -> None:
        """业务切片、复制或作用域外的表每次计算，也不被设为只读。"""

        data = frame(["000001.XSHE", "600000.XSHG"])
        part = data.iloc[:1]
        with self.counting() as counter, symbol_universe_scope():
            frame_symbols(data, "chengjiao")
            frame_symbols(data, "chengjiao")
            frame_symbols(part, "chengjiao")
            self.assertTrue(codes_of(data).flags.writeable)
            self.assertTrue(codes_of(part).flags.writeable)
        self.assertEqual(counter.call_count, 3)

    def test_replaced_column_or_changed_rows_recompute(self) -> None:
        data = frame(["600000.XSHG", "000001.XSHE"])
        with symbol_universe_scope({"chengjiao": data}):
            self.assertEqual(frame_symbols(data, "chengjiao"), {"600000.XSHG", "000001.XSHE"})
            data["symbol"] = pd.Categorical(["300750.XSHE", "300750.XSHE"])
            self.assertEqual(frame_symbols(data, "chengjiao"), {"300750.XSHE"})
            data.drop(index=[1], inplace=True)
            self.assertEqual(frame_symbols(data, "chengjiao"), {"300750.XSHE"})

    def test_in_place_symbol_write_is_rejected_inside_scope(self) -> None:
        """登记输入原地改写symbol报错，不能内容已变却返回旧全集。"""

        data = frame(["000001.XSHE", "600000.XSHG"])
        with symbol_universe_scope({"chengjiao": data}):
            self.assertEqual(frame_symbols(data, "chengjiao"), {"000001.XSHE", "600000.XSHG"})
            with self.assertRaisesRegex(ValueError, "read-only"):
                data.iloc[1, data.columns.get_loc("symbol")] = "000001.XSHE"
            with self.assertRaisesRegex(ValueError, "read-only"):
                codes_of(data)[1] = 0
            self.assertEqual(frame_symbols(data, "chengjiao"), set(data["symbol"].dropna()))

    def test_views_taken_inside_scope_cannot_write_through(self) -> None:
        """登记先于业务拿到输入：之后取得的切片和编码视图都只读，无法绕过保护。"""

        data = frame(["000001.XSHE", "600000.XSHG"])
        with symbol_universe_scope({"chengjiao": data}):
            frame_symbols(data, "chengjiao")
            view = codes_of(data)[:]
            part = data.iloc[1:]
            with self.assertRaisesRegex(ValueError, "read-only"):
                view[1] = 0
            with self.assertRaisesRegex(ValueError, "read-only"):
                codes_of(part)[0] = 0
            self.assertEqual(frame_symbols(data, "chengjiao"), set(data["symbol"].dropna()))
            self.assertEqual(frame_symbols(data, "chengjiao"), {"000001.XSHE", "600000.XSHG"})

    def test_views_taken_before_registration_cannot_change_registered_input(self) -> None:
        """登记前取得的编码视图或切片写入的是旧副本，登记输入及其缓存全集都不受影响。"""

        data = frame(["000001.XSHE", "600000.XSHG"])
        view = codes_of(data)[:]
        part = data.iloc[1:]
        with symbol_universe_scope({"chengjiao": data}):
            self.assertEqual(frame_symbols(data, "chengjiao"), {"000001.XSHE", "600000.XSHG"})
            view[1] = 0
            codes_of(part)[0] = 0
            self.assertEqual(list(data["symbol"]), ["000001.XSHE", "600000.XSHG"])
            self.assertEqual(frame_symbols(data, "chengjiao"), set(data["symbol"].dropna()))

    def test_whole_column_assignment_inside_scope_recomputes(self) -> None:
        data = frame(["000001.XSHE", "600000.XSHG"])
        with symbol_universe_scope({"chengjiao": data}):
            frame_symbols(data, "chengjiao")
            with warnings.catch_warnings():
                # pandas 2.x对整列loc赋值给出未来行为提示；无论换新数组还是将来报错，都不能返回旧全集。
                warnings.simplefilter("ignore", FutureWarning)
                try:
                    data.loc[:, "symbol"] = "000001.XSHE"
                except (TypeError, ValueError):
                    return
            self.assertEqual(frame_symbols(data, "chengjiao"), set(data["symbol"].dropna()))
            self.assertEqual(frame_symbols(data, "chengjiao"), {"000001.XSHE"})

    def test_scope_exit_restores_writable_codes(self) -> None:
        data = frame(["000001.XSHE", "600000.XSHG"])
        already = frame(["300750.XSHE"])
        codes_of(already).flags.writeable = False
        with symbol_universe_scope({"chengjiao": data, "weituo": already}):
            frame_symbols(data, "chengjiao")
            self.assertFalse(codes_of(data).flags.writeable)
        self.assertTrue(codes_of(data).flags.writeable)
        self.assertFalse(codes_of(already).flags.writeable)
        data.iloc[1, data.columns.get_loc("symbol")] = "000001.XSHE"
        self.assertEqual(frame_symbols(data, "chengjiao"), {"000001.XSHE"})

    def test_slice_taken_before_scope_stays_writable_after_exit(self) -> None:
        """切片不登记、不受保护；退出后切片与原表都可正常写入。"""

        data = frame(["000001.XSHE", "600000.XSHG"])
        part = data.iloc[:1]
        with symbol_universe_scope({"chengjiao": data}):
            frame_symbols(part, "chengjiao")
            frame_symbols(data, "chengjiao")
        self.assertTrue(codes_of(data).flags.writeable)
        self.assertTrue(codes_of(part).flags.writeable)

    def test_collected_frame_with_reused_id_is_not_served_from_cache(self) -> None:
        for index in range(50):
            data = frame([f"{index:06d}.XSHE"])
            with symbol_universe_scope({"chengjiao": data}):
                self.assertEqual(frame_symbols(data, "chengjiao"), {f"{index:06d}.XSHE"})
                del data
                gc.collect()
                other = frame([f"{index:06d}.XSHG"])
                self.assertEqual(frame_symbols(other, "chengjiao"), {f"{index:06d}.XSHG"})

    def test_cache_is_limited_to_scope(self) -> None:
        data = frame(["600000.XSHG"])
        with self.counting() as counter:
            with symbol_universe_scope({"chengjiao": data}):
                frame_symbols(data, "chengjiao")
                frame_symbols(data, "chengjiao")
            frame_symbols(data, "chengjiao")
            with symbol_universe_scope({"chengjiao": data}):
                frame_symbols(data, "chengjiao")
        self.assertEqual(counter.call_count, 3)


class ExplicitSymbolsTest(unittest.TestCase):
    def sparse(self, metric: MinuteMetric) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["000001.XSHE"],
                "minute": pd.Categorical.from_codes([1], categories=minute_labels(DAY), ordered=True),
                metric.name: pd.Series([1], dtype="uint64"),
            }
        )

    def test_matching_symbols_accepted_and_mismatch_rejected(self) -> None:
        metric = TradeCount()
        inputs = {"chengjiao": frame(["000001.XSHE", "600000.XSHG"])}
        expected = metric.finalize_aggregated_day(DAY, inputs, self.sparse(metric))
        explicit = metric.finalize_aggregated_day(
            DAY, inputs, self.sparse(metric), symbols=["600000.XSHG", "000001.XSHE"]
        )
        pd.testing.assert_frame_equal(explicit, expected)
        for wrong in (["000001.XSHE"], ["000001.XSHE", "600000.XSHG", "300750.XSHE"]):
            with self.assertRaisesRegex(ValueError, "symbol_union"):
                metric.finalize_aggregated_day(DAY, inputs, self.sparse(metric), symbols=wrong)


if __name__ == "__main__":
    unittest.main()
