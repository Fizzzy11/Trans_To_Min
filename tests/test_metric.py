from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow as pa

from trans_to_min.metric import MinuteMetric
from trans_to_min.metrics import MinuteTradeVolume
from trans_to_min.storage import metric_root, output_schema
from trans_to_min.views import trade_order_participation


class NullableTradeCount(MinuteMetric):
    """用于验证可空无符号整数输出的测试指标。"""

    name = "nullable_trade_count_min"
    inputs = ("chengjiao",)
    input_columns = {
        "chengjiao": ("symbol", "timestamp"),
    }
    value_type = "uint64"
    empty_value = None

    def prepare_events(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return inputs["chengjiao"].copy()

    def aggregate_minutes(self, events: pd.DataFrame) -> pd.DataFrame:
        return (
            events.groupby(["symbol", "minute"], observed=True)
            .size()
            .rename(self.name)
            .reset_index()
        )


class ParticipatingOrderCount(NullableTradeCount):
    """验证旧参与订单视图与基类分钟开关衔接的指标。"""

    name = "participating_order_count_min"
    input_columns = {
        "chengjiao": ("symbol", "channel", "time_ms", "buy_order_num", "sell_order_num"),
    }

    def prepare_events(self, trade_date, inputs):
        return trade_order_participation(
            inputs["chengjiao"], trade_date, order_side="buy", time_scope="per_minute",
        )


class MetricTest(unittest.TestCase):
    def test_metric_name_rejects_unsafe_values(self) -> None:
        invalid_names = (
            "minute_invalid",
            "../escape_min",
            "a/b_min",
            r"a\b_min",
            ".hidden_min",
            "double__name_min",
            "Upper_min",
            "含义_min",
        )
        for name in invalid_names:
            with self.subTest(name=name), self.assertRaises(TypeError):
                type(
                    "InvalidMetric",
                    (MinuteMetric,),
                    {
                        "name": name,
                        "inputs": ("chengjiao",),
                        "input_columns": {
                            "chengjiao": ("symbol", "timestamp"),
                        },
                        "prepare_events": lambda self, trade_date, inputs: inputs[
                            "chengjiao"
                        ],
                    },
                )

        with TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                metric_root(Path(temporary), "../escape_min")

    def test_nullable_integer_preserves_empty_minutes(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["000001.XSHE"],
                "timestamp": pd.to_datetime(
                    ["2025-01-02 09:30:00.001"]
                ).tz_localize("Asia/Shanghai"),
            }
        )
        metric = NullableTradeCount()
        result = metric.compute_day(
            date(2025, 1, 2), {"chengjiao": frame}
        )
        self.assertEqual(str(result[metric.name].dtype), "UInt64")
        self.assertEqual(result[metric.name].isna().sum(), 241)
        self.assertEqual(result[metric.name].dropna().iloc[0], 1)

        schema = output_schema(metric.name, metric.value_type)
        table = pa.Table.from_pandas(
            result.reindex(columns=schema.names),
            schema=schema,
            preserve_index=False,
            safe=True,
        )
        self.assertEqual(table.column(metric.name).null_count, 241)

    def test_trade_volume_uses_union_and_fills_zero(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["000001.XSHE", "000001.XSHE", "600000.XSHG"],
                "time_ms": pd.Series(
                    [34_200_001, 34_259_999, 46_800_001], dtype="uint32"
                ),
                "volume": pd.Series([100, 200, 50], dtype="uint64"),
            }
        )
        result = MinuteTradeVolume().compute_day(
            date(2025, 1, 2), {"chengjiao": frame}
        )
        self.assertEqual(len(result), 2 * 242)
        first = result[
            (result["symbol"] == "000001.XSHE")
            & (result["minute"].dt.strftime("%H:%M:%S") == "09:31:00")
        ]
        self.assertEqual(first["trade_volume_min"].iloc[0], 300)
        empty = result[
            (result["symbol"] == "600000.XSHG")
            & (result["minute"].dt.strftime("%H:%M:%S") == "09:31:00")
        ]
        self.assertEqual(empty["trade_volume_min"].iloc[0], 0)

    def test_default_trade_volume_keeps_close_but_excludes_post_market(self) -> None:
        """默认基类在聚合前排除映射后无效的盘后事件，保留输入股票全集。"""

        frame = pd.DataFrame({
            "symbol": ["000001.XSHE"] * 5 + ["600000.XSHG"],
            "time_ms": pd.Series(
                [53_940_000, 53_940_001, 54_000_000, 54_000_001, 55_800_000, 55_800_000],
                dtype="uint32",
            ),
            "volume": pd.Series([1, 2, 3, 10, 100, 200], dtype="uint64"),
        })
        original = frame.copy(deep=True)
        result = MinuteTradeVolume().compute_day(date(2025, 1, 2), {"chengjiao": frame})
        self.assertEqual(len(result), 2 * 242)
        close = result[result["minute"].dt.strftime("%H:%M:%S") == "15:00:00"]
        self.assertEqual(close.set_index("symbol")["trade_volume_min"].to_dict(),
                         {"000001.XSHE": 5, "600000.XSHG": 0})
        self.assertEqual(result["trade_volume_min"].sum(), 6)
        pd.testing.assert_frame_equal(frame, original)

    def test_default_timestamp_metric_retains_null_empty_minutes(self) -> None:
        """时间戳指标的盘后事件被排除，只有盘后记录的股票仍保留空分钟。"""

        frame = pd.DataFrame({
            "symbol": ["000001.XSHE", "000001.XSHE", "600000.XSHG"],
            "timestamp": pd.to_datetime(
                ["2025-01-02 15:00:00", "2025-01-02 15:00:00.001",
                 "2025-01-02 15:30:00"], format="mixed",
            ).tz_localize("Asia/Shanghai"),
        })
        metric = NullableTradeCount()
        result = metric.compute_day(date(2025, 1, 2), {"chengjiao": frame})
        self.assertEqual(len(result), 2 * 242)
        self.assertEqual(result[metric.name].dropna().tolist(), [1])
        self.assertEqual(result.loc[result[metric.name].notna(), "minute"].dt.strftime("%H:%M:%S").tolist(),
                         ["15:00:00"])
        self.assertTrue(result.loc[result["symbol"] == "600000.XSHG", metric.name].isna().all())

    def test_metric_explicit_post_market_opt_in_remains_available(self) -> None:
        """业务显式开启盘后折入时，兼容旧版聚合结果。"""

        metric = MinuteTradeVolume()
        metric.include_post_market = True
        frame = pd.DataFrame({
            "symbol": ["000001.XSHE"] * 3,
            "time_ms": pd.Series([54_000_000, 54_000_001, 55_800_000], dtype="uint32"),
            "volume": pd.Series([1, 10, 100], dtype="uint64"),
        })
        result = metric.compute_day(date(2025, 1, 2), {"chengjiao": frame})
        self.assertEqual(len(result), 242)
        close = result[result["minute"].dt.strftime("%H:%M:%S") == "15:00:00"]
        self.assertEqual(close[metric.name].iloc[0], 111)

    def test_order_participation_preserves_metric_post_market_choice(self) -> None:
        """旧视图不提前丢盘后订单，默认排除与显式兼容均由基类最终生效。"""

        frame = pd.DataFrame({
            "symbol": ["000001.XSHE"] * 3,
            "channel": [1, 1, 1],
            "time_ms": [54_000_000, 54_000_001, 55_800_000],
            "buy_order_num": [10, 10, 11],
            "sell_order_num": [20, 21, 22],
        })
        original = frame.copy(deep=True)
        view = trade_order_participation(
            frame, date(2025, 1, 2), order_side="buy", time_scope="per_minute",
        )
        self.assertEqual(view.set_index("order_num")["time_ms"].to_dict(),
                         {10: 54_000_000, 11: 55_800_000})
        for include_post_market, expected_count in ((False, 1), (True, 2)):
            with self.subTest(include_post_market=include_post_market):
                metric = ParticipatingOrderCount()
                metric.include_post_market = include_post_market
                result = metric.compute_day(date(2025, 1, 2), {"chengjiao": frame})
                self.assertEqual(len(result), 242)
                self.assertEqual(result[metric.name].dropna().tolist(), [expected_count])
        pd.testing.assert_frame_equal(frame, original)
