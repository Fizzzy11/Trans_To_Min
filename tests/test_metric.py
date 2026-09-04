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
