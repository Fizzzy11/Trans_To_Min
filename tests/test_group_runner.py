"""同源多指标组的输入共享、独立发布和恢复测试。"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import (
    __version__,
    MinuteConversionRunner,
    MinuteMetric,
    MinuteMetricGroup,
    MinuteMetricGroupRunner,
    RunConfig,
    input_path,
)
from trans_to_min.adapter import L2InputAdapter
from trans_to_min.storage import month_path


TRADE_DATE = date(2025, 1, 2)


class MultiTableCount(MinuteMetric):
    """两个输入表的逐笔总笔数。"""

    name = "multi_table_count_min"
    inputs = ("weituo", "chengjiao")
    input_columns = {
        "weituo": ("symbol", "time_ms"),
        "chengjiao": ("symbol", "time_ms"),
    }
    value_type = "uint64"
    empty_value = 0

    def prepare_events(self, trade_date, inputs):
        return pd.concat(
            [
                inputs["weituo"].loc[:, ["symbol", "time_ms"]],
                inputs["chengjiao"].loc[:, ["symbol", "time_ms"]],
            ],
            ignore_index=True,
        )

    def aggregate_minutes(self, events):
        return (
            events.groupby(["symbol", "minute"], observed=True)
            .size()
            .astype("uint64")
            .rename(self.name)
            .reset_index()
        )


class MultiTableVolume(MinuteMetric):
    """两个输入表的逐笔量之和。"""

    name = "multi_table_volume_min"
    inputs = ("chengjiao", "weituo")
    input_columns = {
        "chengjiao": ("symbol", "time_ms", "volume"),
        "weituo": ("symbol", "time_ms", "volume"),
    }
    value_type = "uint64"
    empty_value = 0

    def prepare_events(self, trade_date, inputs):
        return pd.concat(
            [
                inputs["weituo"].loc[:, ["symbol", "time_ms", "volume"]],
                inputs["chengjiao"].loc[:, ["symbol", "time_ms", "volume"]],
            ],
            ignore_index=True,
        )

    def aggregate_minutes(self, events):
        return (
            events.groupby(["symbol", "minute"], observed=True)["volume"]
            .sum()
            .astype("uint64")
            .rename(self.name)
            .reset_index()
        )


class SingleTableMetric(MinuteMetric):
    """用于验证不兼容输入签名。"""

    name = "single_table_test_min"
    inputs = ("chengjiao",)
    input_columns = {"chengjiao": ("symbol", "time_ms")}

    def prepare_events(self, trade_date, inputs):
        return inputs["chengjiao"].copy()


class ThreeTableCount(MinuteMetric):
    """用于验证三表输入组合的基础字段。"""

    name = "three_table_count_min"
    inputs = ("weituo", "chengjiao", "chedan")
    input_columns = {
        dataset: ("symbol", "time_ms") for dataset in inputs
    }

    def prepare_events(self, trade_date, inputs):
        return inputs["chengjiao"].copy()


class ThreeTableVolume(MinuteMetric):
    """用于验证三表输入组合的字段并集。"""

    name = "three_table_volume_min"
    inputs = ("chedan", "chengjiao", "weituo")
    input_columns = {
        dataset: ("symbol", "time_ms", "volume") for dataset in inputs
    }

    def prepare_events(self, trade_date, inputs):
        return inputs["chengjiao"].copy()


def _write_input(root: Path, dataset: str, volume: int) -> None:
    path = input_path(root, dataset, TRADE_DATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "symbol": pa.array(["000001.XSHE"], type=pa.string()),
            "date": pa.array([TRADE_DATE], type=pa.date32()),
            "time": pa.array([34_200_001], type=pa.uint32()),
            "volume": pa.array([volume], type=pa.uint32()),
        }
    )
    pq.write_table(table, path)


class MetricGroupTest(unittest.TestCase):
    def test_group_merges_columns_and_rejects_different_input_sets(self) -> None:
        group = MinuteMetricGroup(
            "multi_table_group",
            (MultiTableCount(), MultiTableVolume()),
        )
        self.assertEqual(group.inputs, ("weituo", "chengjiao"))
        self.assertEqual(
            group.input_columns,
            {
                "weituo": ("symbol", "time_ms", "volume"),
                "chengjiao": ("symbol", "time_ms", "volume"),
            },
        )
        with self.assertRaisesRegex(ValueError, "相同的逐笔输入表集合"):
            MinuteMetricGroup(
                "invalid_group",
                (MultiTableCount(), SingleTableMetric()),
            )

    def test_three_table_group_merges_each_table_columns(self) -> None:
        group = MinuteMetricGroup(
            "three_table_group",
            (ThreeTableCount(), ThreeTableVolume()),
        )
        self.assertEqual(group.inputs, ("weituo", "chengjiao", "chedan"))
        self.assertEqual(
            group.input_columns,
            {
                dataset: ("symbol", "time_ms", "volume")
                for dataset in group.inputs
            },
        )

    def test_multi_table_group_reads_each_physical_file_once(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            _write_input(input_root, "weituo", 100)
            _write_input(input_root, "chengjiao", 200)
            group = MinuteMetricGroup(
                "multi_table_group",
                (MultiTableCount(), MultiTableVolume()),
            )
            roots = {
                "multi_table_count_min": root / "count_category",
                "multi_table_volume_min": root / "volume_category",
            }
            config = RunConfig(
                input_root=input_root,
                output_root=root / "output",
                date_workers=1,
            )
            original_read = L2InputAdapter.read

            def counting_read(adapter, dataset, trade_date, columns):
                return original_read(adapter, dataset, trade_date, columns)

            with patch.object(
                L2InputAdapter,
                "read",
                autospec=True,
                side_effect=counting_read,
            ) as mocked_read:
                result = MinuteMetricGroupRunner(group, config, roots).run(
                    [TRADE_DATE]
                )

            self.assertEqual(mocked_read.call_count, 2)
            self.assertEqual(result["status"], "complete")
            for metric_name, output_root in roots.items():
                path = month_path(output_root, metric_name, "202501")
                restored = pd.read_parquet(path)
                self.assertEqual(len(restored), 242)
                self.assertTrue(
                    (output_root / metric_name / "manifest.json").is_file()
                )
                manifest = json.loads(
                    (output_root / metric_name / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["framework_version"], __version__)
                self.assertEqual(manifest["metric_group"], "multi_table_group")
            volume = pd.read_parquet(
                month_path(
                    roots["multi_table_volume_min"],
                    "multi_table_volume_min",
                    "202501",
                )
            )
            self.assertEqual(int(volume["multi_table_volume_min"].max()), 300)

    def test_resume_computes_only_missing_metric_dates(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            _write_input(input_root, "weituo", 100)
            _write_input(input_root, "chengjiao", 200)
            roots = {
                "multi_table_count_min": root / "count_category",
                "multi_table_volume_min": root / "volume_category",
            }
            base_config = RunConfig(
                input_root=input_root,
                output_root=roots["multi_table_count_min"],
                date_workers=1,
                publish_while_computing=False,
            )
            MinuteConversionRunner(MultiTableCount(), base_config).run([TRADE_DATE])

            group = MinuteMetricGroup(
                "multi_table_group",
                (MultiTableCount(), MultiTableVolume()),
            )
            resume_config = RunConfig(
                input_root=input_root,
                output_root=root / "output",
                date_workers=1,
                publish_while_computing=False,
                resume=True,
            )
            result = MinuteMetricGroupRunner(group, resume_config, roots).run(
                [TRADE_DATE]
            )
            count_state = result["metrics"]["multi_table_count_min"]
            volume_state = result["metrics"]["multi_table_volume_min"]
            self.assertEqual(count_state["progress"]["pending"], 0)
            self.assertEqual(count_state["progress"]["skipped"], 1)
            self.assertEqual(volume_state["progress"]["completed"], 1)
            self.assertTrue(
                month_path(
                    roots["multi_table_volume_min"],
                    "multi_table_volume_min",
                    "202501",
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
