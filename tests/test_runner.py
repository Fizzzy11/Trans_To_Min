from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import (
    MinuteConversionRunner,
    RunConfig,
    input_path,
    recommend_date_workers,
)
from trans_to_min.metric import MinuteMetric
from trans_to_min.metrics import MinuteTradeVolume
from trans_to_min.storage import month_path


def _write_trade(root: Path, value_date: date, volume: int) -> None:
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
        metadata={
            b"format_version": b"1",
            b"price_scale": b"10000",
            b"money_scale": b"10000",
            b"time_encoding": b"milliseconds_since_midnight",
            b"type_encoding": b"1=1,2=2,3=U,50=50,85=85",
        },
    )
    table = pa.table(
        {
            "symbol": ["000001.XSHE"],
            "date": [value_date],
            "time": [34_200_001],
            "buy_order_num": [1],
            "sell_order_num": [2],
            "channel": [2011],
            "price": [100_000],
            "volume": [volume],
            "money": [volume * 100_000],
            "side": [49],
        },
        schema=schema,
    )
    path = input_path(root, "chengjiao", value_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _touch_input(root: Path, dataset: str, value_date: date) -> None:
    """创建只供日期发现测试使用的占位文件。"""

    path = input_path(root, dataset, value_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class MultiInputMetric(MinuteMetric):
    """用于验证多输入日期完整性的测试指标。"""

    name = "multi_input_test_min"
    inputs = ("weituo", "chengjiao")
    input_columns = {
        "weituo": ("symbol", "timestamp"),
        "chengjiao": ("symbol", "timestamp"),
    }

    def prepare_events(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return pd.concat(inputs.values(), ignore_index=True)


class RunnerTest(unittest.TestCase):
    def test_config_rejects_invalid_parallel_options(self) -> None:
        self.assertEqual(RunConfig().date_workers, 0)
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            RunConfig(date_workers=-1)
        with self.assertRaisesRegex(ValueError, "arrow_threads_per_worker"):
            RunConfig(arrow_threads_per_worker=0)
        with self.assertRaisesRegex(ValueError, "Snappy"):
            RunConfig(staging_compression_level=1)
        with self.assertRaisesRegex(ValueError, "safety_factor"):
            RunConfig(adaptive_memory_safety_factor=0.9)
        with self.assertRaisesRegex(ValueError, "不能同时"):
            RunConfig(overwrite_dates=True, resume=True)

    def test_multi_input_dates_only_validate_target_range(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            outside = date(2025, 1, 2)
            target = date(2025, 1, 3)
            _touch_input(input_root, "chengjiao", outside)
            _touch_input(input_root, "chengjiao", target)
            _touch_input(input_root, "weituo", target)

            runner = MinuteConversionRunner(
                MultiInputMetric(),
                RunConfig(
                    input_root=input_root,
                    output_root=output_root,
                    date_workers=1,
                ),
            )
            self.assertEqual(
                runner.discover_dates(target, target),
                [target],
            )
            with self.assertRaisesRegex(
                RuntimeError, "weituo.*20250102"
            ):
                runner.discover_dates(outside, target)
            with self.assertRaisesRegex(ValueError, "start_date"):
                runner.discover_dates(target, outside)
            with self.assertRaisesRegex(
                FileNotFoundError, "2025-01-02:weituo"
            ):
                runner.run([outside])
            self.assertFalse(output_root.exists())

    def test_runner_adds_dates_and_rejects_implicit_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            self._assert_runner(Path(temporary))

    def _assert_runner(self, tmp_path: Path) -> None:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        first = date(2025, 1, 2)
        second = date(2025, 1, 3)
        _write_trade(input_root, first, 100)
        _write_trade(input_root, second, 200)

        runner = MinuteConversionRunner(
            MinuteTradeVolume(),
            RunConfig(input_root=input_root, output_root=output_root, date_workers=1),
        )
        # 先发布较晚日期，再插入较早日期，验证流式合并仍保持日期主序。
        runner.run([second])
        runner.run([first])

        path = month_path(output_root, "trade_volume_min", "202501")
        result = pd.read_parquet(path)
        self.assertEqual(len(result), 2 * 242)
        self.assertEqual(result["trade_date"].nunique(), 2)
        self.assertEqual(
            result["trade_date"].drop_duplicates().tolist(),
            [first, second],
        )

        with self.assertRaisesRegex(FileExistsError, "overwrite_dates"):
            runner.run([first])

    def test_runner_publishes_monthly_and_resumes_existing_dates(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            january = date(2025, 1, 31)
            february = date(2025, 2, 5)
            _write_trade(input_root, january, 100)
            _write_trade(input_root, february, 200)

            runner = MinuteConversionRunner(
                MinuteTradeVolume(),
                RunConfig(
                    input_root=input_root,
                    output_root=output_root,
                    date_workers=2,
                    arrow_threads_per_worker=1,
                ),
            )
            first = runner.run([january])
            self.assertEqual(first["progress"]["completed"], 1)
            self.assertTrue(
                month_path(
                    output_root, "trade_volume_min", "202501"
                ).is_file()
            )

            resume_runner = MinuteConversionRunner(
                MinuteTradeVolume(),
                RunConfig(
                    input_root=input_root,
                    output_root=output_root,
                    date_workers=2,
                    arrow_threads_per_worker=1,
                    resume=True,
                ),
            )
            resumed = resume_runner.run([january, february])
            self.assertEqual(resumed["skipped_dates"], [january.isoformat()])
            self.assertEqual(resumed["progress"]["completed"], 1)
            self.assertEqual(set(resumed["months"]), {"202502"})
            self.assertTrue(
                month_path(
                    output_root, "trade_volume_min", "202502"
                ).is_file()
            )

            noop = resume_runner.run([january, february])
            self.assertEqual(noop["progress"]["pending"], 0)
            self.assertEqual(noop["progress"]["skipped"], 2)
            self.assertEqual(noop["progress"]["completed"], 0)
            self.assertEqual(noop["months"], {})

    def test_auto_parallelism_and_safe_process_pool(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            dates = [date(2025, 1, 2), date(2025, 1, 3)]
            for index, value in enumerate(dates, start=1):
                _write_trade(input_root, value, index * 100)

            auto_config = RunConfig(
                input_root=input_root,
                output_root=output_root,
                date_workers=0,
                arrow_threads_per_worker=1,
            )
            recommendation = recommend_date_workers(
                MinuteTradeVolume(), auto_config, dates
            )
            self.assertGreaterEqual(recommendation.workers, 1)
            self.assertLessEqual(recommendation.workers, len(dates))
            self.assertGreater(recommendation.estimated_input_bytes_per_day, 0)

            runner = MinuteConversionRunner(
                MinuteTradeVolume(),
                RunConfig(
                    input_root=input_root,
                    output_root=output_root,
                    date_workers=2,
                    arrow_threads_per_worker=1,
                    publish_while_computing=False,
                ),
            )
            result = runner.run(dates)
            self.assertEqual(result["progress"]["completed"], 2)
            self.assertEqual(result["parallelism"]["effective_date_workers"], 2)
            self.assertEqual(len(result["parallelism"]["history"]), 1)
            self.assertGreater(
                result["parallelism"]["history"][0]["observed_peak_rss_bytes"],
                0,
            )

    def test_auto_parallelism_is_recalibrated_between_months(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            dates = [date(2025, 1, 31), date(2025, 2, 5)]
            for value in dates:
                _write_trade(input_root, value, 100)
            runner = MinuteConversionRunner(
                MinuteTradeVolume(),
                RunConfig(
                    input_root=input_root,
                    output_root=output_root,
                    date_workers=0,
                    arrow_threads_per_worker=1,
                    publish_while_computing=False,
                ),
            )
            result = runner.run(dates)
            history = result["parallelism"]["history"]
            self.assertEqual([item["month"] for item in history], ["202501", "202502"])
            self.assertIn("adaptive_recommendation", history[0])
            self.assertEqual(history[0]["next_month"], "202502")
            self.assertGreater(history[0]["observed_peak_rss_bytes"], 0)
