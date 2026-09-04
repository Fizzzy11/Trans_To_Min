from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import RunConfig
from trans_to_min.minute import minute_labels
from trans_to_min.storage import (
    output_schema,
    publish_month,
    write_day_frame,
)


METRIC_NAME = "storage_test_min"


def _day_frame(value: date, metric_value: int) -> pd.DataFrame:
    """构造一个股票完整242分钟的测试结果。"""

    return pd.DataFrame(
        {
            "trade_date": value,
            "minute": minute_labels(value),
            "symbol": "000001.XSHE",
            METRIC_NAME: np.full(242, metric_value, dtype="uint64"),
        }
    )


class StorageTest(unittest.TestCase):
    def test_single_pass_merge_handles_cross_date_row_groups(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "202501.parquet"
            schema = output_schema(METRIC_NAME, "uint64")
            first = date(2025, 1, 2)
            middle = date(2025, 1, 3)
            last = date(2025, 1, 4)

            existing = pd.concat(
                [_day_frame(first, 2), _day_frame(last, 4)],
                ignore_index=True,
            )
            # 300行Row Group会跨越两个交易日，用于验证按日期流式读取。
            pq.write_table(
                pa.Table.from_pandas(
                    existing,
                    schema=schema,
                    preserve_index=False,
                    safe=True,
                ),
                destination,
                compression="zstd",
                row_group_size=300,
            )

            config = RunConfig(
                output_root=root,
                date_workers=1,
                row_group_size=100,
                publish_while_computing=False,
            )
            middle_path = root / "20250103.parquet"
            write_day_frame(
                _day_frame(middle, 3),
                middle_path,
                METRIC_NAME,
                "uint64",
                config,
            )
            staging_codec = pq.ParquetFile(middle_path).metadata.row_group(0).column(
                3
            ).compression
            self.assertEqual(staging_codec, "SNAPPY")

            import trans_to_min.storage as storage

            with patch.object(
                storage,
                "_existing_month_batches",
                wraps=storage._existing_month_batches,
            ) as scan:
                publish_month(
                    [middle_path],
                    destination,
                    METRIC_NAME,
                    "uint64",
                    config,
                    {middle},
                )
            self.assertEqual(scan.call_count, 1)

            result = pd.read_parquet(destination)
            self.assertEqual(
                result["trade_date"].drop_duplicates().tolist(),
                [first, middle, last],
            )
            self.assertEqual(
                result.groupby("trade_date", observed=True)[METRIC_NAME]
                .first()
                .tolist(),
                [2, 3, 4],
            )
            final_codec = pq.ParquetFile(destination).metadata.row_group(0).column(
                3
            ).compression
            self.assertEqual(final_codec, "ZSTD")

    def test_single_pass_merge_replaces_only_requested_date(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "202501.parquet"
            target = date(2025, 1, 2)
            schema = output_schema(METRIC_NAME, "uint64")
            pq.write_table(
                pa.Table.from_pandas(
                    _day_frame(target, 1),
                    schema=schema,
                    preserve_index=False,
                    safe=True,
                ),
                destination,
            )
            config = RunConfig(
                output_root=root,
                date_workers=1,
                overwrite_dates=True,
                publish_while_computing=False,
            )
            day_path = root / "20250102.parquet"
            write_day_frame(
                _day_frame(target, 9),
                day_path,
                METRIC_NAME,
                "uint64",
                config,
            )
            publish_month(
                [day_path],
                destination,
                METRIC_NAME,
                "uint64",
                config,
                {target},
            )
            result = pd.read_parquet(destination)
            self.assertEqual(result[METRIC_NAME].unique().tolist(), [9])


if __name__ == "__main__":
    unittest.main()
