from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from trans_to_min.minute import (
    assign_minute_codes,
    assign_minute_labels,
    assign_timestamp_minute_codes,
    minute_labels,
)


class MinuteBoundaryTest(unittest.TestCase):
    def test_minute_labels_have_fixed_242_points(self) -> None:
        labels = minute_labels("2025-01-02")
        self.assertEqual(len(labels), 242)
        self.assertEqual(str(labels[0]), "2025-01-02 09:30:00+08:00")
        self.assertEqual(str(labels[120]), "2025-01-02 11:30:00+08:00")
        self.assertEqual(str(labels[121]), "2025-01-02 13:00:00+08:00")
        self.assertEqual(str(labels[-1]), "2025-01-02 15:00:00+08:00")

    def test_assign_minute_labels_matches_fixed_rules(self) -> None:
        values = pd.Series(
            pd.to_datetime(
                [
                    "2025-01-02 09:15:00",
                    "2025-01-02 09:30:00",
                    "2025-01-02 09:30:00.001",
                    "2025-01-02 09:31:00",
                    "2025-01-02 11:30:30",
                    "2025-01-02 11:31:00",
                    "2025-01-02 11:31:00.001",
                    "2025-01-02 12:59:00",
                    "2025-01-02 13:00:00.001",
                    "2025-01-02 14:59:00",
                    "2025-01-02 14:59:00.001",
                    "2025-01-02 15:30:00",
                ],
                format="mixed",
            )
        )
        labels = assign_minute_labels(values, "2025-01-02")
        expected = [
            "09:30:00",
            "09:30:00",
            "09:31:00",
            "09:31:00",
            "11:30:00",
            "11:30:00",
            None,
            "13:00:00",
            "13:01:00",
            "14:59:00",
            "15:00:00",
            None,
        ]
        actual = [
            None if pd.isna(value) else value.strftime("%H:%M:%S")
            for value in labels
        ]
        self.assertEqual(actual, expected)

    def test_integer_minute_codes_match_timestamp_path(self) -> None:
        time_ms = pd.Series(
            [
                33_300_000,
                34_200_000,
                34_200_001,
                34_260_000,
                41_430_000,
                41_460_000,
                41_460_001,
                46_740_000,
                46_800_001,
                53_940_000,
                53_940_001,
                55_800_000,
            ],
            dtype="uint32",
        )
        self.assertEqual(
            assign_minute_codes(time_ms).tolist(),
            [0, 0, 1, 1, 120, 120, -1, 121, 122, 240, 241, -1],
        )
        without_extensions = assign_minute_codes(
            time_ms,
            include_pre_market=False,
            include_post_market=False,
        )
        self.assertEqual(without_extensions.iloc[0], -1)
        self.assertEqual(without_extensions.iloc[-1], -1)

    def test_default_close_boundary_excludes_later_milliseconds(self) -> None:
        """15:00结束标签只接收(14:59, 15:00]内的毫秒事件。"""

        time_ms = pd.Series(
            [53_940_000, 53_940_001, 53_999_999, 54_000_000,
             54_000_001, 55_800_000, 86_399_999],
            dtype="uint32",
        )
        self.assertEqual(
            assign_minute_codes(time_ms).tolist(),
            [240, 241, 241, 241, -1, -1, -1],
        )

    def test_timestamp_defaults_use_local_close_boundary(self) -> None:
        """两个时间戳入口均默认排除盘后，时区转换不改变截止时刻。"""

        values = pd.Series(pd.to_datetime(
            ["2025-01-02 14:59:00", "2025-01-02 14:59:00.001",
             "2025-01-02 14:59:59.999", "2025-01-02 15:00:00",
             "2025-01-02 15:00:00.001", "2025-01-02 15:30:00"],
            format="mixed",
        ))
        for timestamps in (values, values.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")):
            with self.subTest(timezone=str(timestamps.dt.tz)):
                self.assertEqual(
                    assign_timestamp_minute_codes(timestamps).tolist(),
                    [240, 241, 241, 241, -1, -1],
                )
                labels = assign_minute_labels(timestamps, "2025-01-02")
                self.assertEqual(
                    [None if pd.isna(value) else value.strftime("%H:%M:%S")
                     for value in labels],
                    ["14:59:00", "15:00:00", "15:00:00", "15:00:00", None, None],
                )

    def test_explicit_post_market_opt_in_preserves_legacy_mapping(self) -> None:
        """显式开启盘后折入仍可复现旧映射，不增加分钟标签。"""

        time_ms = pd.Series([54_000_001, 55_800_000, 86_399_999], dtype="uint32")
        timestamps = pd.Series(pd.to_datetime(
            ["2025-01-02 15:00:00.001", "2025-01-02 15:30:00",
             "2025-01-02 23:59:59.999"], format="mixed",
        ))
        self.assertEqual(assign_minute_codes(time_ms, include_post_market=True).tolist(), [241] * 3)
        self.assertEqual(
            assign_timestamp_minute_codes(timestamps, include_post_market=True).tolist(),
            [241] * 3,
        )
        labels = assign_minute_labels(timestamps, "2025-01-02", include_post_market=True)
        self.assertEqual(labels.dt.strftime("%H:%M:%S").tolist(), ["15:00:00"] * 3)


def _reference_minute_codes(values, include_pre_market, include_post_market):
    """1.4.0的逐掩码实现，作为查表版minute_codes的对照。"""

    from trans_to_min.minute import (
        MILLISECONDS_PER_MINUTE as M, T0930, T1129, T1131, T1259, T1300, T1459, T1500)

    codes = np.full(len(values), -1, dtype=np.int16)
    if include_pre_market:
        codes[values <= T0930] = 0
    else:
        codes[values == T0930] = 0
    morning = (values > T0930) & (values <= T1129)
    codes[morning] = ((values[morning] - T0930 + M - 1) // M).astype(np.int16)
    codes[(values > T1129) & (values <= T1131)] = 120
    codes[(values >= T1259) & (values <= T1300)] = 121
    afternoon = (values > T1300) & (values <= T1459)
    codes[afternoon] = (121 + (values[afternoon] - T1300 + M - 1) // M).astype(np.int16)
    if include_post_market:
        codes[values > T1459] = 241
    else:
        codes[(values > T1459) & (values <= T1500)] = 241
    return codes


class LookupMinuteCodesTest(unittest.TestCase):
    def test_lookup_matches_reference_at_every_boundary(self) -> None:
        """每个整分钟前后各2秒的全部毫秒及随机全日毫秒上，与逐掩码实现完全一致。"""

        from trans_to_min.minute import minute_codes

        edges = np.arange(0, 86_400_000 + 1, 60_000, dtype=np.int64)
        near = (edges[:, None] + np.arange(-2000, 2001, dtype=np.int64)[None, :]).ravel()
        random = np.random.default_rng(5).integers(0, 86_400_000, 2_000_000)
        values = np.concatenate((near, random))
        values = values[(values >= 0) & (values < 86_400_000)]
        for pre in (False, True):
            for post in (False, True):
                np.testing.assert_array_equal(
                    minute_codes(values, include_pre_market=pre, include_post_market=post),
                    _reference_minute_codes(values, pre, post),
                )
        with self.assertRaises(ValueError):
            minute_codes(np.array([86_400_000]))
        with self.assertRaises(TypeError):
            minute_codes(np.array([1.5]))
