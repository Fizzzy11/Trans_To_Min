from __future__ import annotations

import pandas as pd
import unittest

from trans_to_min.minute import (
    assign_minute_codes,
    assign_minute_labels,
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
            "15:00:00",
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
            [0, 0, 1, 1, 120, 120, -1, 121, 122, 240, 241, 241],
        )
        without_extensions = assign_minute_codes(
            time_ms,
            include_pre_market=False,
            include_post_market=False,
        )
        self.assertEqual(without_extensions.iloc[0], -1)
        self.assertEqual(without_extensions.iloc[-1], -1)
