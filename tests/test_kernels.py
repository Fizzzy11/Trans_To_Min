"""分段归约运算与朴素写法的对拍及边界测试。"""

from __future__ import annotations

from collections import defaultdict
import math
import unittest

import numpy as np

from trans_to_min import kernels


def _naive(keys, values, reducer):
    groups = defaultdict(list)
    for key, value in zip(keys.tolist(), values.tolist()):
        groups[key].append(value)
    ordered = sorted(groups)
    return np.array(ordered), [reducer(groups[key]) for key in ordered]


class KernelRandomTest(unittest.TestCase):
    def setUp(self) -> None:
        self.random = np.random.default_rng(20261117)

    def _sorted_keys(self, length: int) -> np.ndarray:
        return np.sort(self.random.integers(0, max(1, length // 5), length)).astype(np.int64)

    def test_sum_count_extremes_first_last_match_naive(self) -> None:
        """随机数据上各归约与Python逐组结果一致。"""

        for length in (1, 2, 17, 1000):
            keys = self._sorted_keys(length)
            volume = self.random.integers(0, 10**6, length).astype(np.uint32)
            money = self.random.random(length) * 1e6
            unique, sums = kernels.sum_by(keys, volume)
            naive_keys, naive_sums = _naive(keys, volume, lambda v: sum(v))
            np.testing.assert_array_equal(unique, naive_keys)
            self.assertEqual(sums.dtype, np.uint64)
            self.assertEqual(sums.tolist(), naive_sums)
            _, float_sums = kernels.sum_by(keys, money)
            _, naive_float = _naive(keys, money, math.fsum)
            np.testing.assert_allclose(float_sums, naive_float, rtol=1e-12)
            _, counts = kernels.count_by(keys)
            self.assertEqual(counts.tolist(), _naive(keys, volume, len)[1])
            for function, reducer in (
                (kernels.min_by, min),
                (kernels.max_by, max),
                (kernels.first_by, lambda v: v[0]),
                (kernels.last_by, lambda v: v[-1]),
            ):
                _, result = function(keys, money)
                self.assertEqual(result.tolist(), _naive(keys, money, reducer)[1])

    def test_distinct_count_matches_naive_in_both_paths(self) -> None:
        """打包路径与超64位回退路径都与集合计数一致。"""

        keys = self._sorted_keys(2000)
        for high in (50, 2**62):
            ids = self.random.integers(0, high, 2000, dtype=np.uint64)
            ids[::3] = ids[0]
            unique, counts = kernels.distinct_count_by(keys, ids)
            naive_keys, naive = _naive(keys, ids, lambda v: len(set(v)))
            np.testing.assert_array_equal(unique, naive_keys)
            self.assertEqual(counts.tolist(), naive)

    def test_group_sum_and_broadcast_on_unsorted_keys(self) -> None:
        """无序键分组后的和、计数及回映射与朴素结果一致，且排序稳定。"""

        stock = self.random.integers(0, 7, 3000)
        order = self.random.integers(1, 400, 3000)
        amount = self.random.integers(0, 10**9, 3000).astype(np.uint64)
        keys = kernels.pack(stock, order)
        groups = kernels.group(keys)
        totals = groups.sum(amount)
        naive_keys, naive_totals = _naive(keys, amount, sum)
        np.testing.assert_array_equal(groups.keys, naive_keys)
        self.assertEqual(totals.tolist(), naive_totals)
        self.assertEqual(groups.count().tolist(), _naive(keys, amount, len)[1])
        lookup = dict(zip(naive_keys.tolist(), naive_totals))
        self.assertEqual(
            groups.broadcast(totals).tolist(), [lookup[key] for key in keys.tolist()]
        )
        rows = np.arange(len(keys))
        _, first_rows = kernels.first_by(groups.take(keys), groups.take(rows))
        self.assertEqual(
            first_rows.tolist(),
            [int(np.flatnonzero(keys == key)[0]) for key in naive_keys.tolist()],
        )

    def test_group_on_sorted_keys_does_not_reorder(self) -> None:
        keys = np.array([1, 1, 3, 5, 5, 5])
        groups = kernels.group(keys)
        self.assertIsNone(groups.order)
        self.assertEqual(groups.inverse.tolist(), [0, 0, 1, 2, 2, 2])


class KernelBoundaryTest(unittest.TestCase):
    def test_empty_inputs(self) -> None:
        empty = np.zeros(0, dtype=np.int64)
        for function in (kernels.sum_by, kernels.min_by, kernels.first_by):
            unique, values = function(empty, np.zeros(0))
            self.assertEqual(len(unique), 0)
            self.assertEqual(len(values), 0)
        unique, counts = kernels.distinct_count_by(empty, np.zeros(0, dtype=np.uint64))
        self.assertEqual((len(unique), len(counts)), (0, 0))
        self.assertEqual(kernels.group(empty).size, 0)

    def test_uint64_overflow_is_reported(self) -> None:
        """段内和超出uint64时报错；接近上限但不溢出时精确返回。"""

        big = np.array([2**63, 2**63], dtype=np.uint64)
        with self.assertRaises(OverflowError):
            kernels.sum_by(np.array([0, 0]), big)
        _, sums = kernels.sum_by(np.array([0, 1]), big)
        self.assertEqual(sums.tolist(), [2**63, 2**63])
        near = np.array([2**63, 2**63 - 1], dtype=np.uint64)
        _, sums = kernels.sum_by(np.array([0, 0]), near)
        self.assertEqual(int(sums[0]), kernels.UINT64_MAX)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            kernels.runs(np.array([2, 1]))
        with self.assertRaises(ValueError):
            kernels.sum_by(np.array([0, 0]), np.array([1.0, np.nan]))
        with self.assertRaises(ValueError):
            kernels.sum_by(np.array([0]), np.array([-1]))
        with self.assertRaises(TypeError):
            kernels.sum_by(np.array([0]), np.array([True]))
        with self.assertRaises(TypeError):
            kernels.runs(np.array([0.5]))
        with self.assertRaises(OverflowError):
            kernels.pack(np.array([2**40]), np.array([2**40]))
        with self.assertRaises(ValueError):
            kernels.cell_keys(np.array([0]), np.array([242]))

    def test_cell_keys_and_dense(self) -> None:
        keys = kernels.cell_keys(np.array([0, 1, 1]), np.array([0, 0, 241]))
        self.assertEqual(keys.tolist(), [0, 242, 483])
        unique, sums = kernels.sum_by(keys, np.array([5, 6, 7], dtype=np.uint64))
        output = kernels.dense(unique, sums, 2 * 242, 0)
        self.assertEqual((int(output[0]), int(output[242]), int(output[483])), (5, 6, 7))
        self.assertEqual(int(output.sum()), 18)


if __name__ == "__main__":
    unittest.main()
