"""有序分段归约运算。

分钟指标的主体计算可以归结为"按某个整数键分段，再对每段做归约"。本模块只用整列
numpy运算实现这些归约，不做逐行Python回调，也不用哈希分组：

- 键已非降序时（例如块内按股票×分钟编码的键），直接用``*_by``函数按相邻段归约；
- 键无序时（例如同一股票内的订单号），先用``group``做一次稳定排序，再在排序结果上归约，
  并可用``Groups.broadcast``把每组结果映射回原始行。

约定：

- 整数求和在uint64域完成，并检测溢出，溢出时报错，不静默回绕；整数输入不得为负。
- 浮点求和固定使用``np.add.reduceat``，按段内行顺序累加；输入不得含NaN或无穷。
  缺值行由调用方先筛掉，本模块不把缺值当作0。
- 计数结果为uint64。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UINT64_MAX = int(np.iinfo(np.uint64).max)

__all__ = [
    "Groups",
    "UINT64_MAX",
    "cell_keys",
    "count_by",
    "dense",
    "distinct_count_by",
    "first_by",
    "group",
    "last_by",
    "max_by",
    "min_by",
    "pack",
    "runs",
    "sum_by",
]


def _integer_array(values, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label}必须是一维数组")
    if array.dtype.kind not in "iu":
        raise TypeError(f"{label}必须是整数数组，实际为{array.dtype}")
    return array


def _values_array(values, length: int, label: str = "值") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label}必须是一维数组")
    if len(array) != length:
        raise ValueError(f"{label}长度{len(array)}与键长度{length}不一致")
    if array.dtype.kind == "b" or array.dtype.kind not in "iuf":
        raise TypeError(f"{label}必须是整数或浮点数组，实际为{array.dtype}")
    return array


def _nonnegative_uint64(array: np.ndarray, label: str) -> np.ndarray:
    if array.dtype.kind == "i":
        if len(array) and int(array.min()) < 0:
            raise ValueError(f"{label}不能为负数")
        return array.astype(np.uint64)
    return array.astype(np.uint64, copy=False)


def _finite_float(array: np.ndarray, label: str) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}含NaN或无穷；缺值行须由调用方先筛掉")
    return values


def _ends(starts: np.ndarray, length: int) -> np.ndarray:
    return np.append(starts[1:], length).astype(np.int64, copy=False)


def runs(keys) -> tuple[np.ndarray, np.ndarray]:
    """返回非降序键的(唯一键, 每段起点)；键无序时报错。"""

    array = _integer_array(keys, "分段键")
    if not len(array):
        return array[:0].copy(), np.zeros(0, dtype=np.int64)
    if (array[1:] < array[:-1]).any():
        raise ValueError("分段键必须非降序；无序键先用group排序")
    starts = np.flatnonzero(array[1:] != array[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), starts.astype(np.int64)))
    return array[starts], starts


def _sum_sorted(values: np.ndarray, starts: np.ndarray, label: str) -> np.ndarray:
    length = len(values)
    if values.dtype.kind == "f":
        values = _finite_float(values, label)
        if not length:
            return np.zeros(0, dtype=np.float64)
        sums = np.add.reduceat(values, starts)
        if not np.isfinite(sums).all():
            raise OverflowError(f"{label}浮点求和溢出")
        return sums
    values = _nonnegative_uint64(values, label)
    if not length:
        return np.zeros(0, dtype=np.uint64)
    sums = np.add.reduceat(values, starts)
    if int(values.max()) * length > UINT64_MAX:
        # 只对可能溢出的段用Python整数精确复核，正常数据不会走到这里。
        counts = np.diff(np.append(starts, length)).astype(np.uint64)
        maxima = np.maximum.reduceat(values, starts)
        risky = np.flatnonzero(maxima > np.uint64(UINT64_MAX) // counts)
        ends = _ends(starts, length)
        for index in risky:
            exact = sum(int(value) for value in values[starts[index]:ends[index]])
            if exact > UINT64_MAX:
                raise OverflowError(f"{label}uint64求和溢出")
    return sums


def sum_by(keys, values, *, label: str = "值") -> tuple[np.ndarray, np.ndarray]:
    """按非降序键分段求和，返回(唯一键, 每段和)。"""

    unique, starts = runs(keys)
    array = _values_array(values, len(np.asarray(keys)), label)
    return unique, _sum_sorted(array, starts, label)


def count_by(keys) -> tuple[np.ndarray, np.ndarray]:
    """按非降序键分段计行数，返回(唯一键, uint64行数)。"""

    array = _integer_array(keys, "分段键")
    unique, starts = runs(array)
    return unique, np.diff(np.append(starts, len(array))).astype(np.uint64)


def _extreme_by(keys, values, ufunc, label: str) -> tuple[np.ndarray, np.ndarray]:
    unique, starts = runs(keys)
    array = _values_array(values, len(np.asarray(keys)), label)
    if array.dtype.kind == "f":
        array = _finite_float(array, label)
    if not len(array):
        return unique, array[:0].copy()
    return unique, ufunc.reduceat(array, starts)


def min_by(keys, values, *, label: str = "值") -> tuple[np.ndarray, np.ndarray]:
    """按非降序键分段取最小值。"""

    return _extreme_by(keys, values, np.minimum, label)


def max_by(keys, values, *, label: str = "值") -> tuple[np.ndarray, np.ndarray]:
    """按非降序键分段取最大值。"""

    return _extreme_by(keys, values, np.maximum, label)


def first_by(keys, values, *, label: str = "值") -> tuple[np.ndarray, np.ndarray]:
    """按非降序键取每段第一行的值（段内按原行顺序）。"""

    unique, starts = runs(keys)
    array = _values_array(values, len(np.asarray(keys)), label)
    return unique, array[starts]


def last_by(keys, values, *, label: str = "值") -> tuple[np.ndarray, np.ndarray]:
    """按非降序键取每段最后一行的值（段内按原行顺序）。"""

    unique, starts = runs(keys)
    array = _values_array(values, len(np.asarray(keys)), label)
    return unique, array[_ends(starts, len(array)) - 1]


def distinct_count_by(keys, ids, *, label: str = "身份") -> tuple[np.ndarray, np.ndarray]:
    """按非降序键分段，统计每段内不同身份的个数。

    身份（如订单号）在段内可以无序。实现为"段序号与身份打包成一个uint64后排序，
    数相邻变化"；打包超过64位时退回双键lexsort。
    """

    array = _integer_array(keys, "分段键")
    unique, starts = runs(array)
    identity = _nonnegative_uint64(_integer_array(ids, label), label)
    length = len(array)
    if len(identity) != length:
        raise ValueError(f"{label}长度{len(identity)}与键长度{length}不一致")
    if not length:
        return unique, np.zeros(0, dtype=np.uint64)
    segment_count = len(starts)
    counts = np.diff(np.append(starts, length))
    segment = np.repeat(np.arange(segment_count, dtype=np.int64), counts)
    id_bits = max(1, int(identity.max()).bit_length())
    segment_bits = max(1, (segment_count - 1).bit_length())
    if id_bits + segment_bits <= 64:
        packed = (segment.astype(np.uint64) << np.uint64(id_bits)) | identity
        packed.sort()
        new = np.empty(length, dtype=bool)
        new[0] = True
        np.not_equal(packed[1:], packed[:-1], out=new[1:])
        owners = (packed[new] >> np.uint64(id_bits)).astype(np.int64)
    else:
        order = np.lexsort((identity, segment))
        sorted_segment, sorted_identity = segment[order], identity[order]
        new = np.empty(length, dtype=bool)
        new[0] = True
        new[1:] = (sorted_segment[1:] != sorted_segment[:-1]) | (
            sorted_identity[1:] != sorted_identity[:-1]
        )
        owners = sorted_segment[new]
    return unique, np.bincount(owners, minlength=segment_count).astype(np.uint64)


def pack(*columns) -> np.ndarray:
    """把若干非负整数列按位拼成一个uint64键，第一列为最高位。

    各列位宽取该列最大值所需位数；总位宽超过64时报错，调用方应减少列或先压缩编码。
    """

    if not columns:
        raise ValueError("pack至少需要一列")
    arrays = [
        _nonnegative_uint64(_integer_array(column, f"第{index + 1}列"), f"第{index + 1}列")
        for index, column in enumerate(columns)
    ]
    length = len(arrays[0])
    if any(len(array) != length for array in arrays):
        raise ValueError("pack各列长度必须一致")
    widths = [max(1, int(array.max()).bit_length()) if length else 1 for array in arrays]
    if sum(widths) > 64:
        raise OverflowError(f"pack总位宽{sum(widths)}超过64位：{widths}")
    packed = np.zeros(length, dtype=np.uint64)
    for array, width in zip(arrays, widths):
        packed <<= np.uint64(width)
        packed |= array
    return packed


def cell_keys(stock_codes, minutes, minute_count: int = 242) -> np.ndarray:
    """返回"股票×分钟"稠密键：股票码×minute_count+分钟序号。"""

    stock = _integer_array(stock_codes, "股票码").astype(np.int64)
    minute = _integer_array(minutes, "分钟序号").astype(np.int64)
    if len(stock) != len(minute):
        raise ValueError("股票码与分钟序号长度不一致")
    if len(minute) and (int(minute.min()) < 0 or int(minute.max()) >= minute_count):
        raise ValueError(f"分钟序号必须在[0, {minute_count})内")
    return stock * minute_count + minute


def dense(keys, values, size: int, fill) -> np.ndarray:
    """把(唯一键, 值)展开成长度为size的稠密数组，缺失位置填fill。"""

    index = _integer_array(keys, "键")
    array = np.asarray(values)
    if len(index) != len(array):
        raise ValueError("键与值长度不一致")
    if len(index) and (int(index.min()) < 0 or int(index.max()) >= size):
        raise ValueError(f"键必须在[0, {size})内")
    output = np.full(size, fill, dtype=array.dtype)
    output[index] = array
    return output


@dataclass(frozen=True)
class Groups:
    """无序键的分组结果；由group创建。

    ``order``为None表示原键已非降序、未重排；否则为稳定排序的行序。
    ``inverse``给出每个原始行所属组的序号。
    """

    keys: np.ndarray
    starts: np.ndarray
    order: np.ndarray | None
    inverse: np.ndarray
    length: int

    @property
    def size(self) -> int:
        return len(self.keys)

    def take(self, values) -> np.ndarray:
        """按分组顺序重排一列值。"""

        array = _values_array(values, self.length)
        return array if self.order is None else array[self.order]

    def sum(self, values, *, label: str = "值") -> np.ndarray:
        """每组求和，规则同sum_by。"""

        return _sum_sorted(self.take(values), self.starts, label)

    def count(self) -> np.ndarray:
        """每组行数（uint64）。"""

        return np.diff(np.append(self.starts, self.length)).astype(np.uint64)

    def broadcast(self, group_values) -> np.ndarray:
        """把每组一个的值映射回原始行顺序。"""

        array = np.asarray(group_values)
        if len(array) != self.size:
            raise ValueError(f"组值长度{len(array)}与组数{self.size}不一致")
        return array[self.inverse]


def group(keys) -> Groups:
    """对整数键（可无序）做一次稳定排序分组。"""

    array = _integer_array(keys, "分组键")
    length = len(array)
    if length and (array[1:] < array[:-1]).any():
        order = np.argsort(array, kind="stable")
        sorted_keys = array[order]
    else:
        order = None
        sorted_keys = array
    unique, starts = runs(sorted_keys)
    segment = np.repeat(
        np.arange(len(starts), dtype=np.int64), np.diff(np.append(starts, length))
    )
    if order is None:
        inverse = segment
    else:
        inverse = np.empty(length, dtype=np.int64)
        inverse[order] = segment
    return Groups(unique, starts, order, inverse, length)
