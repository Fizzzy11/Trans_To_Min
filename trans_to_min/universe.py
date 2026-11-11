"""单日股票全集缓存：执行器读入的同一张逐笔输入在单日计算内只真正求一次股票集合。

执行器读入当日输入后、交给业务计算前，以``symbol_universe_scope(inputs)``登记这些输入；
作用域内``frame_symbols``只对登记的输入表缓存其非空symbol集合，组内各指标补齐242分钟时
复用同一结果，业务无需自行计算或传入股票全集。未登记的表（业务切片、复制或新建的表）
以及作用域外的调用每次计算，行为与逐次计算相同。

登记时把输入表symbol换成独立的分类编码副本并设为只读：登记前已存在的视图不再指向登记
数据；之后由输入派生的视图同样只读，因此直接或经视图原地改写symbol都会报错，不会出现
内容已变而缓存仍返回旧全集。整列替换symbol或增删行使缓存条件不再成立，改为重新计算。
需要修改symbol时先复制输入表。作用域退出时恢复登记时设为只读的编码数组；作用域内由
输入派生的视图仍保持只读。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import warnings
import weakref

import numpy as np
import pandas as pd


@dataclass
class _Registered:
    frame: weakref.ref
    values: weakref.ref
    rows: int
    symbols: frozenset[str] | None = None


@dataclass
class _Scope:
    registered: dict[int, _Registered] = field(default_factory=dict)
    protected: list[weakref.ref] = field(default_factory=list)


_ACTIVE: ContextVar[_Scope | None] = ContextVar(
    "trans_to_min_symbol_universe", default=None
)


@contextmanager
def symbol_universe_scope(
    inputs: Mapping[str, pd.DataFrame] | None = None,
) -> Iterator[None]:
    """登记执行器当日读入的输入并开启缓存；退出即清空，不跨日期、不跨进程。

    必须在执行器刚读入、尚未交给业务代码时调用，保证不存在先于只读保护的视图。
    """

    scope = _Scope()
    for frame in (inputs or {}).values():
        _register(scope, frame)
    token = _ACTIVE.set(scope)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        for reference in reversed(scope.protected):
            codes = reference()
            if codes is not None:
                try:
                    codes.flags.writeable = True
                except ValueError:
                    # 底层基数组已被外部设为只读时保持只读，不越权改写。
                    pass
        scope.registered.clear()
        scope.protected.clear()


def frame_symbols(frame: pd.DataFrame, dataset: str) -> frozenset[str]:
    """返回一张输入表中实际出现的非空symbol集合；登记的输入在作用域内只计算一次。"""

    if "symbol" not in frame:
        raise ValueError(f"{dataset} 缺少symbol字段")
    column = frame["symbol"]
    scope = _ACTIVE.get()
    entry = scope.registered.get(id(frame)) if scope is not None else None
    if (
        entry is not None
        and entry.frame() is frame
        and entry.values() is column.array
        and entry.rows == len(frame)
    ):
        if entry.symbols is None:
            entry.symbols = _compute_symbols(column)
        return entry.symbols
    return _compute_symbols(column)


def _register(scope: _Scope, frame: pd.DataFrame) -> None:
    """登记一张执行器输入：只处理分类symbol列，编码设为只读并记录以便退出恢复。"""

    if not isinstance(frame, pd.DataFrame) or "symbol" not in frame:
        return
    original = frame["symbol"].array
    original_codes = _backing_codes(original)
    if original_codes is None:
        return
    # 换成独立编码副本：登记前已存在的任何视图都不再指向登记输入的数据，无法绕过只读写入。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame["symbol"] = original.copy()
    values = frame["symbol"].array
    codes = _backing_codes(values)
    if codes is None or np.shares_memory(codes, original_codes):
        return
    try:
        entry = _Registered(weakref.ref(frame), weakref.ref(values), len(frame))
        reference = weakref.ref(codes)
    except TypeError:
        # 当前pandas的对象不支持弱引用时只放弃缓存，结果仍逐次计算。
        return
    codes.flags.writeable = False
    if original_codes.flags.writeable:
        # 原本可写的输入在退出时恢复可写；原本只读的保持只读。
        scope.protected.append(reference)
    scope.registered[id(frame)] = entry


def _backing_codes(values) -> np.ndarray | None:
    """返回分类列实际存储编码的数组；无法取得时不缓存。"""

    if not isinstance(values, pd.Categorical):
        return None
    codes = getattr(values, "_ndarray", None)
    return codes if isinstance(codes, np.ndarray) else None


def _compute_symbols(column: pd.Series) -> frozenset[str]:
    """分类列只统计实际出现的编码；未出现的分类和空值都不计入。"""

    values = column.array
    if isinstance(values, pd.Categorical):
        codes = np.asarray(values.codes)
        observed = codes[codes >= 0]
        if not len(observed):
            return frozenset()
        present = np.bincount(observed, minlength=len(values.categories)) > 0
        return frozenset(str(value) for value in values.categories[present])
    return frozenset(str(value) for value in column.dropna().unique().tolist())
