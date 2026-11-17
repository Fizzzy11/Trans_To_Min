"""按股票分块计算的指标组。

适用于"每只股票的分钟值只依赖该股票自己的逐笔数据"的指标。框架把一天的输入按股票
切成若干块，块在日期进程内用线程并行计算，再按股票顺序合并为标准单指标日长表，
之后的暂存、发布与校验沿用普通指标组的流程。

业务只实现 :meth:`BlockMetricGroup.compute_block`：输入一个 :class:`Block`（各表所需列的
numpy数组，已按股票、分钟排好，同一分钟内保持源顺序），返回每个指标在块内各
"股票×分钟"格的值。分段求和、计数、去重计数等用 :mod:`trans_to_min.kernels`。

分块指标组所在模块不得导入pandas，``compute_block``只能使用numpy与kernels；
pandas只允许出现在独立参考等其他模块。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
import inspect
import os
import re
import sys
import types

import numpy as np
import pandas as pd

from .config import MINUTE_COUNT, RunConfig
from .group import MinuteMetricGroup
from .metric import MinuteMetric, PANDAS_NULLABLE_INTEGER_TYPES
from .minute import minute_codes, minute_labels


RESERVED_COLUMNS = ("stock", "minute")
DEFAULT_BLOCK_ROW_BUDGET = 250_000
_PANDAS_SOURCE = re.compile(r"\bpd\.|\bpandas\b")


@dataclass(frozen=True)
class Block:
    """一组连续股票的全天输入。

    ``tables[dataset]``包含所声明字段的numpy数组（不含symbol），另有两列：
    ``stock``为块内股票序号（0起，对应``symbols``），``minute``为0~241的分钟序号，
    不属于分钟轴的行为-1。行按股票排序；分钟轴内的行按分钟非降序，同一分钟内保持
    源文件顺序；轴外行的位置不作保证，业务按``minute >= 0``筛选后使用。因此筛选后的
    格键非降序，可直接用kernels的``*_by``函数；按订单号等其他键归约时用``kernels.group``。
    数组均为只读，业务不得原地修改。
    """

    trade_date: date
    symbols: tuple[str, ...]
    stock_offset: int
    tables: Mapping[str, Mapping[str, np.ndarray]]

    @property
    def stock_count(self) -> int:
        return len(self.symbols)

    @property
    def cell_count(self) -> int:
        """块内"股票×分钟"格数，即stock_count×242。"""

        return self.stock_count * MINUTE_COUNT

    def rows(self, dataset: str) -> int:
        return len(self.tables[dataset]["stock"])

    def cells(self, dataset: str, mask: np.ndarray | None = None) -> np.ndarray:
        """返回所选行的块内格键（stock×242+minute），选中行不得落在分钟轴外。"""

        table = self.tables[dataset]
        stock, minute = table["stock"], table["minute"]
        if mask is not None:
            stock, minute = stock[mask], minute[mask]
        if len(minute) and int(minute.min()) < 0:
            raise ValueError(f"{dataset}所选行含分钟轴外记录，先按minute>=0筛选")
        return stock.astype(np.int64) * MINUTE_COUNT + minute


@dataclass(frozen=True)
class CellValues:
    """一个指标在块内的稀疏结果：格键唯一，未出现的格按指标的empty_value填充。

    三者都必须是numpy数组。``nulls``可选，标记值为NULL的格（例如有事件但全缺价的金额）。
    """

    keys: np.ndarray
    values: np.ndarray
    nulls: np.ndarray | None = None


class BlockMetricGroup(MinuteMetricGroup):
    """按股票分块并行计算的指标组；业务子类实现compute_block。"""

    stock_separable = True

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _forbid_pandas(cls)

    def __init__(self, name: str, metrics: Sequence[MinuteMetric]) -> None:
        super().__init__(name, metrics)
        flags = {(m.include_pre_market, m.include_post_market) for m in self.metrics}
        if len(flags) != 1:
            raise ValueError("分块指标组内各指标的盘前/盘后折入设置必须一致")
        self._include_pre_market, self._include_post_market = flags.pop()
        for dataset, columns in self.input_columns.items():
            if "time_ms" not in columns:
                raise ValueError(f"分块指标组的{dataset}必须声明time_ms字段")
            if "timestamp" in columns:
                raise ValueError("分块指标组使用time_ms，不声明timestamp")
            clash = sorted(set(columns) & set(RESERVED_COLUMNS))
            if clash:
                raise ValueError(f"{dataset}字段名与分块保留列冲突：{clash}")

    def compute_block(
        self, block: Block, metric_names: tuple[str, ...]
    ) -> Mapping[str, CellValues]:
        """计算一个块，返回{指标名: CellValues}，须恰好覆盖metric_names。"""

        raise NotImplementedError(f"{type(self).__name__}需要实现compute_block")

    def iter_compute_day(
        self,
        trade_date: date,
        inputs: Mapping[str, pd.DataFrame],
        metric_names: Iterable[str] | None = None,
        *,
        config: RunConfig | None = None,
    ) -> Iterator[tuple[MinuteMetric, pd.DataFrame]]:
        missing = sorted(set(self.inputs) - set(inputs))
        if missing:
            raise ValueError(f"指标组缺少逐笔输入：{missing}")
        selected = self.select_metrics(metric_names)
        names = tuple(metric.name for metric in selected)
        symbols = selected[0].symbol_union(dict(inputs))
        if not symbols:
            for metric in selected:
                yield metric, metric._empty_output()
            return
        tables, bounds, stock_reordered = self._stock_tables(inputs, symbols)
        budget, threads = _block_options(config)
        spans = _block_spans(bounds, len(symbols), budget)

        def run(span: tuple[int, int]):
            return self._run_block(trade_date, symbols, tables, bounds, span, names)

        if threads <= 1 or len(spans) == 1:
            outcomes = [run(span) for span in spans]
        else:
            with ThreadPoolExecutor(max_workers=min(threads, len(spans))) as pool:
                outcomes = list(pool.map(run, spans))
        stats = {
            "blocks": len(spans),
            "threads": min(threads, len(spans)),
            "stock_reordered_tables": stock_reordered,
            "minute_reordered_blocks": sum(reordered for _, _, reordered in outcomes),
        }
        placed = [(offset, result) for offset, result, _ in outcomes]
        for metric in selected:
            frame = _assemble(metric, trade_date, symbols, placed)
            frame.attrs["block_stats"] = stats
            yield metric, frame

    def _run_block(
        self,
        trade_date: date,
        symbols: list[str],
        tables: dict[str, dict[str, np.ndarray]],
        bounds: dict[str, np.ndarray],
        span: tuple[int, int],
        names: tuple[str, ...],
    ) -> tuple[int, dict[str, CellValues], bool]:
        """在线程内构造一个块（分钟编码、块内顺序检查）并计算、检查结果。"""

        start, stop = span
        reordered = False
        block_tables = {}
        for dataset, table in tables.items():
            low, high = int(bounds[dataset][start]), int(bounds[dataset][stop])
            view = {name: values[low:high] for name, values in table.items()}
            view["stock"] = view["stock"] - np.int32(start)
            view["minute"] = minute_codes(
                view["time_ms"],
                include_pre_market=self._include_pre_market,
                include_post_market=self._include_post_market,
            )
            # 上游已按symbol→time排序时，分钟轴内各行的(股票, 分钟)天然非降序，这里只检查一遍；
            # 轴外记录（午休、盘后等）位置不影响聚合，不参与检查。不成立时在块内稳定重排：
            # 轴外记录排在该股票最前，同分钟内保持源顺序。
            minute = view["minute"]
            inside = minute >= 0
            cells = view["stock"][inside].astype(np.int64) * MINUTE_COUNT + minute[inside]
            if len(cells) > 1 and (cells[1:] < cells[:-1]).any():
                key = view["stock"].astype(np.int64) * (MINUTE_COUNT + 1) + minute + 1
                order = np.argsort(key, kind="stable")
                view = {name: values[order] for name, values in view.items()}
                reordered = True
            for values in view.values():
                values.flags.writeable = False
            block_tables[dataset] = view
        block = Block(trade_date, tuple(symbols[start:stop]), start, block_tables)
        return start, self._checked_block(block, names), reordered

    def _checked_block(self, block: Block, names: tuple[str, ...]) -> dict[str, CellValues]:
        result = dict(self.compute_block(block, names))
        if set(result) != set(names):
            raise RuntimeError(
                f"compute_block须恰好返回所选指标：缺少{sorted(set(names) - set(result))}，"
                f"多出{sorted(set(result) - set(names))}"
            )
        for name, cells in result.items():
            if not isinstance(cells, CellValues):
                raise TypeError(f"{name}的块结果必须是CellValues")
            parts = (cells.keys, cells.values) + (
                () if cells.nulls is None else (cells.nulls,)
            )
            if not all(isinstance(part, np.ndarray) for part in parts):
                raise TypeError(f"{name}的块结果必须由numpy数组组成，不接收pandas对象")
            keys = cells.keys
            if keys.ndim != 1 or keys.dtype.kind not in "iu" or len(keys) != len(cells.values):
                raise ValueError(f"{name}的格键须为一维整数数组且与值等长")
            if cells.nulls is not None and len(cells.nulls) != len(keys):
                raise ValueError(f"{name}的nulls长度与格键不一致")
            if len(keys):
                if int(keys.min()) < 0 or int(keys.max()) >= block.cell_count:
                    raise ValueError(f"{name}的格键超出块范围")
                if int(np.bincount(keys, minlength=block.cell_count).max()) > 1:
                    raise ValueError(f"{name}的格键重复")
        return result

    def _stock_tables(
        self,
        inputs: Mapping[str, pd.DataFrame],
        symbols: list[str],
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], list[str]]:
        """整天只做股票编码与股票顺序检查并给出每只股票的行边界，其余工作留给块线程。"""

        universe = pd.Index(symbols)
        tables: dict[str, dict[str, np.ndarray]] = {}
        bounds: dict[str, np.ndarray] = {}
        reordered: list[str] = []
        for dataset in self.inputs:
            frame = inputs[dataset]
            column = frame["symbol"]
            if isinstance(column.dtype, pd.CategoricalDtype):
                codes = column.cat.codes.to_numpy()
                if len(codes) and int(codes.min()) < 0:
                    raise ValueError(f"{dataset}存在空symbol")
                mapping = universe.get_indexer(column.cat.categories).astype(np.int32)
                stock = mapping[codes]
            else:
                stock = universe.get_indexer(column).astype(np.int32)
            if len(stock) and int(stock.min()) < 0:
                raise ValueError(f"{dataset}存在不在股票全集内的symbol")
            table = {"stock": stock}
            for name in self.input_columns[dataset]:
                if name != "symbol":
                    table[name] = _numeric_column(frame[name], dataset, name)
            if len(stock) > 1 and (stock[1:] < stock[:-1]).any():
                # 上游应已按symbol排序；不成立时整表按股票稳定重排一次并记入统计。
                order = np.argsort(stock, kind="stable")
                table = {name: values[order] for name, values in table.items()}
                reordered.append(dataset)
            for values in table.values():
                values.flags.writeable = False
            tables[dataset] = table
            bounds[dataset] = np.searchsorted(
                table["stock"], np.arange(len(symbols) + 1), side="left"
            )
        return tables, bounds, reordered


def _forbid_pandas(cls: type) -> None:
    """分块指标组所在模块不得导入pandas，compute_block源码也不得引用pandas。"""

    module = sys.modules.get(cls.__module__)
    for name, value in (vars(module).items() if module is not None else ()):
        if isinstance(value, types.ModuleType):
            origin = value.__name__
        else:
            origin = getattr(value, "__module__", None)
        if isinstance(origin, str) and (origin == "pandas" or origin.startswith("pandas.")):
            raise TypeError(
                f"分块指标组{cls.__name__}所在模块{cls.__module__}导入了pandas（{name}）；"
                "分块计算只能使用numpy与trans_to_min.kernels，pandas只允许用于独立参考等其他模块"
            )
    method = cls.__dict__.get("compute_block")
    if method is not None:
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError):
            return
        if _PANDAS_SOURCE.search(source):
            raise TypeError(f"{cls.__name__}.compute_block不得使用pandas")


def _numeric_column(series: pd.Series, dataset: str, name: str) -> np.ndarray:
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
        # 取视图再设只读，不改动执行器输入表本身的数组标志。
        return np.asarray(series.to_numpy(copy=False)).view()
    raise TypeError(
        f"分块指标组只接收数值字段：{dataset}.{name}为{dtype}；"
        "可空整数等请在适配器层确认物理类型"
    )


def _block_options(config: RunConfig | None) -> tuple[int, int]:
    if config is None:
        return DEFAULT_BLOCK_ROW_BUDGET, 1
    threads = config.block_threads or config.arrow_threads_per_worker
    return config.block_row_budget, max(1, min(threads, os.cpu_count() or 1))


def _block_spans(
    bounds: dict[str, np.ndarray], stock_count: int, budget: int
) -> list[tuple[int, int]]:
    """按行预算切成连续股票区间；一只股票总在同一块，超预算的单只股票独占一块。"""

    per_stock = sum(np.diff(edges) for edges in bounds.values())
    cumulative = np.concatenate(([0], np.cumsum(per_stock)))
    spans = []
    start = 0
    while start < stock_count:
        stop = int(np.searchsorted(cumulative, cumulative[start] + budget, side="right")) - 1
        spans.append((start, min(max(stop, start + 1), stock_count)))
        start = spans[-1][1]
    return spans


def _assemble(
    metric: MinuteMetric,
    trade_date: date,
    symbols: list[str],
    placed: list[tuple[int, dict[str, CellValues]]],
) -> pd.DataFrame:
    """把各块稀疏结果合成标准单指标日长表（分钟为外层、股票为内层）。"""

    total = len(symbols) * MINUTE_COUNT
    numpy_type = np.dtype(metric.value_type)
    fill = metric.empty_value
    nulls = np.zeros(total, dtype=bool) if fill is not None else np.ones(total, dtype=bool)
    if fill is None:
        fill = np.nan if numpy_type.kind == "f" else 0
    values = np.full(total, fill, dtype=numpy_type)
    for offset, result in placed:
        cells = result[metric.name]
        keys = cells.keys.astype(np.int64) + offset * MINUTE_COUNT
        if numpy_type.kind in "iu" and cells.values.dtype.kind == "f":
            raise TypeError(f"{metric.name}为{metric.value_type}，块结果不能是浮点")
        values[keys] = cells.values
        nulls[keys] = False
        if cells.nulls is not None:
            nulls[keys[cells.nulls.astype(bool)]] = True
    # 块内为股票主序，标准长表为分钟主序。
    values = values.reshape(len(symbols), MINUTE_COUNT).T.ravel()
    nulls = nulls.reshape(len(symbols), MINUTE_COUNT).T.ravel()
    if numpy_type.kind == "f":
        values[nulls] = np.nan
        column = values
    elif metric.value_type in PANDAS_NULLABLE_INTEGER_TYPES and (
        metric.empty_value is None or nulls.any()
    ):
        column = pd.arrays.IntegerArray(values, nulls)
    else:
        column = values
    labels = minute_labels(trade_date)
    result = pd.DataFrame(
        {
            "minute": labels.repeat(len(symbols)),
            "symbol": np.tile(np.asarray(symbols, dtype=object), MINUTE_COUNT),
            metric.name: column,
        }
    )
    result.insert(0, "trade_date", pd.Timestamp(trade_date).date())
    return result
