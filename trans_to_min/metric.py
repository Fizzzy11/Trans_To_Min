from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np
import pandas as pd

from .config import DATASETS, MINUTE_COUNT, validate_metric_name
from .minute import (
    assign_minute_codes,
    assign_timestamp_minute_codes,
    minute_labels,
)


VALUE_TYPES = {"float64", "int64", "uint64"}
PANDAS_NULLABLE_INTEGER_TYPES = {
    "int64": "Int64",
    "uint64": "UInt64",
}


class MinuteMetric(ABC):
    """单个自定义分钟数据的基类。

    子类显式声明逐笔输入和所需字段，并实现逐笔整理与分钟聚合方程。框架负责
    股票并集、左开右闭分钟标签、空分钟填充、结果校验和存储。
    """

    name = ""
    inputs: tuple[str, ...] = ()
    input_columns: dict[str, tuple[str, ...]] = {}
    value_type = "float64"
    empty_value: float | int | None = None
    include_pre_market = True
    include_post_market = True

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        try:
            validate_metric_name(cls.name)
        except ValueError as exc:
            raise TypeError(f"{cls.__name__}.name非法：{exc}") from exc
        if not cls.inputs:
            raise TypeError(f"{cls.__name__} 必须声明inputs")
        unknown = sorted(set(cls.inputs) - set(DATASETS))
        if unknown:
            raise TypeError(f"{cls.__name__} 声明了未知逐笔输入：{unknown}")
        if set(cls.input_columns) != set(cls.inputs):
            raise TypeError(
                f"{cls.__name__}.input_columns必须与inputs逐项对应"
            )
        for dataset, columns in cls.input_columns.items():
            if "symbol" not in columns or not {"timestamp", "time_ms"}.intersection(columns):
                raise TypeError(
                    f"{cls.__name__}.{dataset}至少需要symbol，以及timestamp或time_ms"
                )
        if cls.value_type not in VALUE_TYPES:
            raise TypeError(f"不支持的value_type：{cls.value_type}")

    def compute_day(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """计算一个交易日并补齐当日活跃股票的242分钟序列。"""

        missing = sorted(set(self.inputs) - set(inputs))
        if missing:
            raise ValueError(f"缺少指标输入：{missing}")
        symbols = self.symbol_union(inputs)
        if not symbols:
            return self._empty_output()

        events = self.prepare_events(trade_date, inputs)
        self._validate_events(events)
        if "time_ms" in events:
            minute_codes = assign_minute_codes(
                events["time_ms"],
                include_pre_market=self.include_pre_market,
                include_post_market=self.include_post_market,
            )
        else:
            minute_codes = assign_timestamp_minute_codes(
                events["timestamp"],
                include_pre_market=self.include_pre_market,
                include_post_market=self.include_post_market,
            )
        # 浅复制只创建新的DataFrame管理结构，逐笔业务列继续共享底层数组。
        events = events.copy(deep=False)
        events["minute"] = pd.Categorical.from_codes(
            minute_codes.to_numpy(dtype=np.int16, copy=False),
            categories=minute_labels(trade_date),
            ordered=True,
        )
        valid_minutes = minute_codes.ge(0).to_numpy(copy=False)
        if not bool(valid_minutes.all()):
            events = events.loc[valid_minutes]

        if events.empty:
            aggregated = pd.DataFrame(columns=["symbol", "minute", self.name])
        else:
            aggregated = self.aggregate_minutes(events)
        self._validate_aggregated(aggregated)
        return self._align_minutes(trade_date, symbols, aggregated)

    def symbol_union(self, inputs: dict[str, pd.DataFrame]) -> list[str]:
        """取本指标声明的所有逐笔输入中实际出现股票的并集。"""

        symbols: set[str] = set()
        for dataset in self.inputs:
            frame = inputs[dataset]
            if "symbol" not in frame:
                raise ValueError(f"{dataset} 缺少symbol字段")
            unique = frame["symbol"].dropna().unique().tolist()
            symbols.update(str(value) for value in unique)
        return sorted(symbols)

    def finalize_aggregated_day(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
        aggregated: pd.DataFrame,
    ) -> pd.DataFrame:
        """校验稀疏分钟聚合结果并补齐为标准单指标日长表。

        该公开方法供同源指标组复用共享聚合结果。普通指标仍直接使用
        ``compute_day``，无需调用本方法。
        """

        missing = sorted(set(self.inputs) - set(inputs))
        if missing:
            raise ValueError(f"缺少指标输入：{missing}")
        symbols = self.symbol_union(inputs)
        if not symbols:
            return self._empty_output()
        self._validate_aggregated(aggregated)
        return self._align_minutes(trade_date, symbols, aggregated)

    @abstractmethod
    def prepare_events(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """整理、匹配逐笔记录，返回symbol及timestamp或time_ms事件表。"""

        ...

    def aggregate_minutes(self, events: pd.DataFrame) -> pd.DataFrame:
        """按股票和分钟调用显性的单分钟方程。

        简单指标应覆写本方法并使用向量化groupby；复杂指标可以只实现
        ``process_minute_data``，沿用这里的通用分组流程。
        """

        rows = []
        for (symbol, minute), minute_data in events.groupby(
            ["symbol", "minute"], sort=False, observed=True
        ):
            rows.append(
                {
                    "symbol": str(symbol),
                    "minute": minute,
                    self.name: self.process_minute_data(minute_data),
                }
            )
        return pd.DataFrame(rows, columns=["symbol", "minute", self.name])

    def process_minute_data(self, minute_data: pd.DataFrame) -> float | int:
        """计算一个股票一个分钟的数据；复杂子类可以覆写。"""

        raise NotImplementedError(
            f"{type(self).__name__}需要实现aggregate_minutes或process_minute_data"
        )

    def _align_minutes(
        self,
        trade_date: date,
        symbols: list[str],
        aggregated: pd.DataFrame,
    ) -> pd.DataFrame:
        nullable_integer_type = PANDAS_NULLABLE_INTEGER_TYPES.get(self.value_type)
        target_type = (
            nullable_integer_type
            if self.empty_value is None and nullable_integer_type is not None
            else self.value_type
        )
        labels = minute_labels(trade_date)
        symbol_count = len(symbols)
        row_count = symbol_count * MINUTE_COUNT
        positions = np.empty(0, dtype=np.int64)
        source_values = pd.Series(dtype=target_type)
        if not aggregated.empty:
            symbol_codes = pd.Categorical(
                aggregated["symbol"], categories=symbols
            ).codes.astype(np.int64, copy=False)
            minute_values = aggregated["minute"]
            if isinstance(minute_values.dtype, pd.CategoricalDtype):
                # 业务聚合通常会保留框架创建的242分类分钟轴。
                category_positions = labels.get_indexer(minute_values.cat.categories)
                raw_codes = minute_values.cat.codes.to_numpy(dtype=np.int64, copy=False)
                minute_codes = np.full(len(raw_codes), -1, dtype=np.int64)
                valid_raw = raw_codes >= 0
                minute_codes[valid_raw] = category_positions[raw_codes[valid_raw]]
            else:
                minute_codes = labels.get_indexer(pd.DatetimeIndex(minute_values))
            invalid = (symbol_codes < 0) | (minute_codes < 0)
            if bool(invalid.any()):
                examples = aggregated.loc[invalid, ["symbol", "minute"]].head().to_dict(
                    "records"
                )
                raise ValueError(f"分钟聚合结果包含未知股票或分钟：{examples}")
            positions = minute_codes * symbol_count + symbol_codes
            source_values = aggregated[self.name]

        values = self._dense_values(row_count, positions, source_values, target_type)
        result = pd.DataFrame(
            {
                "minute": labels.repeat(symbol_count),
                "symbol": np.tile(np.asarray(symbols, dtype=object), MINUTE_COUNT),
                self.name: values,
            }
        )
        result.insert(0, "trade_date", pd.Timestamp(trade_date).date())
        if len(result) != row_count:
            raise RuntimeError(f"分钟补齐后行数异常：{len(result)} != {row_count}")
        return result

    def _dense_values(
        self,
        row_count: int,
        positions: np.ndarray,
        source: pd.Series,
        target_type: str,
    ) -> pd.Array | np.ndarray:
        """把稀疏聚合结果直接放入分钟×股票稠密数组。"""

        source_notna = source.notna().to_numpy(copy=False)
        valid_positions = positions[source_notna]
        valid_source = source.loc[source.notna()]
        if target_type in PANDAS_NULLABLE_INTEGER_TYPES.values():
            numpy_type = np.int64 if target_type == "Int64" else np.uint64
            data = np.zeros(row_count, dtype=numpy_type)
            mask = np.ones(row_count, dtype=bool)
            if len(valid_positions):
                data[valid_positions] = valid_source.to_numpy(dtype=numpy_type, copy=False)
                mask[valid_positions] = False
            return pd.arrays.IntegerArray(data, mask)

        numpy_type = np.dtype(target_type)
        fill_value = self.empty_value
        if fill_value is None:
            if not np.issubdtype(numpy_type, np.floating):
                raise TypeError(f"{target_type}空分钟必须使用可空整数类型")
            fill_value = np.nan
        data = np.full(row_count, fill_value, dtype=numpy_type)
        if len(valid_positions):
            data[valid_positions] = valid_source.to_numpy(dtype=numpy_type, copy=False)
        return data

    def _validate_events(self, events: pd.DataFrame) -> None:
        if not isinstance(events, pd.DataFrame):
            raise TypeError("prepare_events必须返回pandas.DataFrame")
        if "symbol" not in events:
            raise ValueError("事件表缺少字段：['symbol']")
        timing = "time_ms" if "time_ms" in events else "timestamp"
        if timing not in events:
            raise ValueError("事件表至少需要timestamp或time_ms字段")
        if events["symbol"].isna().any() or events[timing].isna().any():
            raise ValueError(f"事件表的symbol和{timing}不允许为空")

    def _validate_aggregated(self, frame: pd.DataFrame) -> None:
        required = {"symbol", "minute", self.name}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"分钟聚合结果缺少字段：{missing}")
        if frame.duplicated(["symbol", "minute"]).any():
            raise ValueError("分钟聚合结果存在重复的symbol + minute")

    def _empty_output(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "minute", "symbol", self.name])
