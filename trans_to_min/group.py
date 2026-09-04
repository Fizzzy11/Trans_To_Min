from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date

import pandas as pd

from .metric import MinuteMetric


GROUP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class MinuteMetricGroup:
    """共享同一组逐笔物理输入的多个分钟指标。

    组内指标必须声明完全相同的逐笔输入表集合。框架会对各指标所需字段取并集，
    每个交易日只读取一次每张物理表。默认实现仍逐指标调用原有``compute_day``，
    业务项目可以覆写``iter_compute_day``以共享关联、分钟标记或聚合中间量。

    计算可以合并，但每个指标仍由组执行器独立写入自己的目录、Parquet字段和
    manifest，不会生成多指标生产宽表。
    """

    def __init__(self, name: str, metrics: Sequence[MinuteMetric]) -> None:
        if not isinstance(name, str) or GROUP_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                "指标组名称只能由小写英文字母、数字和单下划线分段组成，"
                f"且必须以字母开头：{name!r}"
            )
        values = tuple(metrics)
        if not values:
            raise ValueError("指标组至少需要一个分钟指标")
        if any(not isinstance(metric, MinuteMetric) for metric in values):
            raise TypeError("指标组成员必须是MinuteMetric实例")

        names = [metric.name for metric in values]
        duplicates = sorted({value for value in names if names.count(value) > 1})
        if duplicates:
            raise ValueError(f"指标组存在重复指标名：{duplicates}")

        inputs = tuple(values[0].inputs)
        input_set = set(inputs)
        incompatible = {
            metric.name: tuple(metric.inputs)
            for metric in values[1:]
            if set(metric.inputs) != input_set
        }
        if incompatible:
            raise ValueError(
                "指标组成员必须使用相同的逐笔输入表集合；"
                f"基准={inputs}，不一致={incompatible}"
            )

        self.name = name
        self.metrics = values
        self.inputs = inputs
        self._metrics_by_name = {metric.name: metric for metric in values}
        self.input_columns = self._merge_input_columns(values, inputs)

    @property
    def metric_names(self) -> tuple[str, ...]:
        """按声明顺序返回组内全部输出指标名。"""

        return tuple(metric.name for metric in self.metrics)

    def metric(self, name: str) -> MinuteMetric:
        """返回一个组内指标。"""

        try:
            return self._metrics_by_name[name]
        except KeyError as exc:
            raise KeyError(f"指标组{self.name!r}不包含指标{name!r}") from exc

    def select_metrics(self, names: Iterable[str] | None = None) -> tuple[MinuteMetric, ...]:
        """校验并按组声明顺序选择待计算指标。"""

        if names is None:
            return self.metrics
        requested = tuple(dict.fromkeys(names))
        unknown = sorted(set(requested) - set(self._metrics_by_name))
        if unknown:
            raise ValueError(f"指标组{self.name!r}包含未知选择：{unknown}")
        requested_set = set(requested)
        return tuple(
            metric for metric in self.metrics if metric.name in requested_set
        )

    def iter_compute_day(
        self,
        trade_date: date,
        inputs: Mapping[str, pd.DataFrame],
        metric_names: Iterable[str] | None = None,
    ) -> Iterator[tuple[MinuteMetric, pd.DataFrame]]:
        """逐项产生标准单指标日结果，允许业务组覆写并共享中间计算。

        覆写实现必须只产生``metric_names``选择的指标，且每个指标恰好产生一次。
        日结果仍需符合``trade_date, minute, symbol, metric_name``标准长表结构。
        """

        missing = sorted(set(self.inputs) - set(inputs))
        if missing:
            raise ValueError(f"指标组缺少逐笔输入：{missing}")
        for metric in self.select_metrics(metric_names):
            metric_inputs = {dataset: inputs[dataset] for dataset in metric.inputs}
            yield metric, metric.compute_day(trade_date, metric_inputs)

    @staticmethod
    def _merge_input_columns(
        metrics: Sequence[MinuteMetric],
        inputs: Sequence[str],
    ) -> dict[str, tuple[str, ...]]:
        """按指标声明顺序构造每张逐笔表的逻辑字段并集。"""

        result: dict[str, tuple[str, ...]] = {}
        for dataset in inputs:
            columns: list[str] = []
            for metric in metrics:
                for column in metric.input_columns[dataset]:
                    if column not in columns:
                        columns.append(column)
            result[dataset] = tuple(columns)
        return result
