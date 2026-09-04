from __future__ import annotations

from datetime import date

import pandas as pd

from ..metric import MinuteMetric


class MinuteTradeVolume(MinuteMetric):
    """逐笔成交量按左开右闭分钟区间求和的参考指标。"""

    name = "trade_volume_min"
    inputs = ("chengjiao",)
    input_columns = {
        "chengjiao": ("symbol", "time_ms", "volume"),
    }
    value_type = "uint64"
    empty_value = 0

    def prepare_events(
        self,
        trade_date: date,
        inputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """成交量不需要跨表匹配，直接保留成交事件。"""

        return inputs["chengjiao"].loc[:, ["symbol", "time_ms", "volume"]].copy()

    def aggregate_minutes(self, events: pd.DataFrame) -> pd.DataFrame:
        """显性方程：同一股票同一分钟的逐笔成交量求和。"""

        return (
            events.groupby(["symbol", "minute"], sort=False, observed=True)["volume"]
            .sum()
            .astype("uint64")
            .rename(self.name)
            .reset_index()
        )

    def process_minute_data(self, minute_data: pd.DataFrame) -> int:
        """通用分组模式下的等价单分钟方程。"""

        return int(minute_data["volume"].sum())
