from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import MINUTE_COUNT, TIMEZONE


MILLISECONDS_PER_MINUTE = 60_000
MILLISECONDS_PER_DAY = 86_400_000
T0930 = 9 * 3_600_000 + 30 * MILLISECONDS_PER_MINUTE
T1129 = 11 * 3_600_000 + 29 * MILLISECONDS_PER_MINUTE
T1131 = 11 * 3_600_000 + 31 * MILLISECONDS_PER_MINUTE
T1259 = 12 * 3_600_000 + 59 * MILLISECONDS_PER_MINUTE
T1300 = 13 * 3_600_000
T1459 = 14 * 3_600_000 + 59 * MILLISECONDS_PER_MINUTE
T1500 = 15 * 3_600_000


def minute_labels(trade_date: date | str | pd.Timestamp) -> pd.DatetimeIndex:
    """构造框架固定的242个分钟结束标签。"""

    value = pd.Timestamp(trade_date).date().isoformat()
    morning = pd.date_range(
        f"{value} 09:30:00",
        f"{value} 11:30:00",
        freq="1min",
        tz=TIMEZONE,
    )
    afternoon = pd.date_range(
        f"{value} 13:00:00",
        f"{value} 15:00:00",
        freq="1min",
        tz=TIMEZONE,
    )
    result = morning.union(afternoon)
    if len(result) != MINUTE_COUNT:
        raise RuntimeError(f"分钟标签数量异常：{len(result)}")
    return result


def assign_minute_labels(
    timestamps: pd.Series,
    trade_date: date | str | pd.Timestamp,
    *,
    include_pre_market: bool = True,
    include_post_market: bool = True,
) -> pd.Series:
    """按框架固定的左开右闭规则为逐笔事件分配分钟结束标签。

    常规区间采用 ``(上一分钟, 当前分钟]``。09:30、11:30、13:00和
    15:00使用本框架定义的特殊边界；未落入任何区间的午休记录返回NaT。
    """

    codes = assign_timestamp_minute_codes(
        timestamps,
        include_pre_market=include_pre_market,
        include_post_market=include_post_market,
    )
    result = minute_labels_from_codes(codes, trade_date)
    return result


def assign_timestamp_minute_codes(
    timestamps: pd.Series,
    *,
    include_pre_market: bool = True,
    include_post_market: bool = True,
) -> pd.Series:
    """把完整时间戳转换为紧凑分钟编码。"""

    values = pd.to_datetime(timestamps, errors="raise")
    if values.dt.tz is None:
        values = values.dt.tz_localize(TIMEZONE)
    else:
        values = values.dt.tz_convert(TIMEZONE)
    time_ms = (
        values.dt.hour.astype("int64") * 3_600_000
        + values.dt.minute.astype("int64") * MILLISECONDS_PER_MINUTE
        + values.dt.second.astype("int64") * 1_000
        + values.dt.microsecond.astype("int64") // 1_000
    )
    return assign_minute_codes(
        time_ms,
        include_pre_market=include_pre_market,
        include_post_market=include_post_market,
    )


def assign_minute_codes(
    time_ms: pd.Series,
    *,
    include_pre_market: bool = True,
    include_post_market: bool = True,
) -> pd.Series:
    """根据午夜以来毫秒数计算0至241的分钟编码。

    编码与 :func:`minute_labels` 一一对应；不属于交易分钟轴的记录使用 ``-1``。
    该函数只进行整数数组运算，是大规模逐笔数据的推荐切片路径。
    """

    numeric = pd.to_numeric(time_ms, errors="raise")
    if numeric.isna().any():
        raise ValueError("time_ms不允许为空")
    values = numeric.to_numpy(dtype=np.int64, copy=False)
    invalid_range = (values < 0) | (values >= MILLISECONDS_PER_DAY)
    if bool(invalid_range.any()):
        examples = values[invalid_range][:5].tolist()
        raise ValueError(f"time_ms超出单日毫秒范围：{examples}")

    codes = np.full(len(values), -1, dtype=np.int16)
    if include_pre_market:
        codes[values <= T0930] = 0
    else:
        codes[values == T0930] = 0

    regular_morning = (values > T0930) & (values <= T1129)
    codes[regular_morning] = (
        (values[regular_morning] - T0930 + MILLISECONDS_PER_MINUTE - 1)
        // MILLISECONDS_PER_MINUTE
    ).astype(np.int16)
    codes[(values > T1129) & (values <= T1131)] = 120
    codes[(values >= T1259) & (values <= T1300)] = 121

    regular_afternoon = (values > T1300) & (values <= T1459)
    codes[regular_afternoon] = (
        121
        + (
            values[regular_afternoon]
            - T1300
            + MILLISECONDS_PER_MINUTE
            - 1
        )
        // MILLISECONDS_PER_MINUTE
    ).astype(np.int16)
    if include_post_market:
        codes[values > T1459] = 241
    else:
        codes[(values > T1459) & (values <= T1500)] = 241
    return pd.Series(codes, index=time_ms.index, name="minute_code", copy=False)


def minute_labels_from_codes(
    codes: pd.Series,
    trade_date: date | str | pd.Timestamp,
) -> pd.Series:
    """把分钟编码物化为带上海时区的分钟结束标签。"""

    numeric = pd.to_numeric(codes, errors="raise").to_numpy(dtype=np.int16, copy=False)
    valid = (numeric >= 0) & (numeric < MINUTE_COUNT)
    result = pd.Series(
        pd.NaT,
        index=codes.index,
        dtype=f"datetime64[ns, {TIMEZONE}]",
        name="minute",
    )
    if bool(valid.any()):
        result.iloc[np.flatnonzero(valid)] = minute_labels(trade_date).take(numeric[valid])
    return result
