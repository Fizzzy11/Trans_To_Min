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
    include_post_market: bool = False,
) -> pd.Series:
    """按框架固定的左开右闭规则为逐笔事件分配分钟结束标签。

    常规区间采用 ``(上一分钟, 当前分钟]``。09:30、11:30、13:00
    保留特殊边界；15:00默认只包含(14:59, 15:00]，其余午休及盘后记录返回NaT。
    显式开启include_post_market才把盘后事件折入15:00。
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
    include_post_market: bool = False,
) -> pd.Series:
    """按源数据毫秒精度转换时间戳，默认排除15:00之后的盘后事件。"""

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
    include_post_market: bool = False,
) -> pd.Series:
    """根据午夜以来毫秒数计算0至241的分钟编码。

    编码与 :func:`minute_labels` 一一对应；不属于交易分钟轴的记录使用 ``-1``。
    该函数只进行整数数组运算，是大规模逐笔数据的推荐切片路径。
    默认15:00结束标签只接收(14:59, 15:00]；显式开启盘后折入可兼容旧映射。
    """

    numeric = pd.to_numeric(time_ms, errors="raise")
    if numeric.isna().any():
        raise ValueError("time_ms不允许为空")
    codes = minute_codes(
        numeric.to_numpy(dtype=np.int64, copy=False),
        include_pre_market=include_pre_market,
        include_post_market=include_post_market,
    )
    return pd.Series(codes, index=time_ms.index, name="minute_code", copy=False)


def _ceiling_minute_table(include_pre_market: bool, include_post_market: bool) -> np.ndarray:
    """按"向上取整到分钟"的结束分钟（午夜起分钟数）查分钟编码；两个整点特例另行处理。"""

    table = np.full(1441, -1, dtype=np.int16)
    open_minute = T0930 // MILLISECONDS_PER_MINUTE
    if include_pre_market:
        table[: open_minute + 1] = 0
    else:
        table[open_minute] = 0
    table[open_minute + 1 : open_minute + 120] = np.arange(1, 120, dtype=np.int16)
    table[T1129 // MILLISECONDS_PER_MINUTE + 1 : T1131 // MILLISECONDS_PER_MINUTE + 1] = 120
    table[T1259 // MILLISECONDS_PER_MINUTE : T1300 // MILLISECONDS_PER_MINUTE + 1] = 121
    afternoon = T1300 // MILLISECONDS_PER_MINUTE
    table[afternoon + 1 : afternoon + 121] = np.arange(122, 242, dtype=np.int16)
    if include_post_market:
        table[afternoon + 121 :] = 241
    return table


_CEILING_TABLES = {
    (pre, post): _ceiling_minute_table(pre, post)
    for pre in (False, True)
    for post in (False, True)
}


def minute_codes(
    values: np.ndarray,
    *,
    include_pre_market: bool = True,
    include_post_market: bool = False,
) -> np.ndarray:
    """numpy版分钟编码：输入午夜以来毫秒数的整数数组，返回int16编码，轴外为-1。

    规则与 :func:`assign_minute_codes` 完全相同，但只做"向上取整到分钟再查表"，
    适合在分块线程内对大数组调用。
    """

    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"time_ms必须是整数数组，实际为{array.dtype}")
    if len(array) and (int(array.min()) < 0 or int(array.max()) >= MILLISECONDS_PER_DAY):
        invalid = (array < 0) | (array >= MILLISECONDS_PER_DAY)
        raise ValueError(f"time_ms超出单日毫秒范围：{array[invalid][:5].tolist()}")
    ceiling = (array.astype(np.int64) + (MILLISECONDS_PER_MINUTE - 1)) // MILLISECONDS_PER_MINUTE
    codes = _CEILING_TABLES[(bool(include_pre_market), bool(include_post_market))][ceiling]
    # 12:59整点归121，而(12:58, 12:59)不属于分钟轴；两者向上取整相同，需按原值区分。
    noon = ceiling == T1259 // MILLISECONDS_PER_MINUTE
    if noon.any():
        codes[noon & (array != T1259)] = -1
    if not include_pre_market:
        opening = ceiling == T0930 // MILLISECONDS_PER_MINUTE
        if opening.any():
            codes[opening & (array != T0930)] = -1
    return codes


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
