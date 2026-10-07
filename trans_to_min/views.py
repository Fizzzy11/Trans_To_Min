"""显式订单视图与金额校验；原始输入及成交事件不做全局去重。"""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd
from .minute import assign_minute_codes


def validate_amounts(values: pd.Series, *, allow_missing: bool = True) -> pd.Series:
    """仅允许约定的缺失额；非空负数、Inf和非法值仍失败。"""
    result = pd.to_numeric(values, errors="raise").astype("float64")
    known = result.notna()
    if not allow_missing and not bool(known.all()):
        raise ValueError("真实成交金额不允许为空")
    if bool((known & (~np.isfinite(result) | result.lt(0))).any()):
        raise ValueError("金额必须为非负有限数，不能静默过滤异常")
    return result


def _identity(frame: pd.DataFrame, day: date, *, trade: bool = False) -> None:
    timing = "time_ms" if trade or "time_ms" in frame else "timestamp"
    needed = {"symbol", "channel", timing}
    needed |= {"buy_order_num", "sell_order_num"} if trade else {"side", "order_num"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"完整订单视图缺少字段：{missing}")
    if frame[list(needed)].isna().any().any():
        raise ValueError("订单身份和时间不允许为空")
    if "trade_date" in frame and not pd.to_datetime(frame["trade_date"]).dt.date.eq(day).all():
        raise ValueError("订单视图必须在一个完整交易日内构建")
    if timing == "timestamp" and not pd.to_datetime(frame[timing]).dt.date.eq(day).all():
        raise ValueError("订单时间不属于指定交易日")
    for name in needed - {"symbol", "timestamp"}:
        values = pd.to_numeric(frame[name], errors="raise")
        if not (np.isfinite(values) & values.ge(0) & values.eq(values.round())).all():
            raise ValueError(f"{name}必须为非负有限整数")
    if timing == "time_ms" and not frame["time_ms"].lt(86_400_000).all():
        raise ValueError("time_ms超出单日范围")


def aggregate_order_fragments(frame: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """完整日委托片段按股票、频道、方向、订单聚合，先锚定再处理缺额。

    数量和已知额可选，按实际申请列聚合；不生成虚构的订单平均报价。
    同锚点按原始首次出现顺序稳定排列，不修改共享原表。
    """
    _identity(frame, trade_date)
    keys = ["symbol", "channel", "side", "order_num"]
    timing = "time_ms" if "time_ms" in frame else "timestamp"
    columns = keys + [timing] + [x for x in ("volume", "money") if x in frame]
    source = frame.loc[:, columns].copy()
    source["_position"] = np.arange(len(source), dtype=np.int64)
    if "volume" in source:
        volume = pd.to_numeric(source["volume"], errors="raise")
        if not (volume.notna() & np.isfinite(volume) & volume.ge(0) & volume.eq(volume.round())).all():
            raise ValueError("订单数量必须为非负有限整数")
        source["volume"] = volume
    if "money" in source:
        source["money"] = validate_amounts(source["money"])
    groups = source.groupby(keys, observed=True, sort=False, dropna=False)
    result = groups.agg(**{timing: (timing, "min"), "_position": ("_position", "min")})
    if "volume" in source:
        result["volume"] = groups["volume"].sum()
    if "money" in source:
        result["money"] = groups["money"].sum(min_count=1)
        if np.isinf(result["money"]).any():
            raise ValueError("订单金额求和溢出")
    result = result.reset_index().sort_values(
        [timing, "_position"], kind="stable").drop(columns="_position").reset_index(drop=True)
    result.attrs["price_quality"] = {
        "input_fragments": len(frame), "orders": len(result),
        "unknown_money_fragments": int(source["money"].isna().sum()) if "money" in source else None,
        "unknown_money_orders": int(result["money"].isna().sum()) if "money" in result else None,
        "unknown_money_volume": int(source.loc[source["money"].isna(), "volume"].sum()) if "money" in source and "volume" in source else None,
        "mixed_money_orders": int(((groups["money"].count() > 0) & (groups["money"].count() < groups.size())).sum()) if "money" in source else None,
    }
    return result


def trade_order_participation(
    frame: pd.DataFrame, trade_date: date, *, order_side: str, time_scope: str
) -> pd.DataFrame:
    """独立统计买/卖/双侧参与订单，不改变成交原时间、主动方向或量额。

    per_minute是每分钟参与过的订单；first_event是全日首次参与归属。
    输出仅订单身份与锚点，不携带双腿量额，避免冒充单边成交量。
    主动方向筛选由调用方在事件分支显式完成；订单侧别另列order_side。
    """
    _identity(frame, trade_date, trade=True)
    if order_side not in {"buy", "sell", "both"} or time_scope not in {"per_minute", "first_event"}:
        raise ValueError("必须明确买/卖/双侧和每分钟参与/全日首次口径")
    legs = []
    for column, direction, choice in (("buy_order_num", 49, "buy"), ("sell_order_num", 50, "sell")):
        if order_side in {choice, "both"}:
            leg = frame[["symbol", "channel", "time_ms", column]].rename(columns={column: "order_num"}).copy()
            leg["order_side"] = direction
            legs.append(leg)
    source = pd.concat(legs, ignore_index=True)
    # 零订单号不是可计数的订单身份；真实成交仍保留在原事件表。
    source = source.loc[source["order_num"].gt(0)].copy()
    keys = ["symbol", "channel", "order_side", "order_num"]
    if time_scope == "per_minute":
        source["minute_code"] = assign_minute_codes(source["time_ms"]).to_numpy()
        source = source.loc[source["minute_code"].ge(0)]
        keys.append("minute_code")
    return source.groupby(keys, sort=False, observed=True)["time_ms"].min().reset_index()
