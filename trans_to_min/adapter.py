from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATASETS, TIMEZONE


DERIVED_DEPENDENCIES = {
    "trade_date": {"date"},
    "time_ms": {"time"},
    "timestamp": {"date", "time"},
}


def input_path(root: Path, dataset: str, trade_date: int | date | str) -> Path:
    """返回三类标准逐笔数据的单日文件路径。"""

    if dataset not in DATASETS:
        raise ValueError(f"未知逐笔数据集：{dataset}")
    value = _date_text(trade_date)
    return root / dataset / value[:4] / value[:6] / f"{value}_{dataset}.parquet"


class L2InputAdapter:
    """把当前压缩Parquet字段适配为分钟框架使用的逻辑字段。"""

    def __init__(self, input_root: Path) -> None:
        self.input_root = Path(input_root)

    def read(
        self,
        dataset: str,
        trade_date: int | date | str,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """按列读取并归一化一个交易日的逐笔数据。"""

        path = input_path(self.input_root, dataset, trade_date)
        if not path.is_file():
            raise FileNotFoundError(f"逐笔文件不存在：{path}")
        requested = tuple(dict.fromkeys(columns))
        if not requested:
            raise ValueError("columns 不能为空")

        parquet = pq.ParquetFile(path)
        physical = self._physical_columns(dataset, requested, parquet.schema_arrow)
        metadata = parquet.schema_arrow.metadata or {}
        table = parquet.read(columns=physical, use_threads=True)
        # 输入表不再使用，允许Arrow在转成Pandas时立即释放自身缓冲区。
        categories = ["symbol"] if "symbol" in physical else None
        frame = table.to_pandas(
            categories=categories,
            split_blocks=True,
            self_destruct=True,
        )
        normalized = self._normalize(dataset, frame, requested, metadata)
        return normalized.reindex(columns=requested)

    @staticmethod
    def _physical_columns(
        dataset: str,
        requested: Sequence[str],
        schema: pa.Schema,
    ) -> list[str]:
        physical: set[str] = set()
        schema_names = set(schema.names)
        for name in requested:
            if name in DERIVED_DEPENDENCIES:
                physical.update(DERIVED_DEPENDENCIES[name])
            elif name == "money" and "money" not in schema_names:
                physical.update(("price", "volume"))
            elif name in schema_names:
                physical.add(name)
            else:
                raise ValueError(
                    f"{dataset} 不支持逻辑字段 {name!r}，实际字段为 {schema.names}"
                )
        missing = sorted(physical - schema_names)
        if missing:
            raise ValueError(f"{dataset} 缺少适配所需物理字段：{missing}")
        return [name for name in schema.names if name in physical]

    @staticmethod
    def _normalize(
        dataset: str,
        frame: pd.DataFrame,
        requested: Sequence[str],
        metadata: dict[bytes, bytes],
    ) -> pd.DataFrame:
        # frame只在本次读取中使用，直接原位归一化可避免复制整张单日逐笔表。
        result = frame
        if "symbol" in result:
            non_null_symbols = result["symbol"].dropna()
            if not non_null_symbols.empty and isinstance(
                non_null_symbols.iloc[0], (bytes, bytearray)
            ):
                result["symbol"] = result["symbol"].map(_decode_text)
            if not isinstance(result["symbol"].dtype, pd.CategoricalDtype):
                result["symbol"] = result["symbol"].astype("category")

        needs_time = any(name in requested for name in ("time_ms", "timestamp"))
        if needs_time:
            result["time_ms"] = _milliseconds_since_midnight(result["time"])
        if "trade_date" in requested or "timestamp" in requested:
            result["trade_date"] = pd.to_datetime(result["date"]).dt.normalize()
        if "timestamp" in requested:
            timestamp = result["trade_date"] + pd.to_timedelta(result["time_ms"], unit="ms")
            result["timestamp"] = timestamp.dt.tz_localize(TIMEZONE)

        if "price" in requested or ("money" in requested and "money" not in result):
            result["price"] = _normalize_scaled_number(result["price"], metadata, b"price_scale")
        if "money" in requested:
            if "money" in result:
                result["money"] = _normalize_scaled_number(
                    result["money"], metadata, b"money_scale"
                )
            else:
                result["money"] = (
                    result["price"].astype("float64")
                    * pd.to_numeric(result["volume"], errors="raise").astype("float64")
                )
        if "type" in requested and "type" in result:
            result["type"] = _normalize_type(result["type"], metadata)

        if "volume" in result:
            result["volume"] = pd.to_numeric(result["volume"], errors="raise").astype("uint64")
        for name in ("order_num", "buy_order_num", "sell_order_num"):
            if name in result:
                result[name] = pd.to_numeric(result[name], errors="raise").astype("uint64")
        return result


def discover_dataset_dates(root: Path, dataset: str) -> set[int]:
    """发现一个逐笔数据集已有的全部交易日。"""

    if dataset not in DATASETS:
        raise ValueError(f"未知逐笔数据集：{dataset}")
    result: set[int] = set()
    for path in (Path(root) / dataset).glob("*/*/*.parquet"):
        prefix = path.name[:8]
        if prefix.isdigit() and path.name == f"{prefix}_{dataset}.parquet":
            result.add(int(prefix))
    return result


def validate_matching_date_sets(
    root: Path,
    datasets: Iterable[str],
    start_date: int | date | str | None = None,
    end_date: int | date | str | None = None,
) -> list[int]:
    """返回目标区间日期，并报告区间内各逐笔输入的缺失日期。"""

    names = tuple(dict.fromkeys(datasets))
    if not names:
        raise ValueError("指标至少需要一个逐笔输入")
    lower = int(_date_text(start_date)) if start_date is not None else None
    upper = int(_date_text(end_date)) if end_date is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"start_date不能晚于end_date：{lower} > {upper}")

    date_sets = {name: discover_dataset_dates(root, name) for name in names}
    for name, values in date_sets.items():
        date_sets[name] = {
            value
            for value in values
            if (lower is None or value >= lower) and (upper is None or value <= upper)
        }
    candidates = set().union(*date_sets.values())
    if not candidates:
        bounds = {
            "start_date": lower,
            "end_date": upper,
        }
        raise RuntimeError(f"目标区间内没有逐笔文件：{bounds}")

    missing = {
        name: sorted(candidates - values)
        for name, values in date_sets.items()
        if candidates - values
    }
    if missing:
        examples = {
            name: [str(value) for value in values[:20]]
            for name, values in missing.items()
        }
        raise RuntimeError(
            "目标区间内逐笔输入日期不完整，"
            f"缺失日期示例：{examples}"
        )
    return sorted(candidates)


def _normalize_scaled_number(
    values: pd.Series,
    metadata: dict[bytes, bytes],
    scale_key: bytes,
) -> pd.Series:
    if not pd.api.types.is_integer_dtype(values.dtype):
        return pd.to_numeric(values, errors="raise").astype("float64")
    raw_scale = metadata.get(scale_key)
    if raw_scale is None:
        raise ValueError(f"整数编码字段缺少Parquet metadata：{scale_key.decode()}")
    try:
        scale = int(raw_scale.decode())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"非法缩放比例 {scale_key!r}={raw_scale!r}") from exc
    if scale <= 0:
        raise ValueError(f"缩放比例必须为正整数：{scale_key.decode()}={scale}")
    return pd.to_numeric(values, errors="raise").astype("float64") / scale


def _normalize_type(values: pd.Series, metadata: dict[bytes, bytes]) -> pd.Series:
    if not pd.api.types.is_integer_dtype(values.dtype):
        return values.map(_decode_text)
    raw = metadata.get(b"type_encoding")
    if raw is None:
        raise ValueError("整数type字段缺少Parquet metadata：type_encoding")
    try:
        pairs = [item.split("=", 1) for item in raw.decode().split(",") if item]
        mapping = {int(code): logical for code, logical in pairs}
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"非法type_encoding：{raw!r}") from exc
    normalized = pd.to_numeric(values, errors="raise").map(mapping)
    if normalized.isna().any():
        unknown = sorted(pd.to_numeric(values[normalized.isna()]).unique().tolist())
        raise ValueError(f"type包含metadata未声明的代码：{unknown}")
    return normalized


def _milliseconds_since_midnight(values: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(values.dtype):
        result = pd.to_numeric(values, errors="raise")
    elif pd.api.types.is_timedelta64_dtype(values.dtype):
        result = (values.dt.total_seconds() * 1_000).round().astype("int64")
    else:
        parsed = pd.to_datetime(values, errors="raise")
        result = (
            parsed.dt.hour.astype("int64") * 3_600_000
            + parsed.dt.minute.astype("int64") * 60_000
            + parsed.dt.second.astype("int64") * 1_000
            + parsed.dt.microsecond.astype("int64") // 1_000
        )
    if ((result < 0) | (result >= 86_400_000)).any():
        examples = result[(result < 0) | (result >= 86_400_000)].head().tolist()
        raise ValueError(f"time超出单日毫秒范围：{examples}")
    return result.astype("uint32", copy=False)


def _decode_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="strict").rstrip("\x00")
    return str(value)


def _date_text(value: int | date | str) -> str:
    if isinstance(value, int):
        text = str(value)
    else:
        text = pd.Timestamp(value).strftime("%Y%m%d")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"交易日格式错误：{value!r}")
    return text
