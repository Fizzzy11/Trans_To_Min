from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import RunConfig, TIMEZONE, validate_metric_name


ARROW_VALUE_TYPES = {
    "float64": pa.float64(),
    "int64": pa.int64(),
    "uint64": pa.uint64(),
}


def output_schema(metric_name: str, value_type: str) -> pa.Schema:
    """构造单个分钟数据的标准长表Schema。"""

    validate_metric_name(metric_name)
    metadata = {
        b"framework": b"trans_to_min",
        b"format_version": b"1",
        b"interval": b"left_open_right_closed",
        b"minute_label": b"interval_end",
        b"timezone": TIMEZONE.encode(),
        b"metric_name": metric_name.encode(),
    }
    return pa.schema(
        [
            pa.field("trade_date", pa.date32(), nullable=False),
            pa.field("minute", pa.timestamp("ms", tz=TIMEZONE), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field(metric_name, ARROW_VALUE_TYPES[value_type], nullable=True),
        ],
        metadata=metadata,
    )


def metric_root(output_root: Path, metric_name: str) -> Path:
    validate_metric_name(metric_name)
    root = Path(output_root).resolve(strict=False)
    destination = (root / metric_name).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"指标输出目录逃逸output_root：{metric_name!r}"
        ) from exc
    return destination


def month_path(output_root: Path, metric_name: str, month: str) -> Path:
    if len(month) != 6 or not month.isdigit():
        raise ValueError(f"月份格式错误：{month!r}")
    return metric_root(output_root, metric_name) / month[:4] / f"{month}.parquet"


def published_dates(path: Path) -> set[date]:
    """只读取日期列，返回一个已发布月份文件中的交易日集合。"""

    path = Path(path)
    if not path.is_file():
        return set()
    table = pq.read_table(path, columns=["trade_date"], use_threads=False)
    values = pc.unique(table.column("trade_date")).to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"月份文件trade_date包含空值：{path}")
    return set(values)


def write_day_frame(
    frame: pd.DataFrame,
    path: Path,
    metric_name: str,
    value_type: str,
    config: RunConfig,
) -> dict:
    """写入并回读校验单日临时分片。"""

    schema = output_schema(metric_name, value_type)
    table = pa.Table.from_pandas(
        frame.reindex(columns=schema.names),
        schema=schema,
        preserve_index=False,
        safe=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        **_parquet_compression_options(
            config.staging_compression,
            config.staging_compression_level,
        ),
        use_dictionary=["trade_date", "minute", "symbol"],
        write_statistics=True,
        row_group_size=config.row_group_size,
        version="2.6",
    )
    metadata = pq.read_metadata(path)
    if metadata.num_rows != len(frame):
        raise RuntimeError(f"单日临时分片行数不一致：{path}")
    return {
        "path": str(path),
        "rows": metadata.num_rows,
        "bytes": path.stat().st_size,
    }


def publish_month(
    day_paths: list[Path],
    destination: Path,
    metric_name: str,
    value_type: str,
    config: RunConfig,
    replaced_dates: set[date],
) -> dict:
    """按日期流式合并日分片，并原子发布月份文件。

    任一时刻最多物化一个批次，避免把新旧整月数据同时读入Pandas。最终文件仍按
    ``trade_date + minute + symbol`` 的日期主序写入。
    """

    schema = output_schema(metric_name, value_type)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    new_paths = _index_day_paths(day_paths)
    if set(new_paths) != replaced_dates:
        raise ValueError(
            "日分片日期与待发布日期不一致："
            f"day_paths={sorted(new_paths)}, replaced_dates={sorted(replaced_dates)}"
        )
    existing_dates: set[date] = set()
    written_rows = 0
    written_dates: list[date] = []
    try:
        writer = pq.ParquetWriter(
            temporary,
            schema,
            **_parquet_compression_options(
                config.compression,
                config.compression_level,
            ),
            use_dictionary=["trade_date", "minute", "symbol"],
            write_statistics=True,
            version="2.6",
        )
        try:
            def write_date_batches(value: date, batches) -> None:
                nonlocal written_rows
                date_rows = 0
                for batch in batches:
                    if batch.num_rows == 0:
                        continue
                    writer.write_batch(batch, row_group_size=config.row_group_size)
                    date_rows += batch.num_rows
                if date_rows:
                    written_dates.append(value)
                    written_rows += date_rows

            ordered_new_dates = iter(sorted(new_paths))
            next_new_date = next(ordered_new_dates, None)
            last_existing_date: date | None = None
            existing_batches = (
                _existing_month_batches(destination, schema, config.row_group_size)
                if destination.is_file()
                else iter(())
            )
            for existing_date, batch in existing_batches:
                existing_dates.add(existing_date)
                if existing_date in replaced_dates and not config.overwrite_dates:
                    raise FileExistsError(
                        "月份文件已包含待生成日期；如需覆盖请启用"
                        f"overwrite_dates：{existing_date.isoformat()}"
                    )
                if existing_date != last_existing_date:
                    if (
                        last_existing_date is not None
                        and existing_date < last_existing_date
                    ):
                        raise RuntimeError("已有月份文件没有按trade_date升序保存")
                    while (
                        next_new_date is not None
                        and next_new_date <= existing_date
                    ):
                        write_date_batches(
                            next_new_date,
                            _day_file_batches(
                                new_paths[next_new_date],
                                schema,
                                config.row_group_size,
                            ),
                        )
                        next_new_date = next(ordered_new_dates, None)
                    last_existing_date = existing_date
                if existing_date not in replaced_dates:
                    if not written_dates or written_dates[-1] != existing_date:
                        written_dates.append(existing_date)
                    writer.write_batch(batch, row_group_size=config.row_group_size)
                    written_rows += batch.num_rows

            while next_new_date is not None:
                write_date_batches(
                    next_new_date,
                    _day_file_batches(
                        new_paths[next_new_date], schema, config.row_group_size
                    ),
                )
                next_new_date = next(ordered_new_dates, None)
        finally:
            writer.close()

        if written_rows == 0:
            raise RuntimeError("月份结果为空，拒绝发布")
        final_dates = sorted((existing_dates - replaced_dates) | set(new_paths))
        if written_dates != final_dates:
            raise RuntimeError(
                "月份写入日期或顺序校验失败："
                f"expected={final_dates}, actual={written_dates}"
            )
        metadata = pq.read_metadata(temporary)
        if metadata.num_rows != written_rows:
            raise RuntimeError("月份临时文件行数校验失败")
        if pq.read_schema(temporary) != schema:
            raise RuntimeError("月份临时文件Schema校验失败")
        actual_dates = published_dates(temporary)
        if actual_dates != set(written_dates):
            raise RuntimeError("月份临时文件交易日校验失败")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    dates = [value.strftime("%Y%m%d") for value in written_dates]
    return {
        "path": str(destination),
        "rows": written_rows,
        "bytes": destination.stat().st_size,
        "date_count": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
    }


def write_manifest(root: Path, payload: dict) -> Path:
    """原子写入指标运行清单。"""

    path = root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _empty_frame(metric_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["trade_date", "minute", "symbol", metric_name])


def _parquet_compression_options(
    compression: str | None,
    compression_level: int | None,
) -> dict[str, str | int | None]:
    """只在显式配置时传递压缩级别，兼容不支持级别的Snappy。"""

    result: dict[str, str | int | None] = {"compression": compression}
    if compression_level is not None:
        result["compression_level"] = compression_level
    return result


def _index_day_paths(day_paths: list[Path]) -> dict[date, Path]:
    """按文件名解析单日分片，并拒绝同日重复输入。"""

    result: dict[date, Path] = {}
    for raw_path in day_paths:
        path = Path(raw_path)
        try:
            value = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(f"单日分片文件名必须为YYYYMMDD.parquet：{path}") from exc
        if value in result:
            raise ValueError(f"存在重复的单日分片：{value}")
        result[value] = path
    return result


def _day_file_batches(
    path: Path,
    schema: pa.Schema,
    batch_size: int,
):
    """按批读取框架刚生成的单日分片。"""

    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != schema:
        raise ValueError(f"单日分片Schema不一致：{path}")
    return parquet.iter_batches(
        batch_size=batch_size,
        columns=schema.names,
        use_threads=True,
    )


def _existing_month_batches(
    path: Path,
    schema: pa.Schema,
    batch_size: int,
):
    """单次顺序扫描已有月文件，并把跨日期批次切成单日片段。"""

    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != schema:
        raise ValueError(f"已有月份文件Schema不一致：{path}")
    date_index = schema.get_field_index("trade_date")
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=schema.names,
        use_threads=True,
    ):
        date_column = batch.column(date_index)
        if date_column.null_count:
            raise ValueError(f"已有月份文件trade_date包含空值：{path}")
        dates = date_column.to_numpy(zero_copy_only=False)
        boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [batch.num_rows]))
        for start, end in zip(starts, ends, strict=True):
            value = batch.column(date_index)[int(start)].as_py()
            yield value, batch.slice(int(start), int(end - start))
