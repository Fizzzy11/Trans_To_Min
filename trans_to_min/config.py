from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("weituo", "chengjiao", "chedan")
TIMEZONE = "Asia/Shanghai"
MINUTE_COUNT = 242
METRIC_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*_min$"
)


def validate_metric_name(name: str) -> str:
    """校验指标名，避免非法列名和输出目录路径逃逸。"""

    if not isinstance(name, str) or METRIC_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "指标名必须采用小写英文业务名称_min格式，"
            f"且不得包含路径分隔符、点或空白：{name!r}"
        )
    return name


@dataclass(frozen=True)
class RunConfig:
    """逐笔转分钟任务的运行配置。"""

    input_root: Path = Path("/data/level2")
    output_root: Path = Path("/data/zhangyuan/trans_to_min")
    date_workers: int = 0
    arrow_threads_per_worker: int = 4
    row_group_size: int = 500_000
    compression: str = "zstd"
    compression_level: int = 3
    staging_compression: str | None = "snappy"
    staging_compression_level: int | None = None
    auto_memory_fraction: float = 0.70
    auto_worker_memory_multiplier: float = 16.0
    adaptive_workers: bool = True
    adaptive_memory_safety_factor: float = 1.20
    publish_while_computing: bool = True
    overwrite_dates: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        if self.date_workers < 0:
            raise ValueError("date_workers不能为负数，0表示自动选择")
        if self.arrow_threads_per_worker < 1:
            raise ValueError("arrow_threads_per_worker 必须为正整数")
        if self.row_group_size < 1:
            raise ValueError("row_group_size 必须为正整数")
        if self.staging_compression is not None and not self.staging_compression:
            raise ValueError("staging_compression必须为非空字符串或None")
        if (
            self.staging_compression_level is not None
            and self.staging_compression in (None, "snappy")
        ):
            raise ValueError("未压缩或Snappy临时分片不支持设置压缩级别")
        if not 0 < self.auto_memory_fraction <= 1:
            raise ValueError("auto_memory_fraction必须在(0, 1]范围内")
        if self.auto_worker_memory_multiplier < 1:
            raise ValueError("auto_worker_memory_multiplier不能小于1")
        if self.adaptive_memory_safety_factor < 1:
            raise ValueError("adaptive_memory_safety_factor不能小于1")
        if self.overwrite_dates and self.resume:
            raise ValueError("overwrite_dates和resume不能同时启用")
