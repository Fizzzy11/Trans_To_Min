from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from .adapter import L2InputAdapter, input_path
from .config import RunConfig
from .metric import MinuteMetric


MEBIBYTE = 1024**2


@dataclass(frozen=True)
class ParallelismRecommendation:
    """框架根据CPU、可用内存和输入列体积给出的并发建议。"""

    workers: int
    cpu_count: int
    available_memory_bytes: int | None
    estimated_input_bytes_per_day: int
    estimated_worker_bytes: int
    cpu_limit: int
    memory_limit: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "workers": self.workers,
            "cpu_count": self.cpu_count,
            "available_memory_bytes": self.available_memory_bytes,
            "estimated_input_bytes_per_day": self.estimated_input_bytes_per_day,
            "estimated_worker_bytes": self.estimated_worker_bytes,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
        }


@dataclass(frozen=True)
class AdaptiveParallelismRecommendation:
    """根据已观测峰值和下一月份输入体积校准后的并发建议。"""

    workers: int
    available_memory_bytes: int | None
    observed_peak_memory_bytes: int
    current_input_bytes_per_day: int
    next_input_bytes_per_day: int
    calibrated_memory_multiplier: float
    estimated_worker_bytes: int
    cpu_limit: int
    memory_limit: int | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "workers": self.workers,
            "available_memory_bytes": self.available_memory_bytes,
            "observed_peak_memory_bytes": self.observed_peak_memory_bytes,
            "current_input_bytes_per_day": self.current_input_bytes_per_day,
            "next_input_bytes_per_day": self.next_input_bytes_per_day,
            "calibrated_memory_multiplier": self.calibrated_memory_multiplier,
            "estimated_worker_bytes": self.estimated_worker_bytes,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
        }


def recommend_date_workers(
    metric: MinuteMetric,
    config: RunConfig,
    dates: list[date],
    *,
    sample_size: int = 5,
) -> ParallelismRecommendation:
    """在不读取业务数据内容的前提下估计适合的日期进程数。"""

    if not dates:
        raise ValueError("dates不能为空")
    cpu_count = max(1, os.cpu_count() or 1)
    input_bytes = _estimate_daily_input_bytes(metric, config.input_root, dates, sample_size)
    # Pandas对象列、业务中间表和输出补齐都会放大Arrow输入体积；保留256MiB底座。
    worker_bytes = max(
        256 * MEBIBYTE,
        int(input_bytes * config.auto_worker_memory_multiplier),
    )
    available = available_memory_bytes()
    memory_limit = (
        max(1, int(available * config.auto_memory_fraction) // worker_bytes)
        if available is not None
        else None
    )
    # Arrow线程主要作用于读取阶段，按每进程最多占用声明线程数约束总线程过量。
    cpu_limit = max(1, cpu_count // config.arrow_threads_per_worker)
    limits = [len(dates), cpu_limit]
    if memory_limit is not None:
        limits.append(memory_limit)
    workers = max(1, min(limits))
    return ParallelismRecommendation(
        workers=workers,
        cpu_count=cpu_count,
        available_memory_bytes=available,
        estimated_input_bytes_per_day=input_bytes,
        estimated_worker_bytes=worker_bytes,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
    )


def recommend_adaptive_date_workers(
    metric: MinuteMetric,
    config: RunConfig,
    current_dates: list[date],
    next_dates: list[date],
    observed_peak_memory_bytes: int,
) -> AdaptiveParallelismRecommendation:
    """用本月真实峰值校准下一月份的日期进程数。"""

    if not current_dates or not next_dates:
        raise ValueError("current_dates和next_dates不能为空")
    if observed_peak_memory_bytes < 1:
        raise ValueError("observed_peak_memory_bytes必须为正整数")
    current_input = _estimate_daily_input_bytes(
        metric, config.input_root, current_dates, sample_size=5
    )
    next_input = _estimate_daily_input_bytes(
        metric, config.input_root, next_dates, sample_size=5
    )
    if current_input > 0:
        observed_multiplier = observed_peak_memory_bytes / current_input
    else:
        observed_multiplier = config.auto_worker_memory_multiplier
    calibrated_multiplier = max(
        1.0,
        observed_multiplier * config.adaptive_memory_safety_factor,
    )
    worker_bytes = max(
        256 * MEBIBYTE,
        int(next_input * calibrated_multiplier),
    )
    available = available_memory_bytes()
    memory_limit = (
        max(1, int(available * config.auto_memory_fraction) // worker_bytes)
        if available is not None
        else None
    )
    cpu_count = max(1, os.cpu_count() or 1)
    cpu_limit = max(1, cpu_count // config.arrow_threads_per_worker)
    limits = [len(next_dates), cpu_limit]
    if memory_limit is not None:
        limits.append(memory_limit)
    return AdaptiveParallelismRecommendation(
        workers=max(1, min(limits)),
        available_memory_bytes=available,
        observed_peak_memory_bytes=observed_peak_memory_bytes,
        current_input_bytes_per_day=current_input,
        next_input_bytes_per_day=next_input,
        calibrated_memory_multiplier=round(calibrated_multiplier, 4),
        estimated_worker_bytes=worker_bytes,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
    )


def process_peak_memory_bytes() -> int | None:
    """返回当前进程生命周期内的常驻内存峰值。"""

    if os.name == "posix":
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if sys.platform == "darwin" else value * 1024
        except (ImportError, OSError, ValueError):
            return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("size", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.size = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            current_process = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(
                current_process,
                ctypes.byref(counters),
                counters.size,
            ):
                return int(counters.peak_working_set_size)
        except (AttributeError, OSError, ValueError):
            return None
    return None


def available_memory_bytes() -> int | None:
    """返回当前可用内存，并在Linux中同时尊重cgroup限制。"""

    candidates: list[int] = []
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
        cgroup_limit = Path("/sys/fs/cgroup/memory.max")
        cgroup_current = Path("/sys/fs/cgroup/memory.current")
        if cgroup_limit.is_file() and cgroup_current.is_file():
            raw_limit = cgroup_limit.read_text(encoding="ascii").strip()
            if raw_limit != "max":
                remaining = int(raw_limit) - int(
                    cgroup_current.read_text(encoding="ascii").strip()
                )
                candidates.append(max(0, remaining))
    elif os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                candidates.append(int(status.available_physical))
        except (AttributeError, OSError, ValueError):
            pass
    return min(candidates) if candidates else None


def _estimate_daily_input_bytes(
    metric: MinuteMetric,
    input_root: Path,
    dates: list[date],
    sample_size: int,
) -> int:
    """从Parquet元数据估计单日所需物理列的最大未压缩体积。"""

    ordered = sorted(set(dates))
    count = min(max(1, sample_size), len(ordered))
    if count == 1:
        sampled = [ordered[0]]
    else:
        sampled = [
            ordered[round(index * (len(ordered) - 1) / (count - 1))]
            for index in range(count)
        ]
    estimates: list[int] = []
    for value in sampled:
        day_bytes = 0
        for dataset in metric.inputs:
            path = input_path(input_root, dataset, value)
            parquet = pq.ParquetFile(path)
            physical = L2InputAdapter._physical_columns(
                dataset,
                metric.input_columns[dataset],
                parquet.schema_arrow,
            )
            column_indexes = [
                parquet.schema_arrow.get_field_index(name) for name in physical
            ]
            for row_group_index in range(parquet.metadata.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_index)
                day_bytes += sum(
                    row_group.column(column_index).total_uncompressed_size
                    for column_index in column_indexes
                )
        estimates.append(day_bytes)
    return max(estimates, default=0)
