from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trans_to_min import MinuteConversionRunner, RunConfig
from trans_to_min.metrics import MinuteTradeVolume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成逐笔成交量分钟数据")
    parser.add_argument("--start-date", help="起始交易日，闭区间")
    parser.add_argument("--end-date", help="结束交易日，闭区间")
    parser.add_argument("--input-root", type=Path, default=Path("/data/level2"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zhangyuan/trans_to_min"),
    )
    parser.add_argument(
        "--date-workers",
        type=int,
        default=0,
        help="日期进程数；0表示根据CPU、可用内存和输入列体积自动选择",
    )
    parser.add_argument(
        "--arrow-threads-per-worker",
        type=int,
        default=4,
        help="每个日期进程允许使用的PyArrow线程数",
    )
    parser.add_argument("--row-group-size", type=int, default=500_000)
    parser.add_argument(
        "--staging-compression",
        choices=("none", "snappy", "zstd"),
        default="snappy",
        help="单日临时分片压缩；正式月份文件仍使用ZSTD",
    )
    parser.add_argument(
        "--staging-compression-level",
        type=int,
        help="临时分片压缩级别；Snappy不应设置",
    )
    parser.add_argument(
        "--auto-memory-fraction",
        type=float,
        default=0.70,
        help="自动并发最多使用的当前可用内存比例",
    )
    parser.add_argument(
        "--auto-worker-memory-multiplier",
        type=float,
        default=16.0,
        help="单进程峰值内存相对所需Parquet列未压缩体积的保守倍数",
    )
    parser.add_argument(
        "--no-adaptive-workers",
        action="store_true",
        help="关闭根据已完成月份真实峰值内存动态校准后续月份并发",
    )
    parser.add_argument(
        "--adaptive-memory-safety-factor",
        type=float,
        default=1.20,
        help="真实峰值内存校准时使用的安全系数",
    )
    parser.add_argument(
        "--no-overlap-publish",
        action="store_true",
        help="关闭月份流式发布与下一月计算的重叠执行",
    )
    parser.add_argument(
        "--overwrite-dates",
        action="store_true",
        help="覆盖月份文件中的同名交易日，保留该月其他日期",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已经发布的日期，从未完成日期继续运行",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = RunConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        date_workers=args.date_workers,
        arrow_threads_per_worker=args.arrow_threads_per_worker,
        row_group_size=args.row_group_size,
        staging_compression=(
            None if args.staging_compression == "none" else args.staging_compression
        ),
        staging_compression_level=args.staging_compression_level,
        auto_memory_fraction=args.auto_memory_fraction,
        auto_worker_memory_multiplier=args.auto_worker_memory_multiplier,
        adaptive_workers=not args.no_adaptive_workers,
        adaptive_memory_safety_factor=args.adaptive_memory_safety_factor,
        publish_while_computing=not args.no_overlap_publish,
        overwrite_dates=args.overwrite_dates,
        resume=args.resume,
    )
    runner = MinuteConversionRunner(MinuteTradeVolume(), config)
    dates = runner.discover_dates(args.start_date, args.end_date)
    result = runner.run(dates)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
