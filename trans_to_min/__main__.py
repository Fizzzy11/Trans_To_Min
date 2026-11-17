"""命令行入口：``python -m trans_to_min run|worker|bench``。"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trans_to_min")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="按运行配置调度全量、日更或重算（默认只打印计划）")
    run.add_argument("plan", help="运行配置TOML")
    run.add_argument("--execute", action="store_true", help="真正派发作业")

    worker = commands.add_parser("worker", help="调度器内部使用：执行单个作业")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--task", required=True)
    worker.add_argument("--dates-file", required=True)
    worker.add_argument("--result", required=True)
    worker.add_argument("--date-workers", type=int, required=True)
    worker.add_argument("--block-threads", type=int, required=True)
    worker.add_argument("--reference-days", default="")
    worker.add_argument("--retain", action="store_true")

    bench = commands.add_parser("bench", help="在隔离目录跑一个真实大日，生成效率报告")
    bench.add_argument("plan", help="运行配置TOML（只取任务与输入根）")
    bench.add_argument("--task", required=True)
    bench.add_argument("--date", required=True, help="YYYY-MM-DD，应选近年的大日")
    bench.add_argument("--work-root", required=True, help="隔离输出目录，结束后删除")
    bench.add_argument("--report-dir", required=True, help="效率报告目录，写入<任务名>.json")
    bench.add_argument("--block-threads", type=int, default=16)

    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        from .plan import load_plan
        from .scheduler import run as run_plan

        return run_plan(load_plan(arguments.plan), execute=arguments.execute)
    if arguments.command == "worker":
        from .worker import main as worker_main

        return worker_main(arguments)
    from .bench import main as bench_main

    return bench_main(arguments)


if __name__ == "__main__":
    sys.exit(main())
