"""供子进程回归使用的真实双工作进程入口，不用于生产任务。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 工作目录刻意不在仓库内，仍显式保证父进程和生成工作进程使用本次源码。
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import trans_to_min
from trans_to_min.metrics import MinuteTradeVolume
from trans_to_min.storage import month_path
from trans_to_min.task_entry import run_task_cli


class LoggingTradeVolume(MinuteTradeVolume):
    """保留参考成交量公式，仅在实际日期工作进程打印业务日志。"""

    def compute_day(self, trade_date: date, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        print(f"工作进程计算日志：{trade_date.isoformat()}", flush=True)
        return super().compute_day(trade_date, inputs)


def check_ready(metric, config, dates) -> None:
    """测试输入由父测试事先写入，仅检查所请求日期确实有输入。"""

    for day in dates:
        if not trans_to_min.input_path(config.input_root, "chengjiao", day).is_file():
            raise ValueError("测试输入缺失")


def validate_output(metric, config, dates) -> dict:
    """独立检查每个授权日期的股票和行数，并公开实际导入的框架路径。"""

    for day in dates:
        frame = pd.read_parquet(month_path(config.output_root, metric.name, day.strftime("%Y%m")))
        daily = frame.loc[frame["trade_date"] == day]
        if len(daily) != 242 or daily["symbol"].unique().tolist() != ["000001.XSHE"]:
            raise ValueError("测试输出的日期、股票或分钟行数不正确")
    return {
        "status": "passed",
        "dates": [day.isoformat() for day in dates],
        "level": "独立日期与股票行数检查",
        "framework_path": str(Path(trans_to_min.__file__).resolve()),
    }


if __name__ == "__main__":
    raise SystemExit(run_task_cli(LoggingTradeVolume(), check_ready=check_ready,
                                 validate_output=validate_output))
