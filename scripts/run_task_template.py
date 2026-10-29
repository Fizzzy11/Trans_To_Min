"""业务任务标准入口模板；未接入业务门禁前只允许预演，不可用于正式生产。"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trans_to_min.metrics import MinuteTradeVolume
from trans_to_min.task_entry import InputsNotReady, run_task_cli


def check_ready(metric, config, dates) -> None:
    """替换为本业务的上游发布和成功质检检查，不得仅用文件存在判断。"""
    raise InputsNotReady("模板尚未接入真实上游发布与质检凭证；禁止生成生产数据")


def validate_output(metric, config, dates) -> dict:
    """替换为独立校验；返回status=passed、精确dates和非空level后方可成功。"""
    raise NotImplementedError("模板尚未接入本业务独立校验，不能返回虚假的通过结果")


def main(argv=None) -> int:
    # 正式接入时仅替换这里的指标类导入和构造，全量与日更始终使用同一个类。
    return run_task_cli(MinuteTradeVolume(), check_ready=check_ready,
                        validate_output=validate_output, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
