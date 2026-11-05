"""默认输出迁移只替换根目录，显式路径和月份层级保持不变。"""

from pathlib import Path
import sys

from scripts.run_trade_volume import parse_args
from trans_to_min import RunConfig
from trans_to_min.storage import month_path
from trans_to_min.task_entry import run_task_cli
from trans_to_min.metrics import MinuteTradeVolume


def test_default_output_keeps_topic_category_metric_month_layout():
    """漏改默认根目录或改变指标月份层级时，最终文件位置应不匹配。"""
    category = RunConfig().output_root / "entropy_min" / "vol_entropy_min"
    actual = month_path(category, "cj_vol_entropy_min", "202609")
    expected = Path(
        "/sd1-data/trans_to_min/entropy_min/vol_entropy_min/"
        "cj_vol_entropy_min/2026/202609.parquet"
    ).resolve()
    assert actual == expected


def test_reference_entry_defaults_to_new_output_root(monkeypatch):
    """参考入口未传路径时，不应仍把新任务指向旧盘。"""
    monkeypatch.setattr(sys, "argv", ["run_trade_volume.py"])
    args = parse_args()
    assert args.output_root == Path("/sd1-data/trans_to_min")
    assert args.input_root == Path("/sd1-data/level2")


def test_explicit_output_root_is_not_redirected(monkeypatch, tmp_path):
    """用户显式选择的输出目录优先，不被新默认值改写。"""
    custom = tmp_path / "custom_min" / "basic_min"
    monkeypatch.setattr(sys, "argv", ["run_trade_volume.py", "--output-root", str(custom)])
    args = parse_args()
    config = RunConfig(output_root=args.output_root)
    assert month_path(config.output_root, "trade_volume_min", "202609") == (
        custom / "trade_volume_min" / "2026" / "202609.parquet"
    ).resolve()
    assert not custom.exists()


def test_standard_entry_still_requires_explicit_output_root(tmp_path):
    """新增默认路径不能使标准入口悄悄省略显式输出参数。"""
    report = tmp_path / "planned.json"
    code = run_task_cli(MinuteTradeVolume(), argv=[
        "--mode", "daily", "--date", "2026-09-30",
        "--input-root", str(tmp_path / "input"),
        "--result-json", str(report),
    ])
    assert code == 2
    assert not report.exists()
