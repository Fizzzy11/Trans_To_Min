"""统一运行入口：配置检查（跨平台）与端到端调度（仅Linux）。"""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import textwrap
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import input_path
from trans_to_min.plan import load_plan, load_task_groups
from trans_to_min.storage import month_path, published_dates


DAYS = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 2, 3), date(2025, 2, 4))

GROUPS_MODULE = '''
"""调度测试任务：分块量与笔数（本模块不导入pandas）。"""

import numpy as np

from trans_to_min import MinuteMetric, kernels
from trans_to_min.blocks import BlockMetricGroup, CellValues
from trans_to_min.plan import TaskGroup
from trans_to_min.reference_check import DayReference

from sched_reference import day_reference


class Volume(MinuteMetric):
    name = "sched_volume_min"
    inputs = ("chengjiao",)
    input_columns = {"chengjiao": ("symbol", "time_ms", "volume")}
    value_type = "uint64"
    empty_value = 0

    def prepare_events(self, trade_date, inputs):
        raise NotImplementedError


class Count(Volume):
    name = "sched_count_min"


class Group(BlockMetricGroup):
    def compute_block(self, block, metric_names):
        table = block.tables["chengjiao"]
        valid = table["minute"] >= 0
        cells = block.cells("chengjiao", valid)
        return {
            "sched_volume_min": CellValues(*kernels.sum_by(cells, table["volume"][valid])),
            "sched_count_min": CellValues(*kernels.count_by(cells)),
        }


class WholeDay(BlockMetricGroup.__mro__[1]):
    pass


def build_task_groups():
    return [TaskGroup(
        Group("sched_group", [Volume(), Count()]),
        {"sched_volume_min": "vol_min", "sched_count_min": "count_min"},
        DayReference(day_reference, level="测试：抽样日全股票独立复算"),
    )]


def build_whole_day_groups():
    return [TaskGroup(WholeDay("plain_group", [Volume()]), {},
                      DayReference(day_reference, level="测试"))]


def build_without_reference():
    return [TaskGroup(Group("sched_group", [Volume(), Count()]))]
'''

REFERENCE_MODULE = '''
"""独立参考：逐行累加，与生产的分段归约写法不同。"""

import numpy as np
import pyarrow.parquet as pq

from trans_to_min.minute import minute_codes


def day_reference(trade_date, sources):
    table = pq.ParquetFile(sources["chengjiao"].open()).read(columns=["symbol", "time", "volume"])
    symbols = sorted(set(table["symbol"].to_pylist()))
    index = {symbol: i for i, symbol in enumerate(symbols)}
    codes = minute_codes(np.asarray(table["time"]).astype(np.int64))
    volume = np.zeros((len(symbols), 242), dtype=np.uint64)
    count = np.zeros((len(symbols), 242), dtype=np.uint64)
    for symbol, code, value in zip(table["symbol"].to_pylist(), codes, table["volume"].to_pylist()):
        if code >= 0:
            volume[index[symbol], code] += value
            count[index[symbol], code] += 1
    return {"symbols": symbols,
            "values": {"sched_volume_min": volume, "sched_count_min": count}, "info": {}}
'''


def _write_task(root: Path) -> Path:
    path = root / "task"
    path.mkdir()
    (path / "sched_groups.py").write_text(textwrap.dedent(GROUPS_MODULE), encoding="utf-8")
    (path / "sched_reference.py").write_text(textwrap.dedent(REFERENCE_MODULE), encoding="utf-8")
    return path


def _write_inputs(root: Path) -> None:
    random = np.random.default_rng(11)
    for day in DAYS:
        rows = 400
        symbols = np.array(["000001.XSHE", "000002.XSHE", "600000.XSHG"])
        symbol = np.sort(random.integers(0, 3, rows))
        time = np.sort(random.integers(34_200_000, 54_000_000, rows))
        order = np.lexsort((time, symbol))
        table = pa.table({
            "symbol": pa.array(symbols[symbol[order]]),
            "date": pa.array([day] * rows, pa.date32()),
            "time": pa.array(time[order].astype(np.uint32)),
            "volume": pa.array(random.integers(1, 1000, rows).astype(np.uint32)),
        })
        path = input_path(root, "chengjiao", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def _write_plan(root: Path, task: Path, *, mode="daily", factory="sched_groups:build_task_groups",
                extra="") -> Path:
    path = root / "plan.toml"
    path.write_text(textwrap.dedent(f'''
        [run]
        mode = "{mode}"
        start = "2025-01-01"
        end = "2025-02-28"
        input_root = "{(root / "input").as_posix()}"
        output_root = "{(root / "output").as_posix()}"
        state_root = "{(root / "state").as_posix()}"
        {extra}

        [[task]]
        name = "sched_task"
        path = "{task.as_posix()}"
        factory = "{factory}"
    '''), encoding="utf-8")
    return path


class PlanValidationTest(unittest.TestCase):
    def test_plan_rules(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _write_task(root)
            plan = load_plan(_write_plan(root, task))
            self.assertEqual((plan.mode, plan.validation, plan.memory_fraction), ("daily", "light", 0.95))
            for extra, message in (
                ("memory_fraction = 0.99", "只能调低"),
                ('validation = "sample"', "validation"),
                ("unknown = 1", "未知字段"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    load_plan(_write_plan(root, task, extra=extra))
            with self.assertRaisesRegex(ValueError, "bench_report"):
                load_plan(_write_plan(root, task, mode="full"))
            inside = _write_plan(root, task).read_text(encoding="utf-8").replace(
                (root / "state").as_posix(), (root / "output" / "state").as_posix())
            (root / "inside.toml").write_text(inside, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "state_root"):
                load_plan(root / "inside.toml")

    def test_task_groups_must_be_blocks_with_reference(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _write_task(root)
            try:
                spec = load_plan(_write_plan(root, task)).tasks[0]
                self.assertEqual(len(load_task_groups(spec)), 1)
                for factory, message in (
                    ("sched_groups:build_whole_day_groups", "whole_day_reason"),
                    ("sched_groups:build_without_reference", "独立参考"),
                ):
                    spec = load_plan(_write_plan(root, task, factory=factory)).tasks[0]
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        load_task_groups(spec)
            finally:
                sys.path.remove(str(task.resolve()))
                for name in ("sched_groups", "sched_reference"):
                    sys.modules.pop(name, None)


@unittest.skipUnless(sys.platform.startswith("linux"), "调度器依赖/proc，仅在Linux上运行")
class SchedulerEndToEndTest(unittest.TestCase):
    def test_run_resume_and_cleanup(self) -> None:
        from trans_to_min.scheduler import run

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _write_task(root)
            _write_inputs(root / "input")
            plan = load_plan(_write_plan(root, task))
            previous = os.environ.get("TRANS_TO_MIN_STAGE_ROOT")
            os.environ["TRANS_TO_MIN_STAGE_ROOT"] = str(root / "stage")
            try:
                self.assertEqual(run(plan, execute=True), 0)
                state = json.loads((root / "state" / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "complete")
                self.assertEqual(sorted(state["jobs"]), ["sched_task/202501", "sched_task/202502"])
                report = json.loads((root / "state" / "run_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["tasks"]["sched_task"]["completed_days"], 4)
                # 代表日为最大输入日、首日、末日去重，最多3个。
                self.assertIn(len(report["tasks"]["sched_task"]["reference_days"]), (2, 3))
                for key in ("202501", "202502"):
                    result = json.loads(
                        (root / "state" / "sched_task" / key / "result.json").read_text(encoding="utf-8"))
                    group = result["groups"]["sched_group"]
                    self.assertEqual(group["validation"]["status"], "passed")
                    self.assertEqual(group["structure_errors"], [])
                output = root / "output" / "sched_task"
                self.assertEqual(
                    published_dates(month_path(output / "vol_min", "sched_volume_min", "202502")),
                    {date(2025, 2, 3), date(2025, 2, 4)})
                self.assertEqual(sorted(entry.name for entry in output.iterdir()),
                                 ["count_min", "vol_min"])
                self.assertFalse((root / "stage").exists())
                # 续跑：全部已发布，不再派发作业。
                (root / "state" / "state.json").unlink()
                self.assertEqual(run(load_plan(_write_plan(root, task)), execute=True), 0)
                state = json.loads((root / "state" / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["jobs"], {})
            finally:
                if previous is None:
                    os.environ.pop("TRANS_TO_MIN_STAGE_ROOT", None)
                else:
                    os.environ["TRANS_TO_MIN_STAGE_ROOT"] = previous
                sys.modules.pop("sched_groups", None)
                sys.modules.pop("sched_reference", None)


if __name__ == "__main__":
    unittest.main()
