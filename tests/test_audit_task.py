"""验收检查脚本：合规任务通过，私藏框架、自建调度、分块模块用pandas、遗留中间数据被查出。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_task.py"
spec = importlib.util.spec_from_file_location("audit_task", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

CLEAN_GROUPS = '''
import numpy as np
from trans_to_min.blocks import BlockMetricGroup


class Good(BlockMetricGroup):
    def compute_block(self, block, metric_names):
        return {}
'''


def _task(root: Path) -> Path:
    task = root / "demo_task"
    (task / "demo_metrics").mkdir(parents=True)
    (task / "demo_metrics" / "__init__.py").write_text("", encoding="utf-8")
    (task / "demo_metrics" / "groups.py").write_text(CLEAN_GROUPS, encoding="utf-8")
    (task / "tests").mkdir()
    (task / "README.md").write_text("# 示例", encoding="utf-8")
    return task


class AuditTaskTest(unittest.TestCase):
    def test_clean_task_passes(self) -> None:
        with TemporaryDirectory() as temporary:
            task = _task(Path(temporary))
            self.assertEqual(audit.check_layout(task), [])
            self.assertEqual(audit.check_sources(task), [])

    def test_violations_are_reported(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)
            (task / "operations").mkdir()
            (task / "runtime" / "copy" / "trans_to_min").mkdir(parents=True)
            (task / "runtime" / "copy" / "trans_to_min" / "__init__.py").write_text("", encoding="utf-8")
            (task / "demo_metrics" / "control.py").write_text(
                "import subprocess, sys, hashlib\nsys.path.insert(0, 'x')\n"
                "import trans_to_min\ntrans_to_min.kernels = None\n", encoding="utf-8")
            (task / "demo_metrics" / "groups.py").write_text(
                "import pandas as pd\n" + CLEAN_GROUPS, encoding="utf-8")
            layout = " ".join(audit.check_layout(task))
            self.assertIn("operations", layout)
            self.assertIn("框架副本", layout)
            sources = " ".join(audit.check_sources(task))
            for text in ("自建进程调度", "sys.path", "改写框架", "导入了pandas", "管控"):
                self.assertIn(text, sources)
            output = root / "output"
            (output / "vol_min" / "x_min" / ".group_staging").mkdir(parents=True)
            (output / "vol_min" / "x_min" / "manifest.json").write_text(
                json.dumps({"framework_version": "1.4.0+private"}), encoding="utf-8")
            outputs = " ".join(audit.check_outputs(output, Path(temporary) / "framework"))
            self.assertIn("非正式框架版本", outputs)
            self.assertIn("遗留中间数据", outputs)
            scratch = root / "scratch"
            (scratch / "demo_task_trial").mkdir(parents=True)
            self.assertEqual(len(audit.check_scratch([scratch], "demo_task")), 1)


if __name__ == "__main__":
    unittest.main()
