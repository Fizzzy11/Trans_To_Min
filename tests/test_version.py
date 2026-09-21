"""基础版本及打包元数据的一致性测试。"""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

from trans_to_min import RunConfig, __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VersionTest(unittest.TestCase):
    def test_version_uses_runtime_single_source(self) -> None:
        """确认1.0.1生产版本及动态打包版本来源。"""

        self.assertEqual(__version__, "1.0.1")
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
            configuration = tomllib.load(stream)
        self.assertEqual(configuration["project"]["dynamic"], ["version"])
        self.assertEqual(
            configuration["tool"]["setuptools"]["dynamic"]["version"]["attr"],
            "trans_to_min._version.__version__",
        )

    def test_release_documents_declare_baseline_version(self) -> None:
        """确认对外文档与基础版本一致并可正常读取中文。"""

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("当前稳定版本为`1.0.1`", readme)
        self.assertIn("## 1.0.1（2026-09-21）", changelog)
        self.assertIn("/sd1-data/level2", readme)
        retired_root = "/" + "data" + "/level2"
        self.assertNotIn(retired_root, readme + changelog)
        self.assertNotIn("\ufffd", readme + changelog)

    def test_default_input_root_uses_current_level2_mount(self) -> None:
        """确认默认取数不会回退到迁移前目录。"""

        self.assertEqual(RunConfig().input_root, Path("/sd1-data/level2"))


if __name__ == "__main__":
    unittest.main()
