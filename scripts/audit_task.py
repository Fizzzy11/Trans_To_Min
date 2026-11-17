"""任务验收检查（只读）：由用户或验收方在交付时运行，不进入任何生成流程。

检查任务目录是否遵守框架约束：没有私藏或改动框架、没有自建调度与管控、分块计算不用pandas、
输出由正式框架生成、运行配置在资源上限内，以及没有遗留的中间数据。发现问题即不验收。

用法：
  python scripts/audit_task.py --task-dir 任务目录 [--output-dir 任务输出目录]
      [--plan 运行配置.toml] [--scratch /data/zhangyuan --scratch /dev/shm] [--report 报告.json]
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import tomllib


ALLOWED_TOP_LEVEL = {
    "README.md", "CHANGELOG.md", "AGENTS.md", "generation_spec.json", "pyproject.toml",
    "docs", "tests", "metric_docs", "reports", "run", ".gitignore",
}
PROCESS_MODULES = {"subprocess", "multiprocessing", "concurrent.futures.process"}
PROCESS_NAMES = {"ProcessPoolExecutor", "Popen", "fork", "spawn", "Pool"}
CONTROL_WORDS = ("hashlib", "sha256", "md5", "approved", "reuse_proof", "receipt")


def _python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _package_root(task_dir: Path) -> list[Path]:
    return [entry for entry in task_dir.iterdir()
            if entry.is_dir() and (entry / "__init__.py").exists() and entry.name != "tests"]


def check_layout(task_dir: Path) -> list[str]:
    problems = []
    packages = {entry.name for entry in _package_root(task_dir)}
    for entry in task_dir.iterdir():
        if entry.name in ALLOWED_TOP_LEVEL or entry.name in packages or entry.name.startswith("."):
            continue
        problems.append(f"任务目录含不允许的条目：{entry.name}（调度、运维、框架副本不得放在任务内）")
    for path in task_dir.rglob("trans_to_min"):
        if path.is_dir() and (path / "__init__.py").exists():
            problems.append(f"任务目录内有框架副本：{path}")
    return problems


def check_sources(task_dir: Path) -> list[str]:
    problems = []
    for path in _python_files(task_dir):
        relative = path.relative_to(task_dir)
        is_test = relative.parts[0] == "tests"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            problems.append(f"{relative}无法解析：{exc}")
            continue
        imports = set()
        defines_block_group = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.ClassDef):
                if any(getattr(base, "id", getattr(base, "attr", "")) == "BlockMetricGroup"
                       for base in node.bases):
                    defines_block_group = True
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    text = ast.unparse(target)
                    if text.startswith(("trans_to_min.", "sys.path")):
                        problems.append(f"{relative}:{node.lineno} 改写框架或sys.path：{text}")
            elif isinstance(node, ast.Call):
                text = ast.unparse(node.func)
                if text in {"sys.path.insert", "sys.path.append", "setattr"} and not is_test:
                    first = ast.unparse(node.args[0]) if node.args else ""
                    if text != "setattr" or "trans_to_min" in first:
                        problems.append(f"{relative}:{node.lineno} 修改导入路径或框架属性：{text}({first})")
                if "patch" in text and "trans_to_min" in ast.unparse(node) and not is_test:
                    problems.append(f"{relative}:{node.lineno} 对框架打补丁")
        if not is_test:
            used = {name for name in imports if name.split(".")[0] in {"subprocess", "multiprocessing"}
                    or name.endswith(tuple(PROCESS_NAMES))}
            if used:
                problems.append(f"{relative} 自建进程调度：{sorted(used)}（统一用python -m trans_to_min run）")
            if defines_block_group and any(name.split(".")[0] == "pandas" for name in imports):
                problems.append(f"{relative} 定义分块指标组的模块导入了pandas")
            text = path.read_text(encoding="utf-8")
            words = [word for word in CONTROL_WORDS if word in text]
            if words:
                problems.append(f"{relative} 疑似自建凭证/审批管控（{words}），需人工复核")
    return problems


def check_outputs(output_dir: Path, framework_root: Path) -> list[str]:
    problems = []
    for manifest in output_dir.rglob("manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            problems.append(f"{manifest}无法读取")
            continue
        version = str(payload.get("framework_version", ""))
        if not version or "+" in version:
            problems.append(f"{manifest.parent.name}由非正式框架版本生成：{version!r}")
        module = payload.get("framework_module")
        if module and not Path(module).resolve().is_relative_to(framework_root):
            problems.append(f"{manifest.parent.name}由非正式安装路径的框架生成：{module}")
    for name in (".group_staging", ".reference_checkpoints", ".task.lock"):
        for path in output_dir.rglob(name):
            problems.append(f"输出目录遗留中间数据：{path}")
    return problems


def check_plan(plan_path: Path) -> list[str]:
    with plan_path.open("rb") as stream:
        raw = tomllib.load(stream)
    run = raw.get("run", {})
    problems = []
    if float(run.get("memory_fraction", 0.95)) > 0.95:
        problems.append(f"memory_fraction超过0.95：{run.get('memory_fraction')}")
    if run.get("validation", "light") != "light":
        problems.append(f"校验等级为{run.get('validation')}，非默认light，需说明理由")
    return problems


def check_scratch(roots: list[Path], task_name: str) -> list[str]:
    problems = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if task_name.lower() in entry.name.lower():
                problems.append(f"遗留中间数据：{entry}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="任务验收检查（只读）")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--scratch", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    task_dir = arguments.task_dir.resolve()
    import trans_to_min

    framework_root = Path(trans_to_min.__file__).resolve().parent.parent
    checks = {
        "目录结构": check_layout(task_dir),
        "源码约束": check_sources(task_dir),
    }
    if arguments.output_dir:
        checks["输出来源"] = check_outputs(arguments.output_dir.resolve(), framework_root)
    if arguments.plan:
        checks["运行配置"] = check_plan(arguments.plan)
    if arguments.scratch:
        checks["中间数据"] = check_scratch(arguments.scratch, task_dir.name)
    problems = sum(len(items) for items in checks.values())
    report = {
        "task_dir": str(task_dir), "framework": str(framework_root),
        "framework_version": trans_to_min.__version__,
        "status": "passed" if problems == 0 else "failed", "checks": checks,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.report:
        arguments.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
