"""运行配置（TOML）与任务加载。

一个运行配置描述一次全量、日更或重算：日期范围、输入输出根、校验等级、内存预算，以及
要一起跑的若干任务。任务只提供一个工厂函数，返回若干 :class:`TaskGroup`；调度、并行、
暂存、清理都由框架负责，任务目录里不需要也不允许自建调度。

运行配置示例::

    [run]
    mode = "full"                 # full：全量（已发布日期跳过）| daily：日更 | repair：重算覆盖
    start = "2024-01-01"
    end = "2026-09-30"
    output_root = "/sd1-data/trans_to_min"
    state_root = "/data/zhangyuan/某项目/runs/cj_size_2024"
    validation = "light"          # light：≤3个代表日独立复算 | full：全部日期独立复算
    bench_report = "/data/zhangyuan/某项目/bench"   # full模式必填：bench报告目录

    [[task]]
    name = "cj_size_min"
    path = "/app/workspace/zhangyuan/逐笔转分钟数据/cj_size_min"
    factory = "minute_metrics.groups:build_task_groups"
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
import importlib
from pathlib import Path
import sys
import tomllib

from .config import validate_metric_name
from .group import MinuteMetricGroup
from .reference_check import DayReference


MODES = ("full", "daily", "repair")
VALIDATIONS = ("light", "full")
MAX_MEMORY_FRACTION = 0.95
LIGHT_REFERENCE_DAYS = 3
_RUN_KEYS = {
    "mode", "start", "end", "input_root", "output_root", "state_root", "validation",
    "memory_fraction", "months_per_job", "bench_report", "source_stage_budget_gb",
}
_TASK_KEYS = {"name", "path", "factory", "output"}


@dataclass(frozen=True)
class TaskGroup:
    """任务工厂返回的一个指标组及其发布位置与独立参考。

    ``output_roots``把组内指标映射到任务输出目录下的子目录（如类别目录"vol_min"），
    省略的指标直接写在任务输出目录下。``reference``为独立参考，light与full校验都需要。
    """

    group: MinuteMetricGroup
    output_roots: Mapping[str, str] = field(default_factory=dict)
    reference: DayReference | None = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    path: Path
    factory: str
    output: str


@dataclass(frozen=True)
class RunPlan:
    path: Path
    mode: str
    start: date
    end: date
    input_root: Path
    output_root: Path
    state_root: Path
    validation: str
    memory_fraction: float
    months_per_job: int
    bench_report: Path | None
    source_stage_budget_gb: float
    tasks: tuple[TaskSpec, ...]

    def task(self, name: str) -> TaskSpec:
        for spec in self.tasks:
            if spec.name == name:
                return spec
        raise KeyError(f"运行配置中没有任务{name!r}")

    def task_output(self, spec: TaskSpec) -> Path:
        return self.output_root / spec.output


def _date(value, label: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label}须为YYYY-MM-DD日期：{value!r}") from exc


def load_plan(path: Path | str) -> RunPlan:
    """读取并检查运行配置；未知字段、越界参数直接报错。"""

    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = sorted(set(raw) - {"run", "task"})
    if unknown:
        raise ValueError(f"运行配置含未知段：{unknown}")
    run = dict(raw.get("run", {}))
    extra = sorted(set(run) - _RUN_KEYS)
    if extra:
        raise ValueError(f"[run]含未知字段：{extra}")
    mode = run.get("mode", "full")
    if mode not in MODES:
        raise ValueError(f"mode只支持{MODES}：{mode!r}")
    validation = run.get("validation", "light")
    if validation not in VALIDATIONS:
        raise ValueError(f"validation只支持{VALIDATIONS}：{validation!r}")
    fraction = float(run.get("memory_fraction", MAX_MEMORY_FRACTION))
    if not 0 < fraction <= MAX_MEMORY_FRACTION:
        raise ValueError(f"memory_fraction须在(0, {MAX_MEMORY_FRACTION}]内，任务只能调低：{fraction}")
    months_per_job = int(run.get("months_per_job", 1))
    if months_per_job < 1:
        raise ValueError("months_per_job须为正整数")
    stage_budget = float(run.get("source_stage_budget_gb", 48.0))
    if stage_budget <= 0:
        raise ValueError("source_stage_budget_gb须为正数")
    start, end = _date(run.get("start"), "start"), _date(run.get("end"), "end")
    if start > end:
        raise ValueError("start不能晚于end")
    for key in ("output_root", "state_root"):
        if not run.get(key):
            raise ValueError(f"[run]缺少{key}")
    output_root = Path(run["output_root"]).resolve(strict=False)
    state_root = Path(run["state_root"]).resolve(strict=False)
    if state_root == output_root or output_root in state_root.parents:
        raise ValueError("state_root不能位于output_root之内，运行状态不得混入数据目录")
    bench = run.get("bench_report")
    if mode == "full" and not bench:
        raise ValueError("full模式须提供bench_report（trans_to_min bench生成的效率报告目录）")
    tasks = []
    for item in raw.get("task", []):
        extra = sorted(set(item) - _TASK_KEYS)
        if extra:
            raise ValueError(f"[[task]]含未知字段：{extra}")
        for key in ("name", "path", "factory"):
            if not item.get(key):
                raise ValueError(f"[[task]]缺少{key}")
        if ":" not in item["factory"]:
            raise ValueError("factory须为'模块:函数'形式")
        tasks.append(TaskSpec(
            name=item["name"], path=Path(item["path"]).resolve(strict=False),
            factory=item["factory"], output=item.get("output", item["name"]),
        ))
    if not tasks:
        raise ValueError("运行配置至少需要一个[[task]]")
    names = [spec.name for spec in tasks]
    if len(set(names)) != len(names):
        raise ValueError(f"任务名重复：{names}")
    return RunPlan(
        path=path, mode=mode, start=start, end=end,
        input_root=Path(run.get("input_root", "/sd1-data/level2")),
        output_root=output_root, state_root=state_root, validation=validation,
        memory_fraction=fraction, months_per_job=months_per_job,
        bench_report=Path(bench).resolve(strict=False) if bench else None,
        source_stage_budget_gb=stage_budget, tasks=tuple(tasks),
    )


def load_task_groups(spec: TaskSpec) -> list[TaskGroup]:
    """导入任务工厂并检查返回值；新入口只接收分块指标组，整天计算须声明理由。"""

    if str(spec.path) not in sys.path:
        sys.path.insert(0, str(spec.path))
    module_name, _, function_name = spec.factory.partition(":")
    factory = getattr(importlib.import_module(module_name), function_name)
    items = list(factory())
    if not items:
        raise ValueError(f"{spec.name}的工厂没有返回任何指标组")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, TaskGroup):
            raise TypeError(f"{spec.name}的工厂必须返回TaskGroup列表")
        group = item.group
        if not getattr(group, "stock_separable", False):
            reason = getattr(group, "whole_day_reason", "")
            if not isinstance(reason, str) or not reason.strip():
                raise TypeError(
                    f"{spec.name}/{group.name}不是BlockMetricGroup；整天计算的指标组须在"
                    "whole_day_reason写明不能按股票分块的理由"
                )
        duplicated = seen & set(group.metric_names)
        if duplicated:
            raise ValueError(f"{spec.name}内指标重复：{sorted(duplicated)}")
        seen |= set(group.metric_names)
        for name, sub in item.output_roots.items():
            validate_metric_name(name)
            if name not in group.metric_names:
                raise ValueError(f"output_roots含组外指标：{name}")
            if Path(sub).is_absolute() or ".." in Path(sub).parts:
                raise ValueError(f"{name}的输出子目录须为任务输出目录下的相对路径：{sub}")
        if item.reference is None:
            raise ValueError(f"{spec.name}/{group.name}缺少独立参考，light与full校验都需要")
    return items


def metric_roots(plan: RunPlan, spec: TaskSpec, item: TaskGroup) -> dict[str, Path]:
    base = plan.task_output(spec)
    return {name: base / item.output_roots.get(name, "") for name in item.group.metric_names}


def task_datasets(items: list[TaskGroup]) -> tuple[str, ...]:
    names: list[str] = []
    for item in items:
        for dataset in item.group.inputs:
            if dataset not in names:
                names.append(dataset)
    return tuple(names)
