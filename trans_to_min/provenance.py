"""用逐日输入身份和计算口径保护续跑；不扫描大文件全部业务列。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq
from ._version import __version__
from .adapter import input_path
from .group import MinuteMetricGroup


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _module_identity(obj):
    module = sys.modules[type(obj).__module__]
    path = Path(module.__file__)
    if not path.is_file():
        raise ValueError("续跑口径要求可追溯的模块源码")
    # 同包辅助公式也参与身份，避免只改导入的共享函数仍误跳过。
    return _digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(path.parent.glob("*.py"))})


def source_contract(metric, day, input_root, *, group=None):
    """来源为不可变日文件；stat、footer和代码身份变化都使旧输出失效。

    不替代全文件内容checksum；mtime等也被人为复原的原位篡改不在此门禁保证内。
    新报价版本并不由日期推断，指标可显式要求特定版本。
    """
    inputs = {}
    for dataset in metric.inputs:
        path = input_path(input_root, dataset, day).resolve()
        before = path.stat()
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        required = getattr(metric, "required_price_policy_version", None)
        if required is not None and metadata.get(b"price_policy_version") != str(required).encode():
            raise ValueError(f"{dataset}要求报价版本{required}，不能消费旧源或未知口径")
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError("读取来源契约期间输入发生变化")
        inputs[dataset] = {
            "path": str(path), "device": after.st_dev, "inode": after.st_ino,
            "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns,
            "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow.remove_metadata()),
            "metadata": {k.hex(): v.hex() for k, v in metadata.items()},
        }
    root = Path(__file__).parent
    framework = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    fields = ("name", "inputs", "input_columns", "value_type", "empty_value",
              "include_pre_market", "include_post_market", "algorithm_version",
              "contract_parameters", "required_price_policy_version")
    definition = {name: getattr(metric, name, None) for name in fields}
    # 默认组只是委派同一指标；自定义共享公式仍参与身份，不能静默混用。
    custom_group = group is not None and type(group).iter_compute_day is not MinuteMetricGroup.iter_compute_day
    return _digest({
        "contract_version": 1, "date": day.isoformat(), "inputs": inputs,
        "framework_version": __version__, "framework_source": framework,
        "metric_module": _module_identity(metric), "definition": definition,
        "group_module": _module_identity(group) if custom_group else None,
        "group_name": group.name if custom_group else None,
    })


def read_contracts(path):
    if not Path(path).is_file():
        return {}
    raw = (pq.read_schema(path).metadata or {}).get(b"source_contracts")
    if raw is None:
        return {}
    value = json.loads(raw.decode())
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
        or any(c not in "0123456789abcdef" for c in v) for k, v in value.items()
    ):
        raise ValueError(f"分钟文件来源契约损坏：{path}")
    return value


def validate_resume(path, metric, dates, input_root, *, group=None):
    """仅相同指标×日期×来源×算法可跳过；旧无凭证输出明确停止。"""
    stored = read_contracts(path)
    for day in dates:
        expected = source_contract(metric, day, input_root, group=group)
        if stored.get(day.isoformat()) != expected:
            raise ValueError(f"{metric.name}/{day}来源或算法口径不匹配，不能resume；需审核后显式重建")
