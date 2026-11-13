"""来源文件身份：只用于一次运行内发现来源被替换，不写入任何输出，也不参与续跑判断。"""
from __future__ import annotations

import hashlib
import json
import pyarrow.parquet as pq
from .adapter import input_path

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# 同一进程内同一份文件（按完整stat身份）的footer只读一次：一个日期进程要为组内每个
# 指标在计算前后各算一次契约，逐次重开footer在机械盘上都是随机读。
_FOOTER_CACHE: dict[tuple, tuple] = {}
_FOOTER_CACHE_LIMIT = 4096


def _footer(path, state):
    key = (str(path), state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
    cached = _FOOTER_CACHE.get(key)
    if cached is None:
        parquet = pq.ParquetFile(path)
        metadata = dict(parquet.schema_arrow.metadata or {})
        cached = (parquet.metadata.num_rows, str(parquet.schema_arrow.remove_metadata()), metadata)
        if len(_FOOTER_CACHE) >= _FOOTER_CACHE_LIMIT:
            _FOOTER_CACHE.clear()
        _FOOTER_CACHE[key] = cached
    return cached


def source_contract(metric, day, input_root, *, group=None):
    """当前来源日文件的身份摘要（stat与footer），并执行指标声明的报价版本门禁。

    只在运行中前后各取一次作比较，不落盘；已发布数据不保存也不核对任何凭证。
    group仅为兼容旧调用保留。
    """
    inputs = {}
    for dataset in metric.inputs:
        path = input_path(input_root, dataset, day).resolve()
        before = path.stat()
        rows, schema, metadata = _footer(path, before)
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
            "rows": rows, "schema": schema,
            "metadata": {k.hex(): v.hex() for k, v in metadata.items()},
        }
    return _digest({"contract_version": 2, "date": day.isoformat(), "inputs": inputs})
