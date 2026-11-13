"""源文件单次读取与内存盘共享暂存。

逐笔日文件按文件身份（设备、inode、大小、mtime、ctime）在暂存根目录只保留一份
完整副本。任何任务、任何进程都通过 :func:`acquire` 取得句柄；生成、独立参考以及
读取同一张表的其他任务都从这一份字节解码，机械盘上的源文件只被顺序读取一次。

三种句柄模式：

``staged``  副本位于内存盘，按文件身份跨进程共享，是Linux下的默认模式；
``buffer``  进程私有的整文件内存副本；没有内存盘、非Linux或预算已满时退回；
``direct``  ``source_stage="off"``，直接把路径交给Arrow，与1.2.4行为一致。

引用计数完全由内核 ``flock`` 承担：每个持有者在副本上保持共享锁，进程即使被
kill锁也会自动释放，不会留下需要人工清理的计数。业务任务不得自行打开源文件，
也不需要了解本模块；框架执行器会替它们取得并释放句柄。
"""
from __future__ import annotations

import logging
import mmap
import os
import threading
import time
import uuid
from collections.abc import Hashable, Sequence
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa

try:  # Windows没有fcntl，只能使用进程私有的buffer模式。
    import fcntl
except ImportError:  # pragma: no cover - 仅非POSIX平台
    fcntl = None


LOGGER = logging.getLogger(__name__)
STAGE_MODES = ("auto", "off")
ROOT_ENVIRONMENT = "TRANS_TO_MIN_STAGE_ROOT"
DEFAULT_BUDGET_BYTES = 48 << 30
_CHUNK_BYTES = 16 << 20
_DIRECT_CHUNK_BYTES = 32 << 20
_STAGED_SUFFIX = ".src"
_TEMPORARY_SUFFIX = ".tmp"
_BUDGET_LOCK = ".budget.lock"


def file_identity(path: Path | str) -> dict[str, int]:
    """返回用于识别“同一份源文件”的stat身份。"""

    state = os.stat(path)
    return _state_identity(state)


def _state_identity(state: os.stat_result) -> dict[str, int]:
    return {
        "dev": state.st_dev,
        "ino": state.st_ino,
        "size": state.st_size,
        "mtime_ns": state.st_mtime_ns,
        "ctime_ns": state.st_ctime_ns,
    }


# Windows上按路径stat与按句柄fstat的st_ctime含义不同（元数据变更时间／创建时间），
# 同一文件也不相等；句柄复核只在POSIX上比较ctime。
_FSTAT_CTIME_COMPARABLE = os.name != "nt"


def _descriptor_matches(descriptor: int, identity: dict[str, int]) -> bool:
    """已打开的文件是否仍是按路径取得身份时的那一份。"""

    current = _state_identity(os.fstat(descriptor))
    if not _FSTAT_CTIME_COMPARABLE:
        current["ctime_ns"] = identity["ctime_ns"]
    return current == identity


def default_root() -> Path | None:
    """默认暂存根目录；平台不支持时返回None。"""

    configured = os.environ.get(ROOT_ENVIRONMENT)
    if configured:
        return Path(configured)
    if fcntl is None or not hasattr(os, "getuid"):
        return None
    base = Path("/dev/shm")
    if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
        return None
    return base / f"trans_to_min_stage_{os.getuid()}"


def _prepare_root(root: Path | str | None) -> Path | None:
    if fcntl is None:
        return None
    path = Path(root) if root is not None else default_root()
    if path is None:
        return None
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        state = path.stat()
    except OSError:
        return None
    if hasattr(os, "getuid") and state.st_uid != os.getuid():
        # 他人拥有的目录不可信：既不能保证内容，也不能保证可清理。
        return None
    return path


def _stage_name(identity: dict[str, int]) -> str:
    return "{dev:x}_{ino:x}_{size:x}_{mtime_ns:x}_{ctime_ns:x}".format(**identity) + _STAGED_SUFFIX


class SourceHandle:
    """一份源文件字节的持有凭证；关闭后不得再调用 :meth:`open`。"""

    def __init__(
        self,
        path: Path,
        identity: dict[str, int],
        mode: str,
        origin: str,
        read_seconds: float,
        *,
        buffer: pa.Buffer | None = None,
        descriptor: int | None = None,
        mapping: mmap.mmap | None = None,
        staged_path: Path | None = None,
        retain: bool = False,
    ) -> None:
        self.path = Path(path)
        self.identity = dict(identity)
        self.mode = mode
        self.origin = origin
        self.read_seconds = round(float(read_seconds), 3)
        self.staged_path = staged_path
        self._buffer = buffer
        self._descriptor = descriptor
        self._mapping = mapping
        self._retain = retain
        self._closed = False

    def open(self):
        """返回可交给 ``pq.ParquetFile`` 的来源；每次调用得到独立的读取游标。"""

        if self._closed:
            raise ValueError("来源句柄已关闭")
        if self.mode == "direct":
            return str(self.path)
        return pa.BufferReader(self._buffer)

    def verify_current(self) -> None:
        """确认磁盘上的源文件仍是取得句柄时的那一份。"""

        if file_identity(self.path) != self.identity:
            raise RuntimeError(f"来源文件在暂存后发生变化：{self.path}")

    def info(self) -> dict:
        return {
            "path": str(self.path),
            "mode": self.mode,
            "origin": self.origin,
            "bytes": self.identity["size"],
            "read_seconds": self.read_seconds,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer = None
        if self._mapping is not None:
            try:
                self._mapping.close()
            except BufferError:
                # 仍有Arrow对象引用映射；交给垃圾回收，映射在unlink后依然有效。
                pass
            self._mapping = None
        if self._descriptor is not None:
            try:
                if not self._retain:
                    _remove_if_unheld(self.staged_path, self._descriptor)
            finally:
                os.close(self._descriptor)
                self._descriptor = None

    def __enter__(self) -> "SourceHandle":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:  # 兜底，正常路径应显式close
        try:
            self.close()
        except Exception:
            pass


def _remove_if_unheld(path: Path | None, descriptor: int) -> bool:
    """本描述符能转成排它锁说明没有其他持有者，此时才删除副本。"""

    if path is None:
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    try:
        if os.fstat(descriptor).st_ino == os.stat(path).st_ino:
            os.unlink(path)
            return True
    except OSError:
        pass
    return False


def _try_remove(path: Path) -> int:
    """删除一份无人持有的副本或残留临时文件，返回释放的字节数。"""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        size = os.fstat(descriptor).st_size
        return size if _remove_if_unheld(path, descriptor) else 0
    finally:
        os.close(descriptor)


def _is_rotational(device: int) -> bool:
    """来源所在块设备是否为机械盘；无法判断时按非机械盘处理。"""

    base = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}")
    for candidate in (base / "queue" / "rotational", base / ".." / "queue" / "rotational"):
        try:
            return candidate.read_text().strip() == "1"
        except OSError:
            continue
    return False


@contextmanager
def _device_lock(stage: Path | None, device: int):
    """机械盘同一时刻只允许一个顺序读者；并发随机读会把吞吐压到个位数MB/s。"""

    if stage is None or not _is_rotational(device):
        yield
        return
    descriptor = os.open(stage / f".read_{device:x}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _staged_handle(path, identity, target, descriptor, origin, seconds, retain) -> SourceHandle:
    mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
    return SourceHandle(
        path, identity, "staged", origin, seconds,
        buffer=pa.py_buffer(mapping), descriptor=descriptor, mapping=mapping,
        staged_path=target, retain=retain,
    )


def _open_staged(path, identity, target, retain, started) -> SourceHandle | None:
    """打开已有副本并加共享锁；副本不存在或刚被最后持有者删除时返回None。"""

    while True:
        try:
            descriptor = os.open(target, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            state = os.fstat(descriptor)
            if state.st_nlink == 0:
                os.close(descriptor)
                continue
            if state.st_size != identity["size"]:
                raise RuntimeError(f"暂存副本大小与来源不符：{target}")
            try:
                os.utime(target)  # 记录最近使用，供预算淘汰排序
            except OSError:
                pass
            return _staged_handle(
                path, identity, target, descriptor, "hit",
                time.perf_counter() - started, retain,
            )
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise


def _usage(stage: Path) -> list[tuple[float, int, Path]]:
    entries = []
    with os.scandir(stage) as iterator:
        for entry in iterator:
            if not entry.name.endswith((_STAGED_SUFFIX, _TEMPORARY_SUFFIX)):
                continue
            try:
                state = entry.stat()
            except OSError:
                continue
            entries.append((state.st_mtime, state.st_size, Path(entry.path)))
    return entries


def _reserve(stage: Path, target: Path, size: int, budget_bytes: int):
    """在预算锁内确认容量并创建带共享锁的临时文件；放不下时返回None。"""

    lock = os.open(stage / _BUDGET_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = sorted(_usage(stage))
        used = sum(item[1] for item in entries)
        for _mtime, _size, path in entries:
            if used + size <= budget_bytes:
                break
            used -= _try_remove(path)
        if used + size > budget_bytes:
            return None
        temporary = stage / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}{_TEMPORARY_SUFFIX}"
        descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            # 先加共享锁再让文件可见，清扫者不会把写入中的文件当作残留删除。
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            os.ftruncate(descriptor, size)
        except BaseException:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        return descriptor, temporary
    finally:
        os.close(lock)


def _copy_direct(path: Path, identity: dict[str, int], descriptor: int) -> bool:
    """用O_DIRECT大块读把源文件抄进副本；文件系统不支持时返回False，由调用方改用普通读。

    普通读经页缓存，每次只向磁盘提交一个预读窗口（通常128K）；机械盘上有其他进程读写时，
    每个窗口之间都会被插队，实测吞吐只有大块直读的三分之一。直读一次提交32M，磁头连续读完，
    也不在页缓存里再留一份。
    """

    if not hasattr(os, "O_DIRECT") or not hasattr(os, "readv"):
        return False
    try:
        source = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return False
    buffer = mmap.mmap(-1, _DIRECT_CHUNK_BYTES)
    try:
        if not _descriptor_matches(source, identity):
            raise RuntimeError(f"来源文件在读取前发生变化：{path}")
        view = memoryview(buffer)
        try:
            remaining = identity["size"]
            while remaining:
                try:
                    count = os.readv(source, [buffer])
                except OSError:
                    if remaining == identity["size"]:
                        return False  # 打开成功但首次直读被拒绝（如内存盘、部分网络文件系统）
                    raise
                if not count:
                    raise RuntimeError(f"来源文件读取提前结束：{path}")
                count = min(count, remaining)
                written = 0
                while written < count:
                    written += os.write(descriptor, view[written:count])
                remaining -= count
        finally:
            view.release()
        if not _descriptor_matches(source, identity):
            raise RuntimeError(f"来源文件在读取期间发生变化：{path}")
        return True
    finally:
        buffer.close()
        os.close(source)


def _copy_source(path: Path, identity: dict[str, int], descriptor: int) -> None:
    if _copy_direct(path, identity, descriptor):
        return
    os.lseek(descriptor, 0, os.SEEK_SET)
    source = os.open(path, os.O_RDONLY)
    try:
        if not _descriptor_matches(source, identity):
            raise RuntimeError(f"来源文件在读取前发生变化：{path}")
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(source, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        remaining = identity["size"]
        while remaining:
            block = os.read(source, min(_CHUNK_BYTES, remaining))
            if not block:
                raise RuntimeError(f"来源文件读取提前结束：{path}")
            view = memoryview(block)
            while view:
                view = view[os.write(descriptor, view):]
            remaining -= len(block)
        if not _descriptor_matches(source, identity):
            raise RuntimeError(f"来源文件在读取期间发生变化：{path}")
        if hasattr(os, "posix_fadvise"):
            # 副本已在内存盘，页缓存再留一份只会重复占用内存。
            os.posix_fadvise(source, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(source)


def _read_private(path: Path, identity: dict[str, int], started: float) -> SourceHandle:
    with open(path, "rb") as stream:
        if not _descriptor_matches(stream.fileno(), identity):
            raise RuntimeError(f"来源文件在读取前发生变化：{path}")
        data = stream.read()
        if not _descriptor_matches(stream.fileno(), identity) or len(data) != identity["size"]:
            raise RuntimeError(f"来源文件在读取期间发生变化：{path}")
    return SourceHandle(
        path, identity, "buffer", "read", time.perf_counter() - started,
        buffer=pa.py_buffer(data),
    )


def acquire(
    path: Path | str,
    *,
    mode: str = "auto",
    root: Path | str | None = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    retain: bool = False,
    staged_only: bool = False,
    cancel: threading.Event | None = None,
) -> SourceHandle | None:
    """取得一份源文件字节。

    ``retain=False`` 时最后一个持有者关闭即删除副本，单任务独立运行不留任何暂存；
    ``retain=True`` 时副本留给稍后读取同一张表的其他任务，由发起方（调度器）用
    :func:`clear` 收尾，期间超出预算按最久未用淘汰无人持有的副本。
    ``staged_only`` 供预读使用：不能放进共享暂存（或已被取消）时返回None。
    """

    if mode not in STAGE_MODES:
        raise ValueError(f"source_stage只支持{STAGE_MODES}：{mode!r}")
    path = Path(path)
    started = time.perf_counter()
    identity = file_identity(path)
    if mode == "off":
        return None if staged_only else SourceHandle(path, identity, "direct", "read", 0.0)
    stage = _prepare_root(root)
    if stage is None or identity["size"] == 0:
        return None if staged_only else _read_private(path, identity, started)

    target = stage / _stage_name(identity)
    handle = _open_staged(path, identity, target, retain, started)
    if handle is not None:
        return handle
    with _device_lock(stage, identity["dev"]):
        if cancel is not None and cancel.is_set():
            return None
        # 等读盘锁期间其他进程可能已放好同一份副本。
        handle = _open_staged(path, identity, target, retain, started)
        if handle is not None:
            return handle
        reserved = _reserve(stage, target, identity["size"], int(budget_bytes))
        if reserved is None:
            return None if staged_only else _read_private(path, identity, started)
        descriptor, temporary = reserved
        try:
            _copy_source(path, identity, descriptor)
            os.fchmod(descriptor, 0o400)
            try:
                # link在目标已存在时失败，绝不替换其他进程正在使用的副本。
                os.link(temporary, target)
            except FileExistsError:
                os.close(descriptor)
                descriptor = None
                handle = _open_staged(path, identity, target, retain, started)
                if handle is not None:
                    return handle
                return None if staged_only else _read_private(path, identity, started)
            return _staged_handle(
                path, identity, target, descriptor, "read",
                time.perf_counter() - started, retain,
            )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise
        finally:
            temporary.unlink(missing_ok=True)


def stage_files(paths: Sequence[Path | str], **options) -> list[SourceHandle]:
    """调度器显式预放一批源文件；返回的句柄在关闭前保证副本不被删除或淘汰。"""

    handles: list[SourceHandle] = []
    try:
        for path in paths:
            handle = acquire(path, **options)
            if handle is not None:
                handles.append(handle)
    except BaseException:
        release(handles)
        raise
    return handles


def release(handles: Sequence[SourceHandle]) -> None:
    for handle in handles:
        handle.close()


def sweep(root: Path | str | None = None, *, unheld: bool = False) -> dict:
    """清理残留临时文件；``unheld=True`` 时同时删除全部无人持有的副本。"""

    stage = _prepare_root(root)
    result = {"root": None if stage is None else str(stage), "removed": 0, "removed_bytes": 0,
              "remaining": 0, "remaining_bytes": 0}
    if stage is None:
        return result
    for _mtime, size, path in _usage(stage):
        freed = 0
        if unheld or path.name.endswith(_TEMPORARY_SUFFIX):
            freed = _try_remove(path)
        if freed or not path.exists():
            result["removed"] += 1
            result["removed_bytes"] += freed
        else:
            result["remaining"] += 1
            result["remaining_bytes"] += size
    return result


def clear(root: Path | str | None = None) -> dict:
    """收尾：删除全部无人持有的副本、锁文件和空的暂存根目录。"""

    stage = _prepare_root(root)
    result = sweep(stage, unheld=True) if stage is not None else sweep(root)
    if stage is None or result["remaining"]:
        return result
    for entry in list(stage.iterdir()):
        if entry.name == _BUDGET_LOCK or (entry.name.startswith(".read_") and entry.name.endswith(".lock")):
            entry.unlink(missing_ok=True)
    try:
        stage.rmdir()
        result["root_removed"] = True
    except OSError:
        result["root_removed"] = False
    return result


def config_options(config) -> dict:
    """把RunConfig翻译成 :func:`acquire` 的参数。"""

    return {
        "mode": config.source_stage,
        "root": config.source_stage_root,
        "budget_bytes": int(config.source_stage_budget_gb * (1 << 30)),
        "retain": config.source_stage_retain,
    }


class Prestager:
    """父进程里的顺序预读线程。

    按日期顺序把即将计算的来源放进共享暂存并持有到该日完成：日期进程忙于计算时
    磁盘不空转，机械盘始终只有一个顺序读者。预读只是提速手段，任何失败都不影响
    正确性——日期进程自己调用 :func:`acquire` 时会补读并暴露真正的错误。
    """

    def __init__(
        self,
        items: Sequence[tuple[Hashable, Sequence[Path]]],
        options: dict,
        *,
        window: int,
    ) -> None:
        self._items = list(items)
        self._options = dict(options)
        self._window = max(1, int(window))
        self._held: dict[Hashable, list[SourceHandle]] = {}
        self._finished: set[Hashable] = set()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Prestager":
        enabled = (
            self._options.get("mode", "auto") == "auto"
            and _prepare_root(self._options.get("root")) is not None
            and len(self._items) > 0
        )
        if enabled:
            self._thread = threading.Thread(
                target=self._run, name="trans-to-min-prestager", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            # 线程可能正阻塞在读盘锁上；它是daemon线程，取得锁后看到取消标志即退出。
            self._thread.join(timeout=30)
        with self._condition:
            held, self._held = self._held, {}
        for handles in held.values():
            release(handles)

    def done(self, key: Hashable) -> None:
        """某日已计算完成（或失败），释放为它预读的副本。"""

        with self._condition:
            self._finished.add(key)
            handles = self._held.pop(key, [])
            self._condition.notify_all()
        release(handles)

    def _run(self) -> None:
        try:
            for key, paths in self._items:
                while not self._stop.is_set():
                    with self._condition:
                        if key in self._finished:
                            break
                        if len(self._held) >= self._window:
                            self._condition.wait(timeout=5)
                            continue
                    handles = self._stage(paths)
                    if handles is None:
                        # 预算已满：等有日期完成后再试。
                        with self._condition:
                            self._condition.wait(timeout=5)
                        continue
                    with self._condition:
                        if key in self._finished or self._stop.is_set():
                            keep = False
                        else:
                            self._held[key] = handles
                            keep = True
                    if not keep:
                        release(handles)
                    break
                if self._stop.is_set():
                    return
        except Exception as exc:  # 预读失败不终止任务
            LOGGER.warning("来源预读提前结束：%r", exc)

    def _stage(self, paths: Sequence[Path]) -> list[SourceHandle] | None:
        handles: list[SourceHandle] = []
        for path in paths:
            handle = acquire(path, staged_only=True, cancel=self._stop, **self._options)
            if handle is None:
                release(handles)
                return None
            handles.append(handle)
        return handles
