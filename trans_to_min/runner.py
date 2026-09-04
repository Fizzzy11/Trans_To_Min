from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from ._version import __version__
from .adapter import L2InputAdapter, input_path, validate_matching_date_sets
from .config import RunConfig
from .metric import MinuteMetric
from .resources import (
    process_peak_memory_bytes,
    recommend_adaptive_date_workers,
    recommend_date_workers,
)
from .storage import (
    metric_root,
    month_path,
    publish_month,
    published_dates,
    write_day_frame,
    write_manifest,
)


LOGGER = logging.getLogger(__name__)


class _ImmediateResult:
    """为同步发布提供与Future一致的最小结果接口。"""

    def __init__(self, value: dict) -> None:
        self.value = value

    def result(self) -> dict:
        return self.value


def _publish_month_worker(connection, arguments: tuple) -> None:
    """在独立进程中发布月份，并通过单向管道返回结果或错误。"""

    try:
        connection.send(("ok", publish_month(*arguments)))
    except BaseException as exc:
        connection.send(("error", repr(exc), traceback.format_exc()))
        raise
    finally:
        connection.close()


class _MonthPublishProcess:
    """不在主进程创建后台线程的异步月份发布句柄。"""

    def __init__(self, arguments: tuple) -> None:
        context = mp.get_context("spawn")
        self.receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_publish_month_worker,
            args=(sender, arguments),
            name="trans-to-min-publisher",
        )
        self.process.start()
        sender.close()

    def result(self) -> dict:
        try:
            message = self.receiver.recv()
        except EOFError as exc:
            self.process.join()
            raise RuntimeError(
                f"月份发布进程异常退出：exitcode={self.process.exitcode}"
            ) from exc
        finally:
            self.receiver.close()
        self.process.join()
        if message[0] == "error":
            raise RuntimeError(f"月份发布失败：{message[1]}\n{message[2]}")
        if self.process.exitcode != 0:
            raise RuntimeError(
                f"月份发布进程异常退出：exitcode={self.process.exitcode}"
            )
        return message[1]

    def cancel(self) -> None:
        """主任务失败时终止仍在运行的发布子进程。"""

        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.receiver.close()


def _configure_worker(arrow_threads: int) -> None:
    """限制每个日期进程的Arrow线程池，避免多进程下线程过量。"""

    import pyarrow as pa

    pa.set_cpu_count(arrow_threads)
    pa.set_io_thread_count(arrow_threads)


def _worker_process_context():
    """选择不会从已初始化Arrow线程状态直接fork的进程启动方式。"""

    methods = mp.get_all_start_methods()
    return mp.get_context("forkserver" if "forkserver" in methods else "spawn")


def _process_one_day(
    metric: MinuteMetric,
    trade_date: date,
    config: RunConfig,
    staging_root: Path,
) -> dict:
    """子进程执行一个交易日的读取、计算和临时写入。"""

    started = time.perf_counter()
    adapter = L2InputAdapter(config.input_root)
    inputs = {
        dataset: adapter.read(dataset, trade_date, metric.input_columns[dataset])
        for dataset in metric.inputs
    }
    result = metric.compute_day(trade_date, inputs)
    path = staging_root / trade_date.strftime("%Y%m") / f"{trade_date:%Y%m%d}.parquet"
    output = write_day_frame(result, path, metric.name, metric.value_type, config)
    output.update(
        {
            "date": trade_date.isoformat(),
            "symbols": int(result["symbol"].nunique()) if not result.empty else 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": process_peak_memory_bytes(),
        }
    )
    return output


class MinuteConversionRunner:
    """按日期并行计算、按月检查点发布单个分钟数据。"""

    def __init__(self, metric: MinuteMetric, config: RunConfig) -> None:
        self.metric = metric
        self.config = config

    def discover_dates(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> list[date]:
        """发现输入日期；起止日期均为闭区间。"""

        values = validate_matching_date_sets(
            self.config.input_root,
            self.metric.inputs,
            start_date=start_date,
            end_date=end_date,
        )
        return [datetime.strptime(str(value), "%Y%m%d").date() for value in values]

    def run(self, dates: list[date]) -> dict:
        """按月计算并发布指定交易日，支持跳过已发布日期。"""

        if not dates:
            raise ValueError("dates不能为空")
        requested_dates = sorted(set(dates))
        existing_dates = self._find_existing_dates(requested_dates)
        if existing_dates and not (
            self.config.overwrite_dates or self.config.resume
        ):
            examples = [value.isoformat() for value in sorted(existing_dates)[:20]]
            raise FileExistsError(
                "输出已包含待生成日期；如需断点续跑请启用resume，"
                f"如需重算请启用overwrite_dates：{examples}"
            )
        pending_dates = (
            [value for value in requested_dates if value not in existing_dates]
            if self.config.resume
            else requested_dates
        )
        self._validate_input_files(pending_dates)

        recommendation = None
        if self.config.date_workers == 0 and pending_dates:
            recommendation = recommend_date_workers(
                self.metric,
                self.config,
                pending_dates,
            )
            effective_workers = recommendation.workers
            LOGGER.info(
                "自动并发 metric=%s workers=%d cpu=%d cpu_limit=%d memory_limit=%s "
                "input_mib=%d worker_mib=%d",
                self.metric.name,
                effective_workers,
                recommendation.cpu_count,
                recommendation.cpu_limit,
                recommendation.memory_limit,
                recommendation.estimated_input_bytes_per_day // (1024**2),
                recommendation.estimated_worker_bytes // (1024**2),
            )
        else:
            effective_workers = self.config.date_workers

        root = metric_root(self.config.output_root, self.metric.name)
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        payload = {
            "status": "running",
            "framework_version": __version__,
            "metric": self.metric.name,
            "run_id": run_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "requested_dates": [value.isoformat() for value in requested_dates],
            "pending_dates": [value.isoformat() for value in pending_dates],
            "skipped_dates": [
                value.isoformat()
                for value in requested_dates
                if value in existing_dates and self.config.resume
            ],
            "config": {
                **asdict(self.config),
                "input_root": str(self.config.input_root),
                "output_root": str(self.config.output_root),
            },
            "parallelism": {
                "requested_date_workers": self.config.date_workers,
                "effective_date_workers": effective_workers,
                "recommendation": (
                    recommendation.as_dict() if recommendation is not None else None
                ),
                "history": [],
            },
            "progress": {
                "requested": len(requested_dates),
                "pending": len(pending_dates),
                "skipped": len(requested_dates) - len(pending_dates),
                "completed": 0,
                "computed": 0,
            },
            "days": {},
            "months": {},
        }
        if not pending_dates:
            return self._finish_noop(root, payload, started)

        staging = root / ".staging" / run_id
        staging.mkdir(parents=True, exist_ok=False)
        grouped: dict[str, list[date]] = defaultdict(list)
        for value in pending_dates:
            grouped[value.strftime("%Y%m")].append(value)
        month_groups = sorted(grouped.items())

        try:
            computed = 0
            published_count = 0
            pending_publish = None
            try:
                for month_index, (month, month_dates) in enumerate(month_groups):
                    workers_this_month = min(effective_workers, len(month_dates))
                    LOGGER.info(
                        "开始月份 metric=%s month=%s dates=%d workers=%d arrow_threads=%d",
                        self.metric.name,
                        month,
                        len(month_dates),
                        workers_this_month,
                        self.config.arrow_threads_per_worker,
                    )
                    day_results = self._run_days(
                        month_dates,
                        staging,
                        date_workers=effective_workers,
                        completed_offset=computed,
                        total_dates=len(pending_dates),
                    )
                    computed += len(month_dates)
                    payload["progress"]["computed"] = computed
                    payload["days"].update(
                        {
                            item["date"]: {
                                key: value
                                for key, value in item.items()
                                if key != "path"
                            }
                            for item in day_results
                        }
                    )

                    if pending_publish is not None:
                        published_count = self._finish_month_publish(
                            pending_publish,
                            payload,
                            root,
                            staging,
                            published_count,
                            len(pending_dates),
                        )
                        pending_publish = None

                    observed_peaks = [
                        int(item["peak_rss_bytes"])
                        for item in day_results
                        if item.get("peak_rss_bytes") is not None
                    ]
                    observed_peak = max(observed_peaks, default=None)
                    history_item = {
                        "month": month,
                        "date_workers": workers_this_month,
                        "observed_peak_rss_bytes": observed_peak,
                    }
                    if (
                        self.config.date_workers == 0
                        and self.config.adaptive_workers
                        and observed_peak is not None
                        and month_index + 1 < len(month_groups)
                    ):
                        next_month, next_month_dates = month_groups[month_index + 1]
                        adaptive = recommend_adaptive_date_workers(
                            self.metric,
                            self.config,
                            month_dates,
                            next_month_dates,
                            observed_peak,
                        )
                        effective_workers = adaptive.workers
                        history_item["next_month"] = next_month
                        history_item["adaptive_recommendation"] = adaptive.as_dict()
                        LOGGER.info(
                            "动态并发 metric=%s current_month=%s next_month=%s "
                            "workers=%d peak_mib=%d calibrated_multiplier=%.4f",
                            self.metric.name,
                            month,
                            next_month,
                            effective_workers,
                            observed_peak // (1024**2),
                            adaptive.calibrated_memory_multiplier,
                        )
                    payload["parallelism"]["history"].append(history_item)

                    paths = [
                        staging / month / f"{value:%Y%m%d}.parquet"
                        for value in month_dates
                    ]
                    arguments = (
                        paths,
                        month_path(
                            self.config.output_root,
                            self.metric.name,
                            month,
                        ),
                        self.metric.name,
                        self.metric.value_type,
                        self.config,
                        set(month_dates),
                    )
                    if self.config.publish_while_computing:
                        pending_publish = (
                            month,
                            month_dates,
                            _MonthPublishProcess(arguments),
                        )
                    else:
                        pending_publish = (
                            month,
                            month_dates,
                            _ImmediateResult(publish_month(*arguments)),
                        )
                        published_count = self._finish_month_publish(
                            pending_publish,
                            payload,
                            root,
                            staging,
                            published_count,
                            len(pending_dates),
                        )
                        pending_publish = None

                if pending_publish is not None:
                    published_count = self._finish_month_publish(
                        pending_publish,
                        payload,
                        root,
                        staging,
                        published_count,
                        len(pending_dates),
                    )
                    pending_publish = None
            finally:
                if pending_publish is not None:
                    future = pending_publish[2]
                    if isinstance(future, _MonthPublishProcess):
                        future.cancel()

            payload["status"] = "complete"
            payload["parallelism"]["final_effective_date_workers"] = effective_workers
            payload["elapsed_seconds"] = round(
                time.perf_counter() - started, 3
            )
            payload["finished_at"] = datetime.now().astimezone().isoformat()
            write_manifest(root, payload)
            return payload
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = repr(exc)
            payload["elapsed_seconds"] = round(
                time.perf_counter() - started, 3
            )
            payload["finished_at"] = datetime.now().astimezone().isoformat()
            write_manifest(root, payload)
            raise
        finally:
            if payload["status"] == "complete":
                shutil.rmtree(staging)

    def _finish_noop(self, root: Path, payload: dict, started: float) -> dict:
        """记录恢复模式下无需计算的成功任务。"""

        payload["status"] = "complete"
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        payload["finished_at"] = datetime.now().astimezone().isoformat()
        write_manifest(root, payload)
        LOGGER.info(
            "无需计算 metric=%s skipped=%d",
            self.metric.name,
            payload["progress"]["skipped"],
        )
        return payload

    def _run_days(
        self,
        dates: list[date],
        staging: Path,
        *,
        date_workers: int,
        completed_offset: int,
        total_dates: int,
    ) -> list[dict]:
        """计算一个月份；方法结束即销毁进程池并归还工作进程内存。"""

        if date_workers == 1 or len(dates) == 1:
            _configure_worker(self.config.arrow_threads_per_worker)
            results = []
            for index, value in enumerate(dates, start=1):
                item = _process_one_day(self.metric, value, self.config, staging)
                results.append(item)
                self._log_day_progress(
                    item,
                    completed_offset + index,
                    total_dates,
                )
            return results

        results = []
        max_workers = min(date_workers, len(dates))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_configure_worker,
            initargs=(self.config.arrow_threads_per_worker,),
            mp_context=_worker_process_context(),
        ) as executor:
            futures = {
                executor.submit(
                    _process_one_day,
                    self.metric,
                    value,
                    self.config,
                    staging,
                ): value
                for value in dates
            }
            for index, future in enumerate(as_completed(futures), start=1):
                item = future.result()
                results.append(item)
                self._log_day_progress(
                    item,
                    completed_offset + index,
                    total_dates,
                )
        return sorted(results, key=lambda item: item["date"])

    def _finish_month_publish(
        self,
        pending_publish,
        payload: dict,
        root: Path,
        staging: Path,
        published_count: int,
        total_dates: int,
    ) -> int:
        """等待一个月份发布完成并记录可恢复检查点。"""

        month, month_dates, future = pending_publish
        published = future.result()
        payload["months"][month] = published
        published_count += len(month_dates)
        payload["progress"]["completed"] = published_count
        payload["last_checkpoint_at"] = datetime.now().astimezone().isoformat()
        write_manifest(root, payload)
        shutil.rmtree(staging / month)
        LOGGER.info(
            "月份发布完成 metric=%s month=%s progress=%d/%d rows=%d",
            self.metric.name,
            month,
            published_count,
            total_dates,
            published["rows"],
        )
        return published_count

    def _log_day_progress(
        self,
        item: dict,
        completed: int,
        total: int,
    ) -> None:
        LOGGER.info(
            "交易日完成 metric=%s date=%s progress=%d/%d elapsed=%.3fs "
            "symbols=%d peak_rss_mib=%s",
            self.metric.name,
            item["date"],
            completed,
            total,
            item["elapsed_seconds"],
            item["symbols"],
            (
                int(item["peak_rss_bytes"]) // (1024**2)
                if item.get("peak_rss_bytes") is not None
                else "unknown"
            ),
        )

    def _find_existing_dates(self, dates: list[date]) -> set[date]:
        """在计算前发现请求范围内已经发布的交易日。"""

        grouped: dict[str, set[date]] = defaultdict(set)
        for value in dates:
            grouped[value.strftime("%Y%m")].add(value)
        existing: set[date] = set()
        for month, requested in grouped.items():
            existing.update(
                published_dates(
                    month_path(
                        self.config.output_root,
                        self.metric.name,
                        month,
                    )
                )
                & requested
            )
        return existing

    def _validate_input_files(self, dates: list[date]) -> None:
        """在创建临时目录前检查显式运行日期的全部输入文件。"""

        missing = [
            (value.isoformat(), dataset)
            for value in dates
            for dataset in self.metric.inputs
            if not input_path(self.config.input_root, dataset, value).is_file()
        ]
        if missing:
            examples = [f"{value}:{dataset}" for value, dataset in missing[:20]]
            raise FileNotFoundError(
                f"运行日期缺少逐笔输入文件，缺失示例：{examples}"
            )
