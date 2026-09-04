from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import shutil
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from ._version import __version__
from .adapter import L2InputAdapter, input_path, validate_matching_date_sets
from .config import RunConfig
from .group import MinuteMetricGroup
from .resources import (
    process_peak_memory_bytes,
    recommend_adaptive_date_workers,
    recommend_date_workers,
)
from .runner import _configure_worker, _worker_process_context
from .storage import (
    metric_root,
    month_path,
    publish_month,
    published_dates,
    write_day_frame,
    write_manifest,
)


LOGGER = logging.getLogger(__name__)


class _ImmediateGroupResult:
    """为同步发布提供与后台发布句柄一致的结果接口。"""

    def __init__(self, value: dict[str, dict]) -> None:
        self.value = value

    def result(self) -> dict[str, dict]:
        return self.value


def _publish_group_month(arguments: tuple[tuple[str, tuple], ...]) -> dict[str, dict]:
    """依次原子发布一个月份中的多个独立指标。"""

    result: dict[str, dict] = {}
    for metric_name, publish_arguments in arguments:
        result[metric_name] = publish_month(*publish_arguments)
    return result


def _publish_group_month_worker(connection, arguments: tuple) -> None:
    """在独立进程中发布一组月份指标。"""

    try:
        connection.send(("ok", _publish_group_month(arguments)))
    except BaseException as exc:
        connection.send(("error", repr(exc), traceback.format_exc()))
        raise
    finally:
        connection.close()


class _GroupMonthPublishProcess:
    """允许下一月份计算与当前月份多指标发布重叠。"""

    def __init__(self, arguments: tuple) -> None:
        context = mp.get_context("spawn")
        self.receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_publish_group_month_worker,
            args=(sender, arguments),
            name="trans-to-min-group-publisher",
        )
        self.process.start()
        sender.close()

    def result(self) -> dict[str, dict]:
        try:
            message = self.receiver.recv()
        except EOFError as exc:
            self.process.join()
            raise RuntimeError(
                f"指标组月份发布进程异常退出：exitcode={self.process.exitcode}"
            ) from exc
        finally:
            self.receiver.close()
        self.process.join()
        if message[0] == "error":
            raise RuntimeError(f"指标组月份发布失败：{message[1]}\n{message[2]}")
        if self.process.exitcode != 0:
            raise RuntimeError(
                f"指标组月份发布进程异常退出：exitcode={self.process.exitcode}"
            )
        return message[1]

    def cancel(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        self.receiver.close()


def _process_one_group_day(
    group: MinuteMetricGroup,
    trade_date: date,
    metric_names: tuple[str, ...],
    config: RunConfig,
    staging_root: Path,
) -> dict:
    """单日一次读取组内物理输入，并分别写入所选指标临时分片。"""

    started = time.perf_counter()
    adapter = L2InputAdapter(config.input_root)
    inputs = {
        dataset: adapter.read(dataset, trade_date, group.input_columns[dataset])
        for dataset in group.inputs
    }
    requested = set(metric_names)
    outputs: dict[str, dict] = {}
    for metric, frame in group.iter_compute_day(trade_date, inputs, metric_names):
        if metric.name not in requested:
            raise RuntimeError(
                f"指标组{group.name!r}产生了未选择的指标：{metric.name}"
            )
        if metric.name in outputs:
            raise RuntimeError(
                f"指标组{group.name!r}重复产生指标：{metric.name}"
            )
        path = (
            staging_root
            / trade_date.strftime("%Y%m")
            / metric.name
            / f"{trade_date:%Y%m%d}.parquet"
        )
        output = write_day_frame(
            frame,
            path,
            metric.name,
            metric.value_type,
            config,
        )
        output["symbols"] = int(frame["symbol"].nunique()) if not frame.empty else 0
        outputs[metric.name] = output
    missing = sorted(requested - set(outputs))
    if missing:
        raise RuntimeError(f"指标组{group.name!r}没有产生所选指标：{missing}")
    return {
        "date": trade_date.isoformat(),
        "metrics": outputs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": process_peak_memory_bytes(),
    }


class MinuteMetricGroupRunner:
    """共享逐笔读取并将多个指标独立发布的组执行器。"""

    def __init__(
        self,
        group: MinuteMetricGroup,
        config: RunConfig,
        output_roots: Mapping[str, Path] | None = None,
    ) -> None:
        self.group = group
        self.config = config
        roots = (
            {name: Path(config.output_root) for name in group.metric_names}
            if output_roots is None
            else {name: Path(path) for name, path in output_roots.items()}
        )
        missing = sorted(set(group.metric_names) - set(roots))
        extra = sorted(set(roots) - set(group.metric_names))
        if missing or extra:
            raise ValueError(
                f"output_roots必须与组内指标逐项对应：missing={missing}, extra={extra}"
            )
        self.output_roots = roots

    def discover_dates(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> list[date]:
        """发现组内全部物理输入共同覆盖的目标交易日。"""

        values = validate_matching_date_sets(
            self.config.input_root,
            self.group.inputs,
            start_date=start_date,
            end_date=end_date,
        )
        return [datetime.strptime(str(value), "%Y%m%d").date() for value in values]

    def run(self, dates: list[date]) -> dict:
        """按日期共享读取，按指标和月份独立检查点发布。"""

        if not dates:
            raise ValueError("dates不能为空")
        requested_dates = sorted(set(dates))
        existing_by_metric = self._find_existing_dates(requested_dates)
        conflicts = {
            name: sorted(values)
            for name, values in existing_by_metric.items()
            if values
        }
        if conflicts and not (self.config.overwrite_dates or self.config.resume):
            examples = {
                name: [value.isoformat() for value in values[:20]]
                for name, values in conflicts.items()
            }
            raise FileExistsError(
                "组内输出已包含待生成日期；如需断点续跑请启用resume，"
                f"如需重算请启用overwrite_dates：{examples}"
            )

        pending_by_metric = {
            name: (
                [value for value in requested_dates if value not in existing_by_metric[name]]
                if self.config.resume
                else list(requested_dates)
            )
            for name in self.group.metric_names
        }
        pending_by_date = {
            value: tuple(
                name
                for name in self.group.metric_names
                if value in pending_by_metric[name]
            )
            for value in requested_dates
        }
        pending_by_date = {
            value: names for value, names in pending_by_date.items() if names
        }
        pending_dates = sorted(pending_by_date)
        self._validate_input_files(pending_dates)

        recommendation = None
        if self.config.date_workers == 0 and pending_dates:
            recommendation = recommend_date_workers(
                self.group,
                self.config,
                pending_dates,
            )
            effective_workers = recommendation.workers
        else:
            effective_workers = self.config.date_workers

        run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        payload = self._initial_payload(
            run_id,
            requested_dates,
            pending_by_metric,
            existing_by_metric,
            effective_workers,
            recommendation,
        )
        if not pending_dates:
            payload["status"] = "complete"
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            payload["finished_at"] = datetime.now().astimezone().isoformat()
            self._write_metric_manifests(payload)
            return payload

        staging = (
            Path(self.config.output_root).resolve(strict=False)
            / ".group_staging"
            / self.group.name
            / run_id
        )
        staging.mkdir(parents=True, exist_ok=False)
        grouped: dict[str, list[date]] = defaultdict(list)
        for value in pending_dates:
            grouped[value.strftime("%Y%m")].append(value)
        month_groups = sorted(grouped.items())

        try:
            computed_dates = 0
            pending_publish = None
            try:
                for month_index, (month, month_dates) in enumerate(month_groups):
                    LOGGER.info(
                        "开始指标组月份 group=%s month=%s dates=%d workers=%d outputs=%d",
                        self.group.name,
                        month,
                        len(month_dates),
                        min(effective_workers, len(month_dates)),
                        sum(len(pending_by_date[value]) for value in month_dates),
                    )
                    day_results = self._run_days(
                        month_dates,
                        pending_by_date,
                        staging,
                        date_workers=effective_workers,
                    )
                    computed_dates += len(month_dates)
                    payload["progress"]["computed_dates"] = computed_dates
                    for item in day_results:
                        for metric_name, output in item["metrics"].items():
                            payload["metrics"][metric_name]["days"][item["date"]] = {
                                key: value
                                for key, value in output.items()
                                if key != "path"
                            }
                            payload["metrics"][metric_name]["progress"]["computed"] += 1

                    if pending_publish is not None:
                        self._finish_month_publish(
                            pending_publish,
                            payload,
                            staging,
                        )
                        pending_publish = None

                    observed_peak = max(
                        (
                            int(item["peak_rss_bytes"])
                            for item in day_results
                            if item.get("peak_rss_bytes") is not None
                        ),
                        default=None,
                    )
                    history_item = {
                        "month": month,
                        "date_workers": min(effective_workers, len(month_dates)),
                        "observed_peak_rss_bytes": observed_peak,
                    }
                    if (
                        self.config.date_workers == 0
                        and self.config.adaptive_workers
                        and observed_peak is not None
                        and month_index + 1 < len(month_groups)
                    ):
                        _, next_dates = month_groups[month_index + 1]
                        adaptive = recommend_adaptive_date_workers(
                            self.group,
                            self.config,
                            month_dates,
                            next_dates,
                            observed_peak,
                        )
                        effective_workers = adaptive.workers
                        history_item["adaptive_recommendation"] = adaptive.as_dict()
                    payload["parallelism"]["history"].append(history_item)

                    arguments = self._month_publish_arguments(
                        month,
                        month_dates,
                        pending_by_metric,
                        staging,
                    )
                    handle = (
                        _GroupMonthPublishProcess(arguments)
                        if self.config.publish_while_computing
                        else _ImmediateGroupResult(_publish_group_month(arguments))
                    )
                    pending_publish = (month, arguments, handle)
                    if not self.config.publish_while_computing:
                        self._finish_month_publish(
                            pending_publish,
                            payload,
                            staging,
                        )
                        pending_publish = None

                if pending_publish is not None:
                    self._finish_month_publish(
                        pending_publish,
                        payload,
                        staging,
                    )
                    pending_publish = None
            finally:
                if pending_publish is not None and isinstance(
                    pending_publish[2], _GroupMonthPublishProcess
                ):
                    pending_publish[2].cancel()

            payload["status"] = "complete"
            payload["parallelism"]["final_effective_date_workers"] = effective_workers
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            payload["finished_at"] = datetime.now().astimezone().isoformat()
            self._write_metric_manifests(payload)
            return payload
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = repr(exc)
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            payload["finished_at"] = datetime.now().astimezone().isoformat()
            self._write_metric_manifests(payload)
            raise
        finally:
            if payload["status"] == "complete":
                shutil.rmtree(staging)

    def _run_days(
        self,
        dates: list[date],
        pending_by_date: Mapping[date, tuple[str, ...]],
        staging: Path,
        *,
        date_workers: int,
    ) -> list[dict]:
        if date_workers == 1 or len(dates) == 1:
            _configure_worker(self.config.arrow_threads_per_worker)
            return [
                _process_one_group_day(
                    self.group,
                    value,
                    pending_by_date[value],
                    self.config,
                    staging,
                )
                for value in dates
            ]

        results = []
        with ProcessPoolExecutor(
            max_workers=min(date_workers, len(dates)),
            initializer=_configure_worker,
            initargs=(self.config.arrow_threads_per_worker,),
            mp_context=_worker_process_context(),
        ) as executor:
            futures = {
                executor.submit(
                    _process_one_group_day,
                    self.group,
                    value,
                    pending_by_date[value],
                    self.config,
                    staging,
                ): value
                for value in dates
            }
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda item: item["date"])

    def _month_publish_arguments(
        self,
        month: str,
        month_dates: list[date],
        pending_by_metric: Mapping[str, list[date]],
        staging: Path,
    ) -> tuple[tuple[str, tuple], ...]:
        result = []
        month_date_set = set(month_dates)
        for metric_name in self.group.metric_names:
            dates = sorted(month_date_set & set(pending_by_metric[metric_name]))
            if not dates:
                continue
            metric = self.group.metric(metric_name)
            paths = [
                staging / month / metric_name / f"{value:%Y%m%d}.parquet"
                for value in dates
            ]
            result.append(
                (
                    metric_name,
                    (
                        paths,
                        month_path(self.output_roots[metric_name], metric_name, month),
                        metric_name,
                        metric.value_type,
                        self.config,
                        set(dates),
                    ),
                )
            )
        return tuple(result)

    def _finish_month_publish(
        self,
        pending_publish,
        payload: dict,
        staging: Path,
    ) -> None:
        month, arguments, handle = pending_publish
        published = handle.result()
        for metric_name, publish_arguments in arguments:
            date_count = len(publish_arguments[-1])
            payload["metrics"][metric_name]["months"][month] = published[metric_name]
            payload["metrics"][metric_name]["progress"]["completed"] += date_count
            payload["progress"]["completed_outputs"] += date_count
        payload["last_checkpoint_at"] = datetime.now().astimezone().isoformat()
        self._write_metric_manifests(payload)
        shutil.rmtree(staging / month)

    def _find_existing_dates(self, dates: list[date]) -> dict[str, set[date]]:
        grouped: dict[str, set[date]] = defaultdict(set)
        for value in dates:
            grouped[value.strftime("%Y%m")].add(value)
        result = {name: set() for name in self.group.metric_names}
        for metric_name in self.group.metric_names:
            for month, requested in grouped.items():
                result[metric_name].update(
                    published_dates(
                        month_path(self.output_roots[metric_name], metric_name, month)
                    )
                    & requested
                )
        return result

    def _validate_input_files(self, dates: list[date]) -> None:
        missing = [
            (value.isoformat(), dataset)
            for value in dates
            for dataset in self.group.inputs
            if not input_path(self.config.input_root, dataset, value).is_file()
        ]
        if missing:
            examples = [f"{value}:{dataset}" for value, dataset in missing[:20]]
            raise FileNotFoundError(
                f"指标组运行日期缺少逐笔输入文件，缺失示例：{examples}"
            )

    def _initial_payload(
        self,
        run_id: str,
        requested_dates: list[date],
        pending_by_metric: Mapping[str, list[date]],
        existing_by_metric: Mapping[str, set[date]],
        effective_workers: int,
        recommendation,
    ) -> dict:
        metrics = {}
        for metric_name in self.group.metric_names:
            pending = pending_by_metric[metric_name]
            metrics[metric_name] = {
                "output_root": str(self.output_roots[metric_name]),
                "requested_dates": [value.isoformat() for value in requested_dates],
                "pending_dates": [value.isoformat() for value in pending],
                "skipped_dates": [
                    value.isoformat()
                    for value in requested_dates
                    if value in existing_by_metric[metric_name] and self.config.resume
                ],
                "progress": {
                    "requested": len(requested_dates),
                    "pending": len(pending),
                    "skipped": len(requested_dates) - len(pending),
                    "computed": 0,
                    "completed": 0,
                },
                "days": {},
                "months": {},
            }
        return {
            "status": "running",
            "framework_version": __version__,
            "metric_group": self.group.name,
            "run_id": run_id,
            "started_at": datetime.now().astimezone().isoformat(),
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
                "requested_dates": len(requested_dates),
                "pending_outputs": sum(len(values) for values in pending_by_metric.values()),
                "computed_dates": 0,
                "completed_outputs": 0,
            },
            "metrics": metrics,
        }

    def _write_metric_manifests(self, payload: dict) -> None:
        """把组运行状态投影为每个最终指标自己的manifest。"""

        for metric_name in self.group.metric_names:
            metric_payload = payload["metrics"][metric_name]
            manifest = {
                "status": payload["status"],
                "framework_version": payload["framework_version"],
                "metric": metric_name,
                "metric_group": self.group.name,
                "run_id": payload["run_id"],
                "started_at": payload["started_at"],
                "requested_dates": metric_payload["requested_dates"],
                "pending_dates": metric_payload["pending_dates"],
                "skipped_dates": metric_payload["skipped_dates"],
                "config": {
                    **payload["config"],
                    "output_root": str(self.output_roots[metric_name]),
                },
                "parallelism": copy.deepcopy(payload["parallelism"]),
                "progress": copy.deepcopy(metric_payload["progress"]),
                "group_progress": copy.deepcopy(payload["progress"]),
                "days": copy.deepcopy(metric_payload["days"]),
                "months": copy.deepcopy(metric_payload["months"]),
            }
            for key in (
                "last_checkpoint_at",
                "elapsed_seconds",
                "finished_at",
                "error",
            ):
                if key in payload:
                    manifest[key] = payload[key]
            write_manifest(
                metric_root(self.output_roots[metric_name], metric_name),
                manifest,
            )
