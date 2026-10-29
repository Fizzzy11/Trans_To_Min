"""用真实分钟输出验证业务任务入口的授权范围、门禁和失败恢复。"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trans_to_min import input_path
from trans_to_min.metrics import MinuteTradeVolume
from trans_to_min.storage import month_path
from trans_to_min.task_entry import InputsNotReady, run_task_cli


def _write_trade(root: Path, day: date, multiplier: int = 1) -> None:
    """放置分钟边界已手算的成交，不用生产公式反推期望。"""

    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("date", pa.date32(), nullable=False),
            pa.field("time", pa.uint32(), nullable=False),
            pa.field("volume", pa.uint32(), nullable=False),
        ],
        metadata={b"format_version": b"1"},
    )
    table = pa.table(
        {
            "symbol": ["000001.XSHE"] * 5,
            "date": [day] * 5,
            "time": [34_200_000, 34_200_001, 41_400_000, 46_800_000, 54_000_000],
            "volume": [value * multiplier for value in (1, 2, 3, 4, 5)],
        },
        schema=schema,
    )
    target = input_path(root, "chengjiao", day)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)


class _Hooks:
    """代表业务显式提供的来源检查和独立校验边界。"""

    def __init__(self) -> None:
        self.ready_calls = []
        self.validation_calls = []

    def check_ready(self, metric, config, dates) -> None:
        self.ready_calls.append((metric, config, list(dates)))

    def validate_output(self, metric, config, dates) -> dict:
        self.validation_calls.append((metric, config, list(dates)))
        return {
            "status": "passed",
            "dates": [day.isoformat() for day in dates],
            "level": "独立分钟公式",
        }


class TaskEntryTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_root = self.root / "input"
        self.output_root = self.root / "output"
        self.metric = MinuteTradeVolume()
        self.hooks = _Hooks()
        self.report_index = 0

    def _arguments(self, mode, dates, *, output_root=None, execute=True):
        self.report_index += 1
        result = self.root / f"结果_{self.report_index}.json"
        dates_file = self.root / f"日期_{self.report_index}.json"
        dates_file.write_text(
            json.dumps([day.isoformat() for day in dates]), encoding="utf-8"
        )
        arguments = [
            "--mode", mode,
            "--dates-file", str(dates_file),
            "--input-root", str(self.input_root),
            "--output-root", str(output_root or self.output_root),
            "--result-json", str(result),
            "--date-workers", "1",
            "--arrow-threads-per-worker", "1",
            "--auto-memory-fraction", "0.25",
        ]
        if execute:
            arguments.append("--execute")
        return arguments, result

    def _invoke(self, mode, dates, **kwargs):
        arguments, report_path = self._arguments(mode, dates, **kwargs)
        code = run_task_cli(
            self.metric,
            check_ready=self.hooks.check_ready,
            validate_output=self.hooks.validate_output,
            argv=arguments,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["protocol_version"], 1)
        self.assertTrue(report["invocation_id"])
        self.assertEqual(report["mode"], mode)
        self.assertEqual(report["metric"], self.metric.name)
        self.assertEqual(report["requested_dates"], [day.isoformat() for day in dates])
        self.assertEqual(report["exit_code"], code)
        for key in ("generation", "validation", "error", "status"):
            self.assertIn(key, report)
        return code, report

    def _assert_day(self, frame, day, multiplier=1):
        day_frame = frame.loc[frame["trade_date"] == day].reset_index(drop=True)
        self.assertEqual(len(day_frame), 242)
        self.assertEqual(day_frame["symbol"].unique().tolist(), ["000001.XSHE"])
        morning = pd.date_range(f"{day} 09:30", f"{day} 11:30", freq="min", tz="Asia/Shanghai")
        afternoon = pd.date_range(f"{day} 13:00", f"{day} 15:00", freq="min", tz="Asia/Shanghai")
        self.assertEqual(day_frame["minute"].tolist(), morning.append(afternoon).tolist())
        expected = [0] * 242
        for index, value in ((0, 1), (1, 2), (120, 3), (121, 4), (241, 5)):
            expected[index] = value * multiplier
        self.assertEqual(day_frame[self.metric.name].tolist(), expected)

    def test_full_and_daily_match_independent_expected_across_months(self):
        dates = [date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 3)]
        for index, day in enumerate(dates, 1):
            _write_trade(self.input_root, day, index)
        full_root = self.root / "full"
        daily_root = self.root / "daily"
        code, report = self._invoke("full", dates, output_root=full_root)
        self.assertEqual((code, report["status"]), (0, "complete"))
        self.assertEqual(report["generation"]["progress"]["completed"], 3)
        self.assertEqual(report["validation"]["status"], "passed")
        self.assertIsNone(report["error"])
        for day in dates:
            self.assertEqual(self._invoke("daily", [day], output_root=daily_root)[0], 0)
        for month in ("202501", "202502"):
            full = pd.read_parquet(month_path(full_root, self.metric.name, month))
            daily = pd.read_parquet(month_path(daily_root, self.metric.name, month))
            pd.testing.assert_frame_equal(full, daily)
            for index, day in enumerate(dates, 1):
                if day.strftime("%Y%m") == month:
                    self._assert_day(full, day, index)

    def test_daily_fills_month_hole_without_dropping_existing_dates(self):
        dates = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)]
        for day in dates:
            _write_trade(self.input_root, day)
        self.assertEqual(self._invoke("full", [dates[0], dates[2]])[0], 0)
        self.assertEqual(self._invoke("daily", [dates[1]])[0], 0)
        frame = pd.read_parquet(month_path(self.output_root, self.metric.name, "202501"))
        self.assertEqual(frame["trade_date"].drop_duplicates().tolist(), dates)
        for day in dates:
            self._assert_day(frame, day)

    def test_repeated_execution_skips_generation_but_revalidates_every_date(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        self.assertEqual(self._invoke("daily", [day])[0], 0)
        path = month_path(self.output_root, self.metric.name, "202501")
        original = path.read_bytes()
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (0, "complete"))
        self.assertEqual(report["generation"]["skipped_dates"], [day.isoformat()])
        self.assertEqual(report["generation"]["progress"]["completed"], 0)
        self.assertEqual([call[2] for call in self.hooks.validation_calls], [[day], [day]])
        self.assertEqual(path.read_bytes(), original)

    def test_resume_rejects_changed_source_without_overwriting_output(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        self.assertEqual(self._invoke("full", [day])[0], 0)
        path = month_path(self.output_root, self.metric.name, "202501")
        original = path.read_bytes()
        _write_trade(self.input_root, day, 100)
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (1, "failed"))
        self.assertTrue(report["error"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(self.hooks.validation_calls), 1)

    def test_validation_failure_can_be_retried_after_successful_generation(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        def reject(metric, config, dates):
            raise ValueError("独立公式不一致")
        self.hooks.validate_output = reject
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (1, "failed"))
        self.assertEqual(report["generation"]["status"], "complete")
        path = month_path(self.output_root, self.metric.name, "202501")
        original = path.read_bytes()
        self.hooks = _Hooks()
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (0, "complete"))
        self.assertEqual(report["generation"]["skipped_dates"], [day.isoformat()])
        self.assertEqual(self.hooks.validation_calls[0][2], [day])
        self.assertEqual(path.read_bytes(), original)

    def test_malformed_or_incomplete_validation_cannot_report_complete(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        invalid = [
            None,
            {"status": "failed", "dates": ["2025-01-02"], "level": "独立公式"},
            {"status": "passed", "dates": [], "level": "独立公式"},
            {"status": "passed", "dates": ["2025-01-02", "2025-01-02"], "level": "独立公式"},
            {"status": "passed", "dates": ["2025-01-03"], "level": "独立公式"},
            {"status": "passed", "dates": ["2025-01-02"], "level": ""},
            {"status": "passed", "dates": ["2025-01-02"], "level": 1},
        ]
        for value in invalid:
            with self.subTest(validation=value):
                self.hooks.validate_output = lambda metric, config, dates, value=value: value
                code, report = self._invoke("daily", [day])
                self.assertEqual((code, report["status"]), (1, "failed"))
                self.assertTrue(report["error"])

    def test_not_ready_does_not_generate_or_validate(self):
        day = date(2025, 1, 2)
        def unavailable(metric, config, dates):
            raise InputsNotReady("尚未提交来源检查点")
        self.hooks.check_ready = unavailable
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (3, "not_ready"))
        self.assertIsNone(report["generation"])
        self.assertIsNone(report["validation"])
        self.assertEqual(self.hooks.validation_calls, [])
        self.assertEqual(list(self.output_root.rglob("*.parquet")), [])
        self.assertFalse((self.output_root / self.metric.name / "manifest.json").exists())
        self.assertFalse((self.output_root / self.metric.name / ".task.lock").exists())

    def test_existing_lock_is_preserved_and_never_stolen(self):
        lock = self.output_root / self.metric.name / ".task.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("另一个任务持有，不能按时间猜测过期", encoding="utf-8")
        original = lock.read_bytes()
        code, report = self._invoke("daily", [date(2025, 1, 2)])
        self.assertEqual((code, report["status"]), (4, "locked"))
        self.assertEqual(lock.read_bytes(), original)
        self.assertIsNone(report["generation"])
        self.assertEqual(list(self.output_root.rglob("*.parquet")), [])

    def test_default_dry_run_never_reads_source_or_calls_hooks(self):
        code, report = self._invoke("full", [date(2025, 1, 2)], execute=False)
        self.assertEqual((code, report["status"]), (0, "planned"))
        self.assertIsNone(report["generation"])
        self.assertIsNone(report["validation"])
        self.assertEqual(self.hooks.ready_calls, [])
        self.assertEqual(self.hooks.validation_calls, [])
        self.assertFalse(self.input_root.exists())
        self.assertFalse(self.output_root.exists())

    def test_empty_daily_is_no_work_without_source_or_hooks(self):
        code, report = self._invoke("daily", [])
        self.assertEqual((code, report["status"]), (0, "no_work"))
        self.assertIsNone(report["generation"])
        self.assertIsNone(report["validation"])
        self.assertEqual(self.hooks.ready_calls, [])
        self.assertEqual(self.hooks.validation_calls, [])
        self.assertFalse(self.input_root.exists())
        self.assertFalse(self.output_root.exists())

    def test_full_and_repair_reject_empty_authorization(self):
        for mode in ("full", "repair"):
            with self.subTest(mode=mode):
                arguments, _ = self._arguments(mode, [])
                code = run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                    validate_output=self.hooks.validate_output, argv=arguments)
                self.assertEqual(code, 2)
        self.assertEqual(self.hooks.ready_calls, [])
        self.assertFalse(self.output_root.exists())

    def test_explicit_repair_replaces_only_authorized_date(self):
        first, second = date(2025, 1, 2), date(2025, 1, 3)
        for day in (first, second):
            _write_trade(self.input_root, day)
        self.assertEqual(self._invoke("full", [first, second])[0], 0)
        _write_trade(self.input_root, first, 10)
        code, report = self._invoke("repair", [first])
        self.assertEqual((code, report["status"]), (0, "complete"))
        self.assertTrue(report["generation"]["config"]["overwrite_dates"])
        self.assertFalse(report["generation"]["config"]["resume"])
        frame = pd.read_parquet(month_path(self.output_root, self.metric.name, "202501"))
        self._assert_day(frame, first, 10)
        self._assert_day(frame, second)
        self.assertEqual(self.hooks.validation_calls[-1][2], [first])

    def test_report_path_inside_input_or_output_is_rejected(self):
        for report_root in (self.input_root, self.output_root):
            with self.subTest(report_root=report_root):
                arguments, _ = self._arguments("daily", [date(2025, 1, 2)])
                arguments[arguments.index("--result-json") + 1] = str(report_root / "报告.json")
                code = run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                    validate_output=self.hooks.validate_output, argv=arguments)
                self.assertEqual(code, 2)
                self.assertFalse(report_root.exists())

    def test_existing_report_is_not_overwritten_even_on_argument_error(self):
        arguments, result = self._arguments("daily", [date(2025, 1, 2)])
        result.write_text("已有审计记录，不得覆盖", encoding="utf-8")
        original = result.read_bytes()
        for invalid_mode in (False, True):
            with self.subTest(invalid_mode=invalid_mode):
                if invalid_mode:
                    arguments[arguments.index("--mode") + 1] = "unknown"
                code = run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                    validate_output=self.hooks.validate_output, argv=arguments)
                self.assertEqual(code, 2)
                self.assertEqual(result.read_bytes(), original)
        self.assertFalse(self.output_root.exists())

    def test_cli_requires_explicit_mode_and_one_date_selector(self):
        for absent in ("--mode", "--dates-file"):
            with self.subTest(absent=absent):
                arguments, _ = self._arguments("daily", [date(2025, 1, 2)])
                index = arguments.index(absent)
                del arguments[index:index + 2]
                code = run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                    validate_output=self.hooks.validate_output, argv=arguments)
                self.assertEqual(code, 2)
        self.assertFalse(self.input_root.exists())
        self.assertFalse(self.output_root.exists())

    def test_date_selectors_are_mutually_exclusive_and_strict_iso(self):
        for extra in (("--date", "2025-01-02"),):
            arguments, _ = self._arguments("daily", [date(2025, 1, 2)])
            arguments.extend(extra)
            self.assertEqual(run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                         validate_output=self.hooks.validate_output, argv=arguments), 2)
        for payload in (["20250102"], ["2025-02-30"], {"dates": ["2025-01-02"]}, [1]):
            with self.subTest(payload=payload):
                arguments, _ = self._arguments("daily", [])
                dates_file = Path(arguments[arguments.index("--dates-file") + 1])
                dates_file.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                             validate_output=self.hooks.validate_output, argv=arguments), 2)

    def test_single_date_and_resource_options_reach_real_execution(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        arguments, result = self._arguments("daily", [day])
        index = arguments.index("--dates-file")
        arguments[index:index + 2] = ["--date", day.isoformat()]
        self.assertEqual(run_task_cli(self.metric, check_ready=self.hooks.check_ready,
                                     validate_output=self.hooks.validate_output, argv=arguments), 0)
        report = json.loads(result.read_text(encoding="utf-8"))
        config = report["generation"]["config"]
        self.assertEqual(config["date_workers"], 1)
        self.assertEqual(config["arrow_threads_per_worker"], 1)
        self.assertEqual(config["auto_memory_fraction"], 0.25)
        self.assertTrue(config["resume"])
        self.assertFalse(config["overwrite_dates"])
        self.assertFalse((self.output_root / self.metric.name / ".task.lock").exists())

    def test_execute_requires_both_business_hooks(self):
        for ready, validate in ((None, None), (self.hooks.check_ready, None),
                                (None, self.hooks.validate_output)):
            with self.subTest(ready=ready, validate=validate):
                arguments, _ = self._arguments("daily", [date(2025, 1, 2)])
                self.assertEqual(run_task_cli(self.metric, check_ready=ready,
                                             validate_output=validate, argv=arguments), 2)
        self.assertFalse(self.output_root.exists())

    def test_false_readiness_result_fails_before_generation(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        self.hooks.check_ready = lambda metric, config, dates: False
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (1, "failed"))
        self.assertIsNone(report["generation"])
        self.assertIsNone(report["validation"])
        self.assertTrue(report["error"])
        self.assertEqual(list(self.output_root.rglob("*.parquet")), [])

    def test_readiness_hook_cannot_clear_authorized_generation_scope(self):
        dates = [date(2025, 1, 2), date(2025, 1, 3)]
        for day in dates:
            _write_trade(self.input_root, day)

        def clear_dates(metric, config, received_dates):
            received_dates.clear()

        self.hooks.check_ready = clear_dates
        code, report = self._invoke("daily", dates)
        self.assertEqual((code, report["status"]), (0, "complete"))
        self.assertEqual(report["generation"]["requested_dates"], [day.isoformat() for day in dates])
        self.assertEqual(report["generation"]["progress"]["completed"], 2)
        self.assertEqual(self.hooks.validation_calls[0][2], dates)
        frame = pd.read_parquet(month_path(self.output_root, self.metric.name, "202501"))
        for day in dates:
            self._assert_day(frame, day)

    def test_validation_hook_cannot_clear_scope_used_for_coverage_checks(self):
        dates = [date(2025, 1, 2), date(2025, 1, 3)]
        for day in dates:
            _write_trade(self.input_root, day)
        original_dates = [day.isoformat() for day in dates]

        def clear_dates_and_validate(metric, config, received_dates):
            received_dates.clear()
            return {"status": "passed", "dates": original_dates, "level": "独立分钟公式"}

        self.hooks.validate_output = clear_dates_and_validate
        code, report = self._invoke("daily", dates)
        self.assertEqual((code, report["status"]), (0, "complete"))
        self.assertEqual(report["generation"]["requested_dates"], original_dates)
        self.assertEqual(report["validation"]["dates"], original_dates)
        frame = pd.read_parquet(month_path(self.output_root, self.metric.name, "202501"))
        for day in dates:
            self._assert_day(frame, day)

    def test_validator_source_rewrite_invalidates_passed_result(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)

        def rewrite_source(metric, config, dates):
            _write_trade(config.input_root, day, 10)
            return {"status": "passed", "dates": [day.isoformat()], "level": "独立分钟公式"}

        self.hooks.validate_output = rewrite_source
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (1, "failed"))
        self.assertEqual(report["generation"]["status"], "complete")
        self.assertIsNone(report["validation"])
        self.assertTrue(report["error"])

    def test_validator_month_rewrite_invalidates_passed_result(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)

        def rewrite_month(metric, config, dates):
            path = month_path(config.output_root, metric.name, "202501")
            table = pq.read_table(path)
            values = table[metric.name].to_pylist()
            values[0] += 100
            # 保留原来源footer，避免仅靠来源契约误认输出未发生变化。
            changed = table.set_column(
                table.schema.get_field_index(metric.name),
                table.schema.field(metric.name),
                pa.array(values, type=pa.uint64()),
            )
            pq.write_table(changed, path)
            return {"status": "passed", "dates": [day.isoformat()], "level": "独立分钟公式"}

        self.hooks.validate_output = rewrite_month
        code, report = self._invoke("daily", [day])
        self.assertEqual((code, report["status"]), (1, "failed"))
        self.assertEqual(report["generation"]["status"], "complete")
        self.assertIsNone(report["validation"])
        self.assertTrue(report["error"])

    def test_business_prints_go_to_stderr_while_stdout_is_one_json(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        ready = self.hooks.check_ready
        validate = self.hooks.validate_output

        def noisy_ready(metric, config, dates):
            print("业务就绪检查日志")
            return ready(metric, config, dates)

        def noisy_validate(metric, config, dates):
            print("业务独立校验日志")
            return validate(metric, config, dates)

        self.hooks.check_ready = noisy_ready
        self.hooks.validate_output = noisy_validate
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code, report = self._invoke("daily", [day])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), report)
        self.assertIn("业务就绪检查日志", stderr.getvalue())
        self.assertIn("业务独立校验日志", stderr.getvalue())

    def _run_template(self, arguments):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_task_template.py"
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

    def test_template_subprocess_help_and_dry_run_work_from_other_directory(self):
        help_result = self._run_template(["--help"])
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--mode", help_result.stdout)
        arguments, path = self._arguments("full", [date(2025, 1, 2)], execute=False)
        result = self._run_template(arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report, json.loads(path.read_text(encoding="utf-8")))
        self.assertFalse(self.input_root.exists())
        self.assertFalse(self.output_root.exists())

    def test_template_subprocess_execute_is_not_ready_without_business_hooks(self):
        arguments, path = self._arguments("daily", [date(2025, 1, 2)])
        result = self._run_template(arguments)
        self.assertEqual(result.returncode, 3, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "not_ready")
        self.assertIsNone(report["generation"])
        self.assertIsNone(report["validation"])
        self.assertEqual(report, json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(list(self.output_root.rglob("*.parquet")), [])
        self.assertFalse((self.output_root / self.metric.name / "manifest.json").exists())

    def test_only_full_mode_keeps_overlapped_month_publication(self):
        day = date(2025, 1, 2)
        _write_trade(self.input_root, day)
        for mode, expected in (("full", True), ("daily", False), ("repair", False)):
            with self.subTest(mode=mode):
                code, report = self._invoke(mode, [day], output_root=self.root / mode)
                self.assertEqual((code, report["status"]), (0, "complete"))
                self.assertEqual(report["generation"]["config"]["publish_while_computing"], expected)

    def test_two_real_workers_can_print_without_corrupting_result_file(self):
        dates = [date(2025, 1, 2), date(2025, 1, 3)]
        for index, day in enumerate(dates, 1):
            _write_trade(self.input_root, day, index)
        arguments, result_path = self._arguments("daily", dates)
        arguments[arguments.index("--date-workers") + 1] = "2"
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "tests" / "task_entry_worker_fixture.py"
        result = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=self.root,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # 子进程标准输出允许混合业务日志，唯一机器接口是独立JSON文件。
        for day in dates:
            self.assertIn(f"工作进程计算日志：{day.isoformat()}", result.stdout)
        report = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual((report["status"], report["exit_code"]), ("complete", 0))
        expected_dates = [day.isoformat() for day in dates]
        self.assertEqual(report["requested_dates"], expected_dates)
        self.assertEqual(report["generation"]["requested_dates"], expected_dates)
        self.assertEqual(report["validation"]["dates"], expected_dates)
        self.assertEqual(report["generation"]["parallelism"]["effective_date_workers"], 2)
        self.assertEqual(
            Path(report["validation"]["framework_path"]),
            project_root / "trans_to_min" / "__init__.py",
        )
        frame = pd.read_parquet(month_path(self.output_root, self.metric.name, "202501"))
        for index, day in enumerate(dates, 1):
            self._assert_day(frame, day, index)
