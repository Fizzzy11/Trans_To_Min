"""1.3.0源文件单次读取：共享暂存、生成期独立参考检查点与框架内逐值比对。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pyarrow.parquet as pq

from trans_to_min import DayReference, MinuteMetricGroupRunner, RunConfig, validate_published
from trans_to_min import source_stage
from trans_to_min.adapter import L2InputAdapter
from trans_to_min.reference_check import CHECKPOINT_DIRECTORY, checkpoint_path
from trans_to_min.storage import month_path

from source_sharing_fixture import (
    build_group, build_nullable_group, day_reference, masked_reference, write_input, wrong_reference,
)


DAY = date(2025, 1, 2)
NEXT_DAY = date(2025, 1, 3)
POSIX = source_stage.fcntl is not None


def staged_files(root: Path) -> list[str]:
    return sorted(item.name for item in root.iterdir() if item.name.endswith((".src", ".tmp")))


@unittest.skipUnless(POSIX, "共享暂存依赖POSIX文件锁")
class SourceStageTest(unittest.TestCase):
    def test_same_file_is_staged_once_and_removed_on_last_release(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_input(root / "input", DAY)
            stage = root / "stage"
            first = source_stage.acquire(path, root=stage)
            second = source_stage.acquire(path, root=stage)
            self.assertEqual((first.mode, first.origin), ("staged", "read"))
            self.assertEqual((second.mode, second.origin), ("staged", "hit"))
            self.assertEqual(len(staged_files(stage)), 1)
            self.assertEqual(
                pq.ParquetFile(first.open()).read(), pq.ParquetFile(second.open()).read()
            )
            self.assertEqual(pq.ParquetFile(first.open()).read(), pq.read_table(path))
            first.close()
            self.assertEqual(len(staged_files(stage)), 1)
            self.assertEqual(pq.ParquetFile(second.open()).read().num_rows, 3)
            second.close()
            self.assertEqual(staged_files(stage), [])
            with self.assertRaisesRegex(ValueError, "已关闭"):
                second.open()

    def test_retained_copy_survives_release_until_cleared(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_input(root / "input", DAY)
            stage = root / "stage"
            source_stage.acquire(path, root=stage, retain=True).close()
            self.assertEqual(len(staged_files(stage)), 1)
            again = source_stage.acquire(path, root=stage, retain=True)
            self.assertEqual(again.origin, "hit")
            held = source_stage.clear(stage)
            self.assertEqual((held["removed"], held["remaining"]), (0, 1))
            again.close()
            cleared = source_stage.clear(stage)
            self.assertEqual((cleared["removed"], cleared["remaining"]), (1, 0))
            self.assertFalse(stage.exists())

    def test_budget_evicts_unheld_then_falls_back_to_private_buffer(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = write_input(root / "input", DAY)
            second_path = write_input(root / "input", NEXT_DAY)
            stage = root / "stage"
            budget = first_path.stat().st_size + 16
            source_stage.acquire(first_path, root=stage, retain=True, budget_bytes=budget).close()
            # 预算只够一份：无人持有的旧副本被淘汰。
            second = source_stage.acquire(second_path, root=stage, retain=True, budget_bytes=budget)
            self.assertEqual(second.mode, "staged")
            self.assertEqual(len(staged_files(stage)), 1)
            # 旧副本仍被持有时不能淘汰：退回进程私有缓冲，预读则放弃。
            first = source_stage.acquire(first_path, root=stage, retain=True, budget_bytes=budget)
            self.assertEqual(first.mode, "buffer")
            self.assertIsNone(source_stage.acquire(
                first_path, root=stage, retain=True, budget_bytes=budget, staged_only=True))
            self.assertEqual(pq.ParquetFile(first.open()).read(), pq.read_table(first_path))
            first.close()
            second.close()

    def test_off_mode_reads_directly_and_changed_source_is_detected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_input(root / "input", DAY)
            direct = source_stage.acquire(path, mode="off")
            self.assertEqual((direct.mode, direct.open()), ("direct", str(path)))
            handle = source_stage.acquire(path, root=root / "stage")
            handle.verify_current()
            write_input(root / "input", DAY, extra_volume=1)
            os.utime(path, ns=(1, 1))
            with self.assertRaisesRegex(RuntimeError, "发生变化"):
                handle.verify_current()
            handle.close()

    def test_descriptor_check_skips_ctime_where_stat_and_fstat_disagree(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_input(root / "input", DAY)
            # 模拟Windows：按路径取得的ctime与按句柄取得的不同，其余字段一致。
            identity = dict(source_stage.file_identity(path), ctime_ns=1)
            with self.assertRaisesRegex(RuntimeError, "发生变化"):
                source_stage._read_private(path, identity, 0.0)
            with mock.patch.object(source_stage, "_FSTAT_CTIME_COMPARABLE", False):
                handle = source_stage._read_private(path, identity, 0.0)
                self.assertEqual(pq.ParquetFile(handle.open()).read(), pq.read_table(path))
                with self.assertRaisesRegex(RuntimeError, "发生变化"):
                    source_stage._read_private(
                        path, dict(identity, mtime_ns=identity["mtime_ns"] + 1), 0.0)

    def test_prestager_holds_copies_until_day_is_done(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [write_input(root / "input", DAY), write_input(root / "input", NEXT_DAY)]
            stage = root / "stage"
            options = {"mode": "auto", "root": str(stage), "budget_bytes": 1 << 30, "retain": False}
            with source_stage.Prestager(
                [(DAY, [paths[0]]), (NEXT_DAY, [paths[1]])], options, window=4
            ) as prestager:
                prestager._thread.join(timeout=30)
                self.assertEqual(len(staged_files(stage)), 2)
                self.assertEqual(source_stage.acquire(paths[0], root=stage).origin, "hit")
                prestager.done(DAY)
                self.assertEqual(len(staged_files(stage)), 1)
            self.assertEqual(staged_files(stage), [])

    def test_adapter_decodes_identically_from_handle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_input(root / "input", DAY)
            adapter = L2InputAdapter(root / "input")
            columns = ("symbol", "time_ms", "volume")
            expected = adapter.read("weituo", DAY, columns)
            with source_stage.acquire(path, root=root / "stage") as handle:
                actual = adapter.read("weituo", DAY, columns, source=handle)
            self.assertTrue(expected.equals(actual))


class ReferenceSharingTest(unittest.TestCase):
    def _run(self, root: Path, days, reference, group_builder=build_group, **overrides):
        for day in days:
            write_input(root / "input", day)
        config = RunConfig(
            input_root=root / "input", output_root=root / "output",
            date_workers=overrides.pop("date_workers", 1),
            source_stage_root=str(root / "stage"), **overrides,
        )
        group = group_builder()
        roots = {name: root / "output" for name in group.metric_names}
        result = MinuteMetricGroupRunner(group, config, roots, reference=reference).run(list(days))
        return group, config, roots, result

    def test_generation_writes_checkpoint_and_validation_uses_it(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = DayReference(day_reference, {"version": "1"})
            group, config, roots, result = self._run(root, [DAY, NEXT_DAY], reference, date_workers=2)
            self.assertEqual(result["status"], "complete")
            day = result["day_sources"][DAY.isoformat()]
            self.assertEqual(day["reference"]["origin"], "generation")
            self.assertEqual(day["sources"]["weituo"]["mode"], "staged" if POSIX else "buffer")
            self.assertTrue(checkpoint_path(config, group.name, DAY).is_file())
            if POSIX:
                self.assertEqual(staged_files(root / "stage"), [])
            report = validate_published(group, config, roots, [DAY, NEXT_DAY], reference)
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertEqual(report["reference_origins"], {"generation": 2})
            self.assertEqual(len(report["coverage"]), 4)
            self.assertTrue(all(item["expected_rows"] == 2 * 242 for item in report["coverage"]))
            self.assertEqual(report["checkpoints_removed"], 2)
            self.assertFalse((root / "output" / CHECKPOINT_DIRECTORY).exists())

    def test_missing_or_stale_checkpoint_falls_back_to_rescan(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = DayReference(day_reference, {"version": "1"})
            group, config, roots, _ = self._run(root, [DAY], reference)
            checkpoint_path(config, group.name, DAY).unlink()
            report = validate_published(group, config, roots, [DAY], reference, clear_on_pass=False)
            self.assertEqual(report["status"], "passed", report["errors"])
            self.assertEqual(report["reference_origins"], {"validation_rescan": 1})
            # 参考版本变化使旧检查点失效，同样重扫。
            newer = replace(reference, identity={"version": "2"})
            report = validate_published(group, config, roots, [DAY], newer)
            self.assertEqual(report["reference_origins"], {"validation_rescan": 1})
            self.assertEqual(report["status"], "passed", report["errors"])

    def test_price_policy_gate_applies_to_validation(self) -> None:
        """来源报价版本不符合指标声明时，即使数值一致，独立校验也必须失败。"""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = DayReference(day_reference)
            group, config, roots, _ = self._run(root, [DAY], reference)
            for metric in group.metrics:
                metric.required_price_policy_version = 3
            report = validate_published(group, config, roots, [DAY], reference, clear_on_pass=False)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("要求报价版本3" in error for error in report["errors"]),
                            report["errors"])

    def test_wrong_values_fail_and_keep_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = DayReference(wrong_reference)
            group, config, roots, _ = self._run(root, [DAY], reference, source_stage="off")
            report = validate_published(group, config, roots, [DAY], reference)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("share_volume_min" in error for error in report["errors"]),
                            report["errors"])
            count = [item for item in report["coverage"] if item["metric"] == "share_count_min"]
            self.assertEqual(count[0]["different_values"], 0)
            self.assertTrue(checkpoint_path(config, group.name, DAY).is_file())

    def test_masked_reference_keeps_expected_nulls(self) -> None:
        empty_minutes = 2 * 242 - 2
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            masked = DayReference(masked_reference)
            group, config, roots, _ = self._run(root, [DAY], masked, build_nullable_group)
            report = validate_published(group, config, roots, [DAY], masked, clear_on_pass=False)
            self.assertEqual(report["status"], "passed", report["errors"])
            # 参考把空分钟算成0而输出为NULL：必须判为不一致。
            report = validate_published(group, config, roots, [DAY], DayReference(day_reference))
            self.assertEqual(report["status"], "failed")
            volume = [item for item in report["coverage"] if item["metric"] == "share_volume_min"]
            self.assertEqual(volume[0]["null_mismatches"], empty_minutes)
        with TemporaryDirectory() as temporary:
            # 反向：参考声明空分钟为NULL而输出填0，同样不一致。
            root = Path(temporary)
            masked = DayReference(masked_reference)
            group, config, roots, _ = self._run(root, [DAY], masked)
            report = validate_published(group, config, roots, [DAY], masked)
            self.assertEqual(report["status"], "failed")
            volume = [item for item in report["coverage"] if item["metric"] == "share_volume_min"]
            self.assertEqual(volume[0]["null_mismatches"], empty_minutes)

    def test_empty_metric_selection_fails_and_keeps_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = DayReference(day_reference)
            group, config, roots, _ = self._run(root, [DAY], reference)
            report = validate_published(group, config, roots, [DAY], reference, metric_names=[])
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("至少一个指标" in error for error in report["errors"]), report["errors"])
            self.assertTrue(checkpoint_path(config, group.name, DAY).is_file())

    def test_legacy_footer_validates_and_accepts_new_days(self) -> None:
        """footer遗留旧凭证的月文件仍可校验、续跑并追加新日期；凭证不读不比，重写后消失。"""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = DayReference(day_reference)
            group, config, roots, _ = self._run(root, [DAY], reference)
            for name in group.metric_names:
                path = month_path(roots[name], name, "202501")
                table = pq.read_table(path)
                metadata = dict(table.schema.metadata)
                metadata[b"source_contracts"] = b'{"2025-01-02": "' + b"0" * 64 + b'"}'
                pq.write_table(table.replace_schema_metadata(metadata), path)
            report = validate_published(group, config, roots, [DAY], reference, clear_on_pass=False)
            self.assertEqual(report["status"], "passed", report["errors"])
            write_input(root / "input", NEXT_DAY)
            result = MinuteMetricGroupRunner(
                group, replace(config, resume=True), roots, reference=reference
            ).run([DAY, NEXT_DAY])
            self.assertEqual(result["status"], "complete")
            for name in group.metric_names:
                self.assertEqual(result["metrics"][name]["progress"]["skipped"], 1)
                path = month_path(roots[name], name, "202501")
                metadata = pq.read_schema(path).metadata or {}
                self.assertNotIn(b"source_contracts", metadata)
                self.assertNotIn(b"source_contract_current_dates", metadata)
            report = validate_published(group, config, roots, [DAY, NEXT_DAY], reference)
            self.assertEqual(report["status"], "passed", report["errors"])

    def test_reference_must_be_module_level_function(self) -> None:
        with self.assertRaisesRegex(ValueError, "模块级函数"):
            DayReference(lambda day, sources: None)
        with self.assertRaisesRegex(ValueError, "source_stage"):
            RunConfig(source_stage="shared")


if __name__ == "__main__":
    unittest.main()
