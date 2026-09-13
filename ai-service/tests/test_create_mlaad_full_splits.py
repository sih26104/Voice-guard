"""Tests for :mod:`scripts.create_mlaad_full_splits` (src/scripts).

No network, no downloads, no model, no real dataset writes: everything runs
on a synthetic audio tree + manifest written into a temporary directory. The
module's path constants (``DATA_DIR`` / ``AUDIO_DIR`` / ``MANIFEST`` /
``SPLITS_CSV`` / ``REPORT_JSON`` / ``PROJECT_ROOT``) are redirected per test
so the real ``data/mlaad_full`` is never touched.

Covered:

* inventory + metadata derivation (strict layout, comma system names)
* manifest/disk cross-check (missing/extra/mismatched metadata all fail)
* split logic: determinism, seed sensitivity, every-file-exactly-once,
  both classes in the core splits, unseen-TTS holdout semantics
* validation: every check fires on a targeted corruption
* end-to-end ``main()``: writes splits.csv + split_report.json, no duplicate
  ``local_path``, report contents (incl. zero-overlap confirmation)
* ``--validate-only`` round-trip on the generated CSV
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import random
import sys
import tempfile
import unittest
import wave
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AI_SERVICE_ROOT / "src"

# The project's own scripts import ``from dataset...`` (ai-service/src on
# sys.path — same convention as running with PYTHONPATH=src). Put src on the
# path BEFORE importing through the ``src.scripts`` package so the package
# __init__ (which imports torch-dependent common.py) can resolve ``dataset``.
for _path in (str(SRC_ROOT), str(AI_SERVICE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

splits_mod = importlib.import_module("src.scripts.create_mlaad_full_splits")

SR = 16_000


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------


def write_wav(path: Path, seconds: float = 0.6) -> None:
    """Write a tiny mono 16 kHz WAV (valid RIFF, real frames)."""
    import math
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * SR)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        silence = b"\x00\x00" * frames
        handle.writeframes(silence)
    # (Silence is fine: split generation never decodes audio. Keep math/
    #  struct imports used for potential tone generation in future edits.)
    assert math and struct


def make_dataset(
    audio_dir: Path,
    systems_by_language: dict = None,
    bonafide_by_language: dict = None,
    per_system: int = 6,
    comma_named_system: str | None = None,
) -> list:
    """Create a fake/on-disk tree + the matching manifest rows.

    Returns the manifest rows (schema of download_mlaad_full's manifest.csv):
    ``local_path`` is repo-root-relative (``data/mlaad_full/audio/...``),
    exactly as the real manifest and ``scan_audio_files`` produce.
    """
    systems_by_language = systems_by_language or {
        "en": ("SysA", "SysB", "SysC", "SysD", "SysE", "SysF"),
        "de": ("SysG", "SysH", "SysI"),
    }
    bonafide_by_language = bonafide_by_language or {"en": 30, "de": 20}

    rows: list = []
    counter = 0
    for language, systems in systems_by_language.items():
        for system in systems:
            for index in range(per_system):
                counter += 1
                hf_path = f"fake/{language}/{system}/f{counter:05d}.wav"
                write_wav(audio_dir / hf_path)
                rows.append(
                    {
                        "local_path": f"data/mlaad_full/audio/{hf_path}",
                        "hf_path": hf_path,
                        "label": "SPOOF",
                        "language": language,
                        "tts_system": system,
                    }
                )
        for index in range(bonafide_by_language.get(language, 0)):
            counter += 1
            hf_path = f"original/{language}/o{counter:05d}.wav"
            write_wav(audio_dir / hf_path)
            rows.append(
                {
                    "local_path": f"data/mlaad_full/audio/{hf_path}",
                    "hf_path": hf_path,
                    "label": "BONAFIDE",
                    "language": language,
                    "tts_system": "",
                }
            )
    if comma_named_system:
        language = "en"
        for index in range(per_system):
            counter += 1
            hf_path = f"fake/{language}/{comma_named_system}/c{counter:05d}.wav"
            write_wav(audio_dir / hf_path)
            rows.append(
                {
                    "local_path": f"data/mlaad_full/audio/{hf_path}",
                    "hf_path": hf_path,
                    "label": "SPOOF",
                    "language": language,
                    "tts_system": comma_named_system,
                }
            )
    return rows


def write_manifest(rows: list, path: Path) -> Path:
    """Write manifest.csv exactly like download_mlaad_full.py does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["local_path", "hf_path", "label", "language", "tts_system"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


class SplitFixtureTest(unittest.TestCase):
    """Base class: temp dataset, redirected module paths, per-test cleanup."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

        self.repo_root = self.tmp / "repo"
        self.data_dir = self.repo_root / "data" / "mlaad_full"
        self.audio_dir = self.data_dir / "audio"
        self.audio_dir.mkdir(parents=True)

        self._originals = (
            splits_mod.PROJECT_ROOT,
            splits_mod.DATA_DIR,
            splits_mod.AUDIO_DIR,
            splits_mod.MANIFEST,
            splits_mod.SPLITS_CSV,
            splits_mod.REPORT_JSON,
        )
        splits_mod.PROJECT_ROOT = self.repo_root
        splits_mod.DATA_DIR = self.data_dir
        splits_mod.AUDIO_DIR = self.audio_dir
        splits_mod.MANIFEST = self.data_dir / "manifest.csv"
        splits_mod.SPLITS_CSV = self.data_dir / "splits.csv"
        splits_mod.REPORT_JSON = self.data_dir / "split_report.json"

    def tearDown(self) -> None:
        (
            splits_mod.PROJECT_ROOT,
            splits_mod.DATA_DIR,
            splits_mod.AUDIO_DIR,
            splits_mod.MANIFEST,
            splits_mod.SPLITS_CSV,
            splits_mod.REPORT_JSON,
        ) = self._originals

    # -- helpers -----------------------------------------------------------

    def build_standard_fixture(self, **kwargs) -> list:
        """Disk tree + manifest; returns manifest rows."""
        self.manifest_rows = make_dataset(self.audio_dir, **kwargs)
        write_manifest(self.manifest_rows, splits_mod.MANIFEST)
        return self.manifest_rows

    def generate(self, rows: list | None = None, seed: int | None = None):
        """Inventory + split + validate; returns (rows, disk_records)."""
        rng = random.Random(seed if seed is not None else splits_mod.SEED)
        disk = splits_mod.scan_audio_files(
            splits_mod.AUDIO_DIR, splits_mod.DATA_DIR, splits_mod.PROJECT_ROOT
        )
        assigned = splits_mod.build_splits(disk, rng)
        splits_mod.validate_or_raise(assigned, disk, splits_mod.PROJECT_ROOT)
        return assigned, disk


# ---------------------------------------------------------------------------
# Inventory & metadata derivation
# ---------------------------------------------------------------------------


class TestDeriveMetadata(unittest.TestCase):
    def test_fake_path_yields_spoof(self) -> None:
        self.assertEqual(
            splits_mod.derive_metadata("fake/de/CaroTTS/x.wav"),
            ("SPOOF", "de", "CaroTTS"),
        )

    def test_original_path_yields_bonafide(self) -> None:
        self.assertEqual(
            splits_mod.derive_metadata("original/en/x.wav"),
            ("BONAFIDE", "en", ""),
        )

    def test_comma_system_name_passes_through(self) -> None:
        name = "Resemble.ai (April 12th, 2025)"
        self.assertEqual(
            splits_mod.derive_metadata(f"fake/en/{name}/x.wav"),
            ("SPOOF", "en", name),
        )

    def test_unexpected_layouts_are_rejected(self) -> None:
        for path in (
            "fake/de/x.wav",          # too shallow (no system level)
            "fake/en/Sys/a/b.wav",    # too deep
            "original/en/a/b.wav",    # too deep
            "other/en/Sys/a.wav",     # unknown top level
        ):
            with self.assertRaises(splits_mod.SplitError, msg=path):
                splits_mod.derive_metadata(path)


class TestScanAudioFiles(SplitFixtureTest):
    def test_scans_all_wavs_and_builds_repo_relative_paths(self) -> None:
        self.build_standard_fixture()
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        self.assertEqual(len(disk), len(self.manifest_rows))
        for record in disk:
            self.assertTrue(record["local_path"].startswith("data/mlaad_full/audio/"))
            self.assertNotIn("\\", record["local_path"])

    def test_scan_is_sorted_and_deterministic(self) -> None:
        self.build_standard_fixture()
        first = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        second = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [r["hf_path"] for r in first],
            sorted(r["hf_path"] for r in first),
        )

    def test_empty_directories_contribute_nothing(self) -> None:
        self.build_standard_fixture(per_system=0, bonafide_by_language={"en": 12})
        (self.audio_dir / "fake" / "en" / "GhostSys").mkdir(parents=True)
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        self.assertEqual(len(disk), 12)
        self.assertTrue(all(r["label"] == "BONAFIDE" for r in disk))

    def test_missing_audio_dir_raises(self) -> None:
        self.audio_dir.rmdir()
        with self.assertRaises(splits_mod.SplitError):
            splits_mod.scan_audio_files(
                self.audio_dir, self.data_dir, self.repo_root
            )

    def test_no_wavs_raises(self) -> None:
        self.build_standard_fixture()
        for wav in self.audio_dir.rglob("*.wav"):
            wav.unlink()
        with self.assertRaises(splits_mod.SplitError):
            splits_mod.scan_audio_files(
                self.audio_dir, self.data_dir, self.repo_root
            )


class TestManifestCrossCheck(SplitFixtureTest):
    def test_matching_manifest_passes(self) -> None:
        rows = self.build_standard_fixture()
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        summary = splits_mod.cross_check_manifest(disk, rows)
        self.assertEqual(summary["matched_paths"], len(rows))
        self.assertEqual(summary["missing_in_manifest"], [])
        self.assertEqual(summary["missing_on_disk"], [])
        self.assertEqual(summary["metadata_mismatches"], [])

    def test_extra_disk_file_fails(self) -> None:
        rows = self.build_standard_fixture()
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        write_wav(self.audio_dir / "original/en/extra.wav")
        extra_disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        with self.assertRaises(splits_mod.SplitError) as context:
            splits_mod.cross_check_manifest(extra_disk, rows)
        self.assertIn("missing_in_manifest=1", str(context.exception))

    def test_stale_manifest_row_fails(self) -> None:
        rows = self.build_standard_fixture()
        stale = rows[:-1]  # manifest claims one fewer file than exists
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        with self.assertRaises(splits_mod.SplitError) as context:
            splits_mod.cross_check_manifest(disk, stale)
        self.assertIn("missing_in_manifest=1", str(context.exception))

    def test_metadata_mismatch_fails(self) -> None:
        rows = self.build_standard_fixture()
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        rows[0]["language"] = "fr"  # corrupt one row's language
        with self.assertRaises(splits_mod.SplitError) as context:
            splits_mod.cross_check_manifest(disk, rows)
        self.assertIn("metadata_mismatches=1", str(context.exception))


class TestManifestLoading(SplitFixtureTest):
    def test_missing_manifest_raises_filenotfound(self) -> None:
        with self.assertRaises(FileNotFoundError):
            splits_mod.load_manifest_rows(self.data_dir / "manifest.csv")

    def test_missing_column_raises(self) -> None:
        self.build_standard_fixture()
        # Write a manifest whose HEADER lacks the language column entirely
        # (DictWriter would silently emit empty values otherwise).
        bad_path = self.data_dir / "bad.csv"
        with bad_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["local_path", "hf_path", "label", "tts_system"],
            )
            writer.writeheader()
            for row in splits_mod.load_manifest_rows(splits_mod.MANIFEST):
                writer.writerow(
                    {k: v for k, v in row.items() if k != "language"}
                )
        with self.assertRaises(splits_mod.SplitError):
            splits_mod.load_manifest_rows(bad_path)

    def test_bad_label_raises(self) -> None:
        self.build_standard_fixture()
        rows = splits_mod.load_manifest_rows(splits_mod.MANIFEST)
        rows[0]["label"] = "SYNTHETIC"
        with self.assertRaises(splits_mod.SplitError):
            splits_mod.cross_check_manifest(
                splits_mod.scan_audio_files(
                    self.audio_dir, self.data_dir, self.repo_root
                ),
                rows,
            )


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


class TestSplitLogic(SplitFixtureTest):
    def test_every_file_assigned_exactly_once(self) -> None:
        rows = self.build_standard_fixture()
        assigned, disk = self.generate(rows)
        self.assertEqual(
            Counter(r["local_path"] for r in assigned),
            Counter(r["local_path"] for r in disk),
        )

    def test_no_duplicate_local_path_anywhere(self) -> None:
        rows = self.build_standard_fixture()
        assigned, _ = self.generate(rows)
        paths = [r["local_path"] for r in assigned]
        self.assertEqual(len(paths), len(set(paths)))

    def test_all_four_splits_present_with_both_classes(self) -> None:
        rows = self.build_standard_fixture()
        assigned, _ = self.generate(rows)
        by_split = {}
        for row in assigned:
            by_split.setdefault(row["split"], set()).add(row["label"])
        for split in ("train", "validation", "test", "test-unseen-tts"):
            self.assertIn(split, by_split, split)
            self.assertIn("SPOOF", by_split[split], split)
        for split in ("train", "validation", "test"):
            self.assertIn("BONAFIDE", by_split[split], split)

    def test_unseen_systems_never_occur_in_core_splits(self) -> None:
        rows = self.build_standard_fixture()
        assigned, _ = self.generate(rows)
        unseen = {
            r["tts_system"] for r in assigned
            if r["split"] == "test-unseen-tts" and r["label"] == "SPOOF"
        }
        self.assertTrue(unseen)
        for split in ("train", "validation", "test"):
            present = {
                r["tts_system"] for r in assigned
                if r["split"] == split and r["label"] == "SPOOF"
            }
            self.assertEqual(present & unseen, set(), split)

    def test_bonafide_support_is_disjoint_from_core_splits(self) -> None:
        """No bonafide file may appear in both test and test-unseen-tts."""
        rows = self.build_standard_fixture()
        assigned, _ = self.generate(rows)
        test_bonafide = {
            r["local_path"] for r in assigned
            if r["split"] == "test" and r["label"] == "BONAFIDE"
        }
        unseen_bonafide = {
            r["local_path"] for r in assigned
            if r["split"] == "test-unseen-tts" and r["label"] == "BONAFIDE"
        }
        self.assertTrue(unseen_bonafide)
        self.assertEqual(test_bonafide & unseen_bonafide, set())

    def test_deterministic_for_fixed_seed(self) -> None:
        rows = self.build_standard_fixture()
        first, _ = self.generate(rows)
        second, _ = self.generate(rows)
        self.assertEqual(
            sorted((r["local_path"], r["split"]) for r in first),
            sorted((r["local_path"], r["split"]) for r in second),
        )

    def test_different_seed_changes_assignment(self) -> None:
        rows = self.build_standard_fixture()
        first, _ = self.generate(rows, seed=1)
        second, _ = self.generate(rows, seed=2)
        self.assertNotEqual(
            sorted((r["local_path"], r["split"]) for r in first),
            sorted((r["local_path"], r["split"]) for r in second),
        )

    def test_holdout_covers_fraction_of_spoof_files(self) -> None:
        rows = self.build_standard_fixture()
        assigned, _ = self.generate(rows)
        spoof_by_split = Counter(
            r["split"] for r in assigned if r["label"] == "SPOOF"
        )
        held = spoof_by_split["test-unseen-tts"]
        total = sum(spoof_by_split.values())
        # Whole-system granularity: ~15% target with floors.
        self.assertGreaterEqual(held / total, 0.10)
        self.assertLessEqual(held / total, 0.35)

    def test_tiny_language_keeps_bonafide_in_core_splits(self) -> None:
        """A language below MIN_BONAFIDE_FOR_SUPPORT gets no bonafide support."""
        rows = self.build_standard_fixture(
            bonafide_by_language={"en": 30, "de": 5},
        )
        assigned, _ = self.generate(rows)
        unseen_bonafide_languages = {
            r["language"] for r in assigned
            if r["split"] == "test-unseen-tts" and r["label"] == "BONAFIDE"
        }
        self.assertNotIn("de", unseen_bonafide_languages)
        self.assertIn("en", unseen_bonafide_languages)


class TestHoldOutUnseenSystems(SplitFixtureTest):
    def test_systems_are_never_split(self) -> None:
        rows = self.build_standard_fixture(per_system=8)
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        spoof = [r for r in disk if r["label"] == "SPOOF"]
        unseen, seen = splits_mod.hold_out_unseen_systems(
            spoof, random.Random(splits_mod.SEED)
        )
        unseen_systems = {r["tts_system"] for r in unseen}
        seen_systems = {r["tts_system"] for r in seen}
        self.assertEqual(unseen_systems & seen_systems, set())
        # Every spoof file lands on one side.
        self.assertEqual(len(unseen) + len(seen), len(spoof))

    def test_spoof_row_without_system_rejected(self) -> None:
        with self.assertRaises(splits_mod.SplitError):
            splits_mod.hold_out_unseen_systems(
                [{"local_path": "x", "language": "en", "tts_system": ""}],
                random.Random(0),
            )


# ---------------------------------------------------------------------------
# Validation checks fire on corruption
# ---------------------------------------------------------------------------


class TestRunValidations(SplitFixtureTest):
    def setUp(self) -> None:
        super().setUp()
        self.rows = self.build_standard_fixture()
        self.assigned, self.disk = self.generate(self.rows)

    def _check(self, checks, name):
        return next(c for c in checks if c["name"] == name)

    def test_all_checks_pass_on_clean_split(self) -> None:
        checks = splits_mod.run_validations(
            self.assigned, self.disk, self.repo_root
        )
        self.assertTrue(all(c["passed"] for c in checks), checks)
        names = {c["name"] for c in checks}
        self.assertIn("every_wav_represented_exactly_once", names)
        self.assertIn("no_duplicate_local_path", names)
        self.assertIn("all_manifest_paths_exist", names)
        self.assertIn("core_splits_contain_both_classes", names)
        self.assertIn("test_unseen_tts_contains_spoof", names)
        self.assertIn("zero_tts_overlap_train_vs_unseen", names)

    def test_missing_file_detected(self) -> None:
        corrupted = self.assigned[:-1]
        checks = splits_mod.run_validations(
            corrupted, self.disk, self.repo_root
        )
        self.assertFalse(
            self._check(checks, "every_wav_represented_exactly_once")["passed"]
        )

    def test_duplicate_path_detected(self) -> None:
        corrupted = self.assigned + [dict(self.assigned[0])]
        checks = splits_mod.run_validations(
            corrupted, self.disk, self.repo_root
        )
        self.assertFalse(self._check(checks, "no_duplicate_local_path")["passed"])

    def test_deleted_audio_file_detected(self) -> None:
        # Delete one WAV from disk: the path-existence check must fail.
        victim = self.repo_root / self.assigned[0]["local_path"]
        victim.unlink()
        checks = splits_mod.run_validations(
            self.assigned, self.disk, self.repo_root
        )
        self.assertFalse(self._check(checks, "all_manifest_paths_exist")["passed"])

    def test_empty_split_detected(self) -> None:
        corrupted = [r for r in self.assigned if r["split"] != "test"]
        checks = splits_mod.run_validations(
            corrupted, self.disk, self.repo_root
        )
        self.assertFalse(
            self._check(checks, "all_expected_splits_present")["passed"]
        )

    def test_single_class_split_detected(self) -> None:
        corrupted = [
            dict(r, label="BONAFIDE")
            if r["split"] == "validation" and r["label"] == "SPOOF"
            else r
            for r in self.assigned
        ]
        checks = splits_mod.run_validations(
            corrupted, self.disk, self.repo_root
        )
        self.assertFalse(
            self._check(checks, "core_splits_contain_both_classes")["passed"]
        )

    def test_unseen_without_spoof_detected(self) -> None:
        corrupted = [
            r for r in self.assigned
            if not (r["split"] == "test-unseen-tts" and r["label"] == "SPOOF")
        ]
        checks = splits_mod.run_validations(
            corrupted, self.disk, self.repo_root
        )
        self.assertFalse(
            self._check(checks, "test_unseen_tts_contains_spoof")["passed"]
        )

    def test_tts_leak_detected(self) -> None:
        """Re-tagging one train spoof row with an unseen system must fail."""
        unseen = next(
            r for r in self.assigned
            if r["split"] == "test-unseen-tts" and r["label"] == "SPOOF"
        )
        corrupted = [
            dict(r, tts_system=unseen["tts_system"])
            if r["split"] == "train" and r["label"] == "SPOOF" and r is self.assigned[0]
            else r
            for r in self.assigned
        ]
        checks = splits_mod.run_validations(
            corrupted, self.disk, self.repo_root
        )
        self.assertFalse(
            self._check(checks, "zero_tts_overlap_train_vs_unseen")["passed"]
        )

    def test_validate_or_raise_raises_on_failure(self) -> None:
        corrupted = self.assigned[:-1]
        with self.assertRaises(splits_mod.SplitError):
            splits_mod.validate_or_raise(
                corrupted, self.disk, self.repo_root
            )


# ---------------------------------------------------------------------------
# Report contents
# ---------------------------------------------------------------------------


class TestBuildReport(SplitFixtureTest):
    def test_report_contents(self) -> None:
        rows = self.build_standard_fixture(comma_named_system="Resemble.ai (April 12th, 2025)")
        disk = splits_mod.scan_audio_files(
            self.audio_dir, self.data_dir, self.repo_root
        )
        cross = splits_mod.cross_check_manifest(disk, rows)
        rng = random.Random(splits_mod.SEED)
        assigned = splits_mod.build_splits(disk, rng)
        checks = splits_mod.validate_or_raise(assigned, disk, self.repo_root)
        report = splits_mod.build_report(
            disk, assigned, checks, cross, splits_mod.SEED, self.data_dir
        )

        self.assertTrue(report["validation"]["passed"])
        self.assertEqual(
            report["dataset"]["total_files"], len(self.manifest_rows)
        )
        self.assertEqual(report["manifest_cross_check"]["matched_paths"], len(rows))

        for name in ("train", "validation", "test", "test-unseen-tts"):
            self.assertIn(name, report["splits"])
            section = report["splits"][name]
            self.assertEqual(
                section["total"],
                section["class_counts"]["BONAFIDE"]
                + section["class_counts"]["SPOOF"],
            )
        self.assertTrue(report["unseen_tts_check"]["confirmed_disjoint"])
        self.assertTrue(report["unseen_tts_check"]["zero_overlap_with_train"])
        self.assertEqual(
            report["tts_system_overlap"]["train_vs_test-unseen-tts"], []
        )
        self.assertIn("Resemble.ai (April 12th, 2025)", report["dataset"]["spoof_tts_systems"])


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


class TestMainEndToEnd(SplitFixtureTest):
    def test_main_writes_outputs_and_passes(self) -> None:
        self.build_standard_fixture()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = splits_mod.main([])
        self.assertEqual(exit_code, 0)

        splits_path = self.data_dir / "splits.csv"
        report_path = self.data_dir / "split_report.json"
        self.assertTrue(splits_path.is_file())
        self.assertTrue(report_path.is_file())

        with splits_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), list(splits_mod.FIELDNAMES))
        self.assertEqual(len(rows), len(self.manifest_rows))
        paths = [r["local_path"] for r in rows]
        self.assertEqual(len(paths), len(set(paths)))

        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["validation"]["passed"])
        self.assertTrue(report["unseen_tts_check"]["confirmed_disjoint"])
        self.assertEqual(
            report["dataset"]["total_files"], len(self.manifest_rows)
        )

    def test_main_is_deterministic(self) -> None:
        self.build_standard_fixture()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            splits_mod.main([])
        first = (self.data_dir / "splits.csv").read_bytes()
        (self.data_dir / "splits.csv").unlink()
        with redirect_stdout(buffer):
            splits_mod.main([])
        second = (self.data_dir / "splits.csv").read_bytes()
        self.assertEqual(first, second)

    def test_validate_only_round_trip(self) -> None:
        self.build_standard_fixture()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(splits_mod.main([]), 0)
        # Clean the outputs: validate-only must regenerate the report from
        # the existing CSV without rebuilding the split.
        (self.data_dir / "split_report.json").unlink()
        with redirect_stdout(buffer):
            exit_code = splits_mod.main(["--validate-only"])
        self.assertEqual(exit_code, 0)
        self.assertFalse((self.data_dir / "splits.csv").exists() is False)
        self.assertTrue((self.data_dir / "split_report.json").is_file())

    def test_main_fails_when_disk_and_manifest_disagree(self) -> None:
        self.build_standard_fixture()
        # Delete a WAV AFTER the manifest was written.
        for wav in self.audio_dir.rglob("*.wav"):
            wav.unlink()
            break
        with redirect_stdout(io.StringIO()):
            exit_code = splits_mod.main([])
        self.assertEqual(exit_code, 1)
        self.assertFalse((self.data_dir / "splits.csv").exists())

    def test_main_fails_without_manifest(self) -> None:
        self.build_standard_fixture()
        splits_mod.MANIFEST.unlink()
        with redirect_stdout(io.StringIO()):
            exit_code = splits_mod.main([])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
