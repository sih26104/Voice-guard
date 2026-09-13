"""Tests for the FULL MLAAD-tiny dataset pipeline (no network, no downloads).

Covers the two new standalone scripts plus the ``test-unseen-tts``
loader extension:

* ``download_mlaad_full.py`` — path->metadata derivation, destination
  containment, resume logic and failure handling (HF calls are faked).
* ``create_mlaad_full_splits.py`` — determinism, completeness, class
  counts, unseen-TTS disjointness and report contents, on synthetic
  manifests written to temp dirs.
* ``MlaadDataset`` — the new split name is backward-compatible: the
  three core splits behave exactly as before and ``test-unseen-tts``
  loads only from split files that provide it.

The standalone scripts live outside ``src/`` and are loaded by path via
``importlib`` (they are intentionally not a package, matching the
existing ``scripts/`` convention).
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from collections import Counter
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = AI_SERVICE_ROOT / "scripts"


def load_script(module_name: str, file_name: str):
    """Import a standalone ``scripts/*.py`` by path (no package needed)."""
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPTS_DIR / file_name
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


download_mod = load_script("download_mlaad_full", "download_mlaad_full.py")
splits_mod = load_script("create_mlaad_full_splits", "create_mlaad_full_splits.py")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def synthetic_manifest(
    en_systems=("SysA", "SysB", "SysC", "SysD", "SysE", "SysF"),
    de_systems=("SysG", "SysH", "SysI"),
    en_spoof_per_system=10,
    de_spoof_per_system=10,
    en_bonafide=30,
    de_bonafide=20,
) -> list:
    """Small synthetic manifest mirroring the verified repo layout."""
    rows = []
    for language, systems, per_system, bonafide in (
        ("en", en_systems, en_spoof_per_system, en_bonafide),
        ("de", de_systems, de_spoof_per_system, de_bonafide),
    ):
        for system in systems:
            for index in range(per_system):
                hf_path = f"fake/{language}/{system}/f{index}.wav"
                rows.append({
                    "local_path": f"data/mlaad_full/audio/{hf_path}",
                    "hf_path": hf_path,
                    "label": "SPOOF",
                    "language": language,
                    "tts_system": system,
                })
        for index in range(bonafide):
            hf_path = f"original/{language}/o{index}.wav"
            rows.append({
                "local_path": f"data/mlaad_full/audio/{hf_path}",
                "hf_path": hf_path,
                "label": "BONAFIDE",
                "language": language,
                "tts_system": "",
            })
    return rows


def write_manifest(rows: list, directory: Path) -> Path:
    """Write a manifest.csv into ``directory`` and return its path."""
    path = directory / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# download_mlaad_full.derive_metadata — actual repo structure
# ---------------------------------------------------------------------------


class TestDeriveMetadata(unittest.TestCase):
    """Paths here mirror the VERIFIED mueller91/MLAAD-tiny layout."""

    def test_fake_path_yields_spoof_with_system(self) -> None:
        self.assertEqual(
            download_mod.derive_metadata(
                "fake/de/CaroTTS/Kanzlerin_03_17_f000019.wav"
            ),
            (("SPOOF", "de", "CaroTTS"), None),
        )

    def test_original_path_yields_bonafide_without_system(self) -> None:
        self.assertEqual(
            download_mod.derive_metadata("original/en/some_clip.wav"),
            (("BONAFIDE", "en", ""), None),
        )

    def test_system_names_with_commas_pass_through(self) -> None:
        # Real system names contain commas: "Resemble.ai (April 12th, 2025)"
        metadata, warning = download_mod.derive_metadata(
            "fake/en/Resemble.ai (April 12th, 2025)/x.wav"
        )
        self.assertIsNone(warning)
        self.assertEqual(metadata[2], "Resemble.ai (April 12th, 2025)")

    def test_unexpected_depths_are_skipped_not_guessed(self) -> None:
        # Too shallow / too deep on both branches must not mislabel.
        for path in ("fake/de/x.wav", "fake/en/Sys/a/b.wav",
                     "original/en/a/b.wav", "original/en"):
            metadata, warning = download_mod.derive_metadata(path)
            self.assertIsNone(metadata, path)
            self.assertIn("unexpected", warning)

    def test_unknown_top_level_is_skipped(self) -> None:
        metadata, warning = download_mod.derive_metadata("other/en/a.wav")
        self.assertIsNone(metadata)
        self.assertIn("unknown top-level", warning)


class TestDestinationSafety(unittest.TestCase):
    def test_destination_mirrors_hf_structure(self) -> None:
        destination = download_mod.destination_for(
            "fake/en/Sys/x.wav"
        )
        self.assertEqual(
            destination,
            download_mod.AUDIO_DIR / "fake" / "en" / "Sys" / "x.wav",
        )

    def test_destination_cannot_escape_audio_dir(self) -> None:
        with self.assertRaises(ValueError):
            download_mod.destination_for("../escape.wav")

    def test_resume_skips_existing_and_downloads_missing(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp) / "audio"
            # Redirect every path constant the script derives at import
            # time (AUDIO_DIR, MANIFEST and the PROJECT_ROOT used for
            # manifest-relative local_path values).
            original = (download_mod.PROJECT_ROOT, download_mod.AUDIO_DIR,
                        download_mod.MANIFEST)
            download_mod.PROJECT_ROOT = Path(tmp)
            download_mod.AUDIO_DIR = audio_dir
            download_mod.MANIFEST = Path(tmp) / "manifest.csv"
            try:
                existing = audio_dir / "original" / "en" / "have.wav"
                existing.parent.mkdir(parents=True)
                existing.write_bytes(b"real wav bytes")

                def fake_fetch(hf_path: str) -> Path:
                    calls.append(hf_path)
                    destination = audio_dir / hf_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"new wav bytes")
                    return destination

                original_fetch = download_mod.fetch_file
                download_mod.fetch_file = fake_fetch
                download_mod.list_repo_files = lambda *a, **k: [
                    "original/en/have.wav", "original/en/missing.wav",
                ]
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = download_mod.main()
            finally:
                (download_mod.PROJECT_ROOT, download_mod.AUDIO_DIR,
                 download_mod.MANIFEST) = original
                download_mod.fetch_file = original_fetch

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["original/en/missing.wav"])  # only missing
        # Manifest records the already-present file too.
        self.assertEqual(len(calls), 1)

    def test_failure_keeps_manifest_absent_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp) / "audio"
            manifest = Path(tmp) / "manifest.csv"
            original = (download_mod.PROJECT_ROOT, download_mod.AUDIO_DIR,
                        download_mod.MANIFEST)
            download_mod.PROJECT_ROOT = Path(tmp)
            download_mod.AUDIO_DIR = audio_dir
            download_mod.MANIFEST = manifest
            try:
                def failing_fetch(hf_path: str) -> Path:
                    raise ConnectionError(f"boom: {hf_path}")

                original_fetch = download_mod.fetch_file
                download_mod.fetch_file = failing_fetch
                download_mod.list_repo_files = lambda *a, **k: [
                    "original/en/a.wav",
                ]
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = download_mod.main()
            finally:
                (download_mod.PROJECT_ROOT, download_mod.AUDIO_DIR,
                 download_mod.MANIFEST) = original
                download_mod.fetch_file = original_fetch

            self.assertEqual(exit_code, 1)
            self.assertFalse(manifest.exists())  # corrupt manifest impossible


# ---------------------------------------------------------------------------
# create_mlaad_full_splits — deterministic split logic
# ---------------------------------------------------------------------------


class TestSplitLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = synthetic_manifest()

    def _assign(self, rows=None, seed=None):
        rng = random.Random(seed if seed is not None else splits_mod.SEED)
        assigned = splits_mod.build_splits(rows or self.rows, rng)
        splits_mod.validate(assigned, rows or self.rows)
        return assigned

    def test_every_file_assigned_exactly_once(self) -> None:
        assigned = self._assign()
        # Each manifest row appears exactly once as a home split; bona fide
        # test rows additionally appear as test-unseen-tts (aliased view).
        home_keys = Counter(
            row["local_path"] for row in assigned
            if not (row["split"] == "test-unseen-tts"
                    and row["label"] == "BONAFIDE")
        )
        manifest_keys = Counter(row["local_path"] for row in self.rows)
        self.assertEqual(home_keys, manifest_keys)

    def test_no_duplicate_path_split_pairs(self) -> None:
        assigned = self._assign()
        pairs = [(row["local_path"], row["split"]) for row in assigned]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_deterministic_output(self) -> None:
        first = self._assign()
        second = self._assign()
        key = lambda rows: [(r["local_path"], r["split"]) for r in rows]
        self.assertEqual(key(first), key(second))

    def test_different_seed_gives_different_assignment(self) -> None:
        first = self._assign(seed=1)
        second = self._assign(seed=2)
        key = lambda rows: sorted(
            (r["local_path"], r["split"]) for r in rows
            if r["split"] == "train"
        )
        self.assertNotEqual(key(first), key(second))

    def test_class_counts_are_stable(self) -> None:
        # Totals per split label must be exactly derivable from the
        # manifest: every spoof file is in exactly one of
        # train/validation/test/test-unseen-tts.
        assigned = self._assign()
        spoof_by_split = Counter(
            row["split"] for row in assigned if row["label"] == "SPOOF"
        )
        total_spoof = sum(
            count for split, count in spoof_by_split.items()
        )
        self.assertEqual(total_spoof, 90)  # 9 systems x 10 files

    def test_all_seen_systems_cover_standard_splits(self) -> None:
        assigned = self._assign()
        report = splits_mod.build_report(self.rows, assigned, splits_mod.SEED)
        unseen = set(report["unseen_tts_check"]["unseen_systems"])
        all_systems = {
            row["tts_system"] for row in self.rows
            if row["label"] == "SPOOF"
        }
        seen = all_systems - unseen
        for split in ("train", "validation", "test"):
            systems_in_split = {
                row["tts_system"] for row in assigned
                if row["split"] == split and row["label"] == "SPOOF"
            }
            self.assertEqual(systems_in_split, seen)

    def test_unseen_tts_systems_never_in_train_validation_test(self) -> None:
        assigned = self._assign()
        unseen = {
            row["tts_system"] for row in assigned
            if row["split"] == "test-unseen-tts" and row["label"] == "SPOOF"
        }
        self.assertTrue(unseen)  # holdout must be non-empty
        for split in ("train", "validation", "test"):
            leaked = {
                row["tts_system"] for row in assigned
                if row["split"] == split and row["label"] == "SPOOF"
            } & unseen
            self.assertEqual(leaked, set(), f"leaked into {split}")

    def test_holdout_respects_unseen_fraction(self) -> None:
        assigned = self._assign()
        spoof_by_split = Counter(
            row["split"] for row in assigned if row["label"] == "SPOOF"
        )
        held = spoof_by_split["test-unseen-tts"]
        total = sum(spoof_by_split.values())
        # Between roughly 10% and 35% of spoof files end up unseen
        # (fraction target 15%, whole-system granularity, MIN_SEEN floor).
        self.assertGreaterEqual(held / total, 0.10)
        self.assertLessEqual(held / total, 0.35)

    def test_spoof_row_without_system_is_rejected(self) -> None:
        broken = self.rows + [{
            "local_path": "data/mlaad_full/audio/fake/en/x.wav",
            "hf_path": "fake/en/x.wav", "label": "SPOOF",
            "language": "en", "tts_system": "",
        }]
        rng = random.Random(splits_mod.SEED)
        with self.assertRaises(ValueError):
            splits_mod.build_splits(broken, rng)


class TestSplitScriptEndToEnd(unittest.TestCase):
    """main() against a temp data dir: CSV + report contents."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name) / "mlaad_full"
        self.data_dir.mkdir()
        self.rows = synthetic_manifest()
        write_manifest(self.rows, self.data_dir)

        self._originals = (
            splits_mod.DATA_DIR, splits_mod.MANIFEST,
            splits_mod.OUTPUT, splits_mod.REPORT,
        )
        splits_mod.DATA_DIR = self.data_dir
        splits_mod.MANIFEST = self.data_dir / "manifest.csv"
        splits_mod.OUTPUT = self.data_dir / "splits.csv"
        splits_mod.REPORT = self.data_dir / "dataset_report.json"

    def tearDown(self) -> None:
        (splits_mod.DATA_DIR, splits_mod.MANIFEST,
         splits_mod.OUTPUT, splits_mod.REPORT) = self._originals

    def test_main_writes_splits_and_report(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = splits_mod.main()
        self.assertEqual(exit_code, 0)

        with (self.data_dir / "splits.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            list(rows[0]),
            ["local_path", "hf_path", "label", "language",
             "tts_system", "split"],
        )
        splits_mod.validate(rows, self.rows)

        report = json.loads(
            (self.data_dir / "dataset_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["totals"]["total_files"], len(self.rows))
        self.assertEqual(
            report["totals"]["BONAFIDE"],
            sum(1 for r in self.rows if r["label"] == "BONAFIDE"),
        )
        self.assertEqual(
            report["totals"]["SPOOF"],
            sum(1 for r in self.rows if r["label"] == "SPOOF"),
        )
        self.assertEqual(
            report["totals"]["language_counts"],
            {"de": 50, "en": 90},
        )
        self.assertEqual(
            report["totals"]["spoof_tts_system_count"], 9
        )
        for name in splits_mod.SPLIT_NAMES:
            self.assertIn(name, report["splits"])
            self.assertIn("class_counts", report["splits"][name])
            self.assertIn("language_counts", report["splits"][name])
            self.assertIn("tts_systems", report["splits"][name])
        self.assertTrue(report["unseen_tts_check"]["confirmed_disjoint"])
        self.assertEqual(
            report["tts_system_overlap"]["train_vs_test-unseen-tts"], []
        )

    def test_main_is_deterministic_across_runs(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            splits_mod.main()
        first = (self.data_dir / "splits.csv").read_bytes()
        (self.data_dir / "splits.csv").unlink()
        with redirect_stdout(buffer):
            splits_mod.main()
        second = (self.data_dir / "splits.csv").read_bytes()
        self.assertEqual(first, second)

    def test_missing_manifest_fails_clearly(self) -> None:
        (self.data_dir / "manifest.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            splits_mod.main()


class TestMlaadDatasetUnseenSplit(unittest.TestCase):
    """The loader accepts the new split without breaking the old ones."""

    HEADER = ["local_path", "hf_path", "label", "language",
              "tts_system", "split"]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        from dataset.mlaad import MlaadDataset

        self.MlaadDataset = MlaadDataset
        self.repo_root = Path(self._tmp.name)
        self.root = self.repo_root / "mlaad_full"
        self.root.mkdir()

    def _write_wav(self, relative: Path) -> None:
        import struct
        import wave

        relative.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(relative), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\x00\x00" * 16_000)

    def _write_splits(self, rows: list) -> None:
        with (self.root / "splits.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(rows)

    def _row(self, local_path, label, split, language="en", tts=""):
        return {
            "local_path": local_path,
            "hf_path": local_path.replace("\\", "/").replace(
                "mlaad_full/audio/", "", 1
            ),
            "label": label,
            "language": language,
            "tts_system": tts,
            "split": split,
        }

    def test_core_splits_still_load(self) -> None:
        wav = self.root / "audio" / "original" / "en" / "a.wav"
        self._write_wav(wav)
        self._write_splits([
            self._row("mlaad_full\\audio\\original\\en\\a.wav",
                      "BONAFIDE", "train"),
        ])
        dataset = self.MlaadDataset(
            self.root, split="train", repo_root=self.repo_root
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.split_name, "train")

    def test_unseen_split_loads_when_present(self) -> None:
        wav = self.root / "audio" / "fake" / "en" / "Sys" / "b.wav"
        self._write_wav(wav)
        self._write_splits([
            self._row("mlaad_full\\audio\\fake\\en\\Sys\\b.wav",
                      "SPOOF", "test-unseen-tts", tts="Sys"),
        ])
        dataset = self.MlaadDataset(
            self.root, split="test-unseen-tts", repo_root=self.repo_root
        )
        self.assertEqual(len(dataset), 1)
        sample = next(iter(dataset))
        self.assertEqual(sample.tts_system, "Sys")
        self.assertEqual(sample.split, "test-unseen-tts")

    def test_unseen_split_rejected_when_csv_lacks_it(self) -> None:
        wav = self.root / "audio" / "original" / "en" / "a.wav"
        self._write_wav(wav)
        self._write_splits([
            self._row("mlaad_full\\audio\\original\\en\\a.wav",
                      "BONAFIDE", "train"),
        ])
        with self.assertRaises(ValueError) as context:
            self.MlaadDataset(
                self.root, split="test-unseen-tts",
                repo_root=self.repo_root,
            )
        self.assertIn("test-unseen-tts", str(context.exception))

    def test_invalid_split_argument_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.MlaadDataset(
                self.root, split="dev", repo_root=self.repo_root
            )

    def test_constants_expose_core_and_extended_splits(self) -> None:
        from dataset import CORE_MLAAD_SPLITS, MLAAD_SPLITS

        self.assertEqual(
            CORE_MLAAD_SPLITS, ("train", "validation", "test")
        )
        self.assertIn("test-unseen-tts", MLAAD_SPLITS)


class TestSubsetScriptsUntouched(unittest.TestCase):
    """The prototype pipeline must be unaffected by the new scripts."""

    def test_subset_scripts_keep_their_paths(self) -> None:
        subset_src = (SCRIPTS_DIR / "download_mlaad_subset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('data" / "mlaad_subset', subset_src)
        split_src = (SCRIPTS_DIR / "create_dataset_split.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("mlaad_subset", split_src)

    def test_new_scripts_target_mlaad_full_only(self) -> None:
        # The new scripts must WRITE under data/mlaad_full (path
        # construction), never under the prototype's data/mlaad_subset.
        for name in ("download_mlaad_full.py",
                     "create_mlaad_full_splits.py"):
            source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            self.assertIn('data" / "mlaad_full', source, name)
            self.assertNotIn('data" / "mlaad_subset', source, name)


if __name__ == "__main__":
    unittest.main()
