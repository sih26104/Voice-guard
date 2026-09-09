"""Unit tests for :mod:`dataset.mlaad`.

Everything runs against **temporary fixtures**: tiny WAVs and synthetic
splits.csv files written to temp dirs and deleted afterwards. The real
dataset is only *listed*, never modified, and exactly one real file is
loaded once to prove the loader works end-to-end against the actual
data (test skipped automatically if the subset is not present).
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset.audio import DurationValidationError
from dataset.labels import Label
from dataset.mlaad import (
    MLAAD_SPLITS,
    MlaadDataset,
    MlaadDatasetError,
    MlaadSample,
)

SR = 16_000

HEADER = ["local_path", "hf_path", "label", "language", "tts_system", "split"]


def write_wav(path: Path, seconds: float = 1.0, sr: int = SR) -> None:
    """Write a tiny sine WAV so preprocessing has real audio to decode."""
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    sf.write(str(path), (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), sr)


def csv_row(rel_path: str, label: str = "BONAFIDE", split: str = "train",
            language: str = "en", tts_system: str = "") -> dict:
    # Mirror the real CSV: local_path is repo-root-relative and includes
    # the dataset directory name (real data uses "data\\mlaad_subset\\...").
    prefixed = "mlaad_subset\\" + rel_path
    return {
        "local_path": prefixed,
        "hf_path": rel_path.replace("\\", "/"),
        "label": label,
        "language": language,
        "tts_system": tts_system,
        "split": split,
    }


def make_fixture(rows: list, repo_root: Path) -> Path:
    """Create audio + splits.csv under repo_root; returns the dataset dir."""
    for row in rows:
        audio_file = repo_root / row["local_path"].replace("\\", "/")
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        write_wav(audio_file, seconds=1.0)
    dataset_dir = repo_root / "mlaad_subset"
    with (dataset_dir / "splits.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return dataset_dir


def default_rows() -> list:
    """2 train / 1 validation / 1 test; mixed labels and languages."""
    return [
        csv_row("audio\\original\\en\\a.wav", "BONAFIDE", "train"),
        csv_row("audio\\fake\\en\\SysA\\b.wav", "SPOOF", "train", tts_system="SysA"),
        csv_row("audio\\original\\de\\c.wav", "BONAFIDE", "validation", "de"),
        csv_row("audio\\fake\\de\\SysB\\d.wav", "SPOOF", "test", "de", "SysB"),
    ]


class MlaadFixtureTest(unittest.TestCase):
    """Base class wiring up a temp dataset fixture per test.

    Fixture ``local_path`` values mirror the real CSV (relative paths with
    Windows separators); the loader's ``repo_root`` is pointed at the temp
    dir so they resolve inside the sandbox.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.root = self.repo_root / "mlaad_subset"
        self.root.mkdir()
        self.rows = default_rows()
        make_fixture(self.rows, self.repo_root)

    def make_dataset(self, split: str = "train") -> MlaadDataset:
        """Loader bound to the temp fixture (matches the real path layout)."""
        return MlaadDataset(self.root, split=split, repo_root=self.repo_root)

    def write_csv_only(self, rows: list) -> None:
        """Replace splits.csv without regenerating audio files."""
        with (self.root / "splits.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)


class TestSplitLoading(MlaadFixtureTest):
    def test_train_split(self) -> None:
        ds = self.make_dataset()
        self.assertEqual(len(ds), 2)
        samples = list(ds)
        self.assertEqual({s.label for s in samples}, {Label.BONAFIDE, Label.SPOOF})
        self.assertTrue(all(s.split == "train" for s in samples))
        self.assertTrue(all(s.audio_path is not None and s.audio_path.is_file() for s in samples))

    def test_validation_split(self) -> None:
        ds = self.make_dataset("validation")
        self.assertEqual(len(ds), 1)
        sample = next(iter(ds))
        self.assertEqual(sample.label, Label.BONAFIDE)
        self.assertEqual(sample.language, "de")

    def test_test_split(self) -> None:
        ds = self.make_dataset("test")
        self.assertEqual(len(ds), 1)
        sample = next(iter(ds))
        self.assertEqual(sample.label, Label.SPOOF)
        self.assertEqual(sample.tts_system, "SysB")

    def test_lazy_iteration_does_not_decode_audio(self) -> None:
        # Corrupt every WAV: iteration must still succeed (existence is
        # checked, decoding is deferred to waveform()).
        for wav in self.root.rglob("*.wav"):
            wav.write_bytes(b"corrupted")
        ds = self.make_dataset()
        self.assertEqual(len(ds), 2)
        self.assertEqual(len(list(ds)), 2)


class TestSampleContent(MlaadFixtureTest):
    def test_label_conversion_is_case_insensitive(self) -> None:
        # Relabel an existing fixture file (audio already materialized).
        self.write_csv_only([csv_row("audio\\original\\en\\a.wav", "Bonafide", "train")])
        sample = next(iter(self.make_dataset()))
        self.assertIs(sample.label, Label.BONAFIDE)

    def test_sample_carries_full_metadata(self) -> None:
        ds = self.make_dataset()
        spoof = next(s for s in ds if s.label is Label.SPOOF)
        self.assertIsInstance(spoof, MlaadSample)
        self.assertEqual(spoof.language, "en")
        self.assertEqual(spoof.tts_system, "SysA")
        bonafide = next(s for s in ds if s.label is Label.BONAFIDE)
        self.assertIsNone(bonafide.tts_system)  # empty CSV cell -> None
        self.assertEqual(spoof.key, "mlaad_subset\\audio\\fake\\en\\SysA\\b.wav")

    def test_by_label_accessor(self) -> None:
        ds = self.make_dataset()
        self.assertEqual(len(ds.by_label(Label.SPOOF)), 1)
        self.assertEqual(len(ds.by_label("bonafide")), 1)

    def test_audio_is_subclass_of_audio_sample(self) -> None:
        from dataset.base_audio_dataset import AudioSample

        sample = next(iter(self.make_dataset()))
        self.assertIsInstance(sample, AudioSample)


class TestErrorHandling(MlaadFixtureTest):
    def test_missing_audio_file_raises(self) -> None:
        (self.root / "audio" / "original" / "en" / "a.wav").unlink()
        ds = self.make_dataset()
        with self.assertRaises(MlaadDatasetError) as ctx:
            list(ds)
        self.assertIn("a.wav", str(ctx.exception))

    def test_invalid_label_raises(self) -> None:
        self.write_csv_only([csv_row("audio\\x.wav", "SYNTHETIC", "train")])
        with self.assertRaises(MlaadDatasetError) as ctx:
            self.make_dataset()
        self.assertIn("SYNTHETIC", str(ctx.exception))

    def test_invalid_split_value_in_csv_raises(self) -> None:
        self.write_csv_only([csv_row("audio\\x.wav", "BONAFIDE", "dev")])
        with self.assertRaises(MlaadDatasetError) as ctx:
            self.make_dataset()
        self.assertIn("dev", str(ctx.exception))

    def test_invalid_split_argument_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.make_dataset("dev")

    def test_malformed_row_too_many_values(self) -> None:
        # 7 values under the 6-column header: the 7th lands in DictReader's
        # None restkey -> must be rejected as a ragged row.
        with (self.root / "splits.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerow(
                ["a.wav", "hf/a.wav", "BONAFIDE", "en", "", "train", "EXTRA"]
            )
        with self.assertRaises(MlaadDatasetError) as ctx:
            MlaadDataset(self.root, split="train", repo_root=self.repo_root)
        self.assertIn("more values than header columns", str(ctx.exception))

    def test_malformed_row_missing_value(self) -> None:
        with (self.root / "splits.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerow(["a.wav", "hf/a.wav", "BONAFIDE", "en", ""])  # split missing
        with self.assertRaises(MlaadDatasetError) as ctx:
            self.make_dataset()
        self.assertIn("split", str(ctx.exception))

    def test_empty_local_path_raises(self) -> None:
        self.write_csv_only([{**csv_row("audio\\original\\en\\a.wav"), "local_path": ""}])
        # Row validation is eager, so the constructor itself raises.
        with self.assertRaises(MlaadDatasetError):
            self.make_dataset()

    def test_empty_language_raises(self) -> None:
        self.write_csv_only([csv_row("audio\\x.wav", "BONAFIDE", "train", language="")])
        with self.assertRaises(MlaadDatasetError):
            self.make_dataset()

    def test_duplicate_local_path_raises(self) -> None:
        rows = [csv_row("audio\\x.wav", "BONAFIDE", "train"),
                csv_row("audio\\x.wav", "SPOOF", "train", tts_system="SysA")]
        self.write_csv_only(rows)
        with self.assertRaises(MlaadDatasetError) as ctx:
            self.make_dataset()
        self.assertIn("duplicate", str(ctx.exception))

    def test_missing_splits_csv_raises(self) -> None:
        (self.root / "splits.csv").unlink()
        with self.assertRaises(MlaadDatasetError):
            self.make_dataset()

    def test_header_only_csv_raises(self) -> None:
        self.write_csv_only([])
        with self.assertRaises(MlaadDatasetError):
            self.make_dataset()

    def test_quoted_commas_in_paths_are_handled(self) -> None:
        # Mirrors the real CSV: TTS system names like "Cartesia.ai (Sonic-3)"
        # contain characters that must survive CSV quoting; hf_path may
        # contain commas in the real data.
        rows = [csv_row("audio\\fake\\en\\Sys, With Comma\\x.wav", "SPOOF", "train",
                        tts_system="Sys, With Comma")]
        make_fixture(rows, self.repo_root)
        sample = next(iter(self.make_dataset()))
        self.assertEqual(sample.tts_system, "Sys, With Comma")
        self.assertTrue(sample.audio_path.is_file())


class TestPreprocessingIntegration(MlaadFixtureTest):
    def test_waveform_passes_through_preprocess(self) -> None:
        ds = self.make_dataset()
        sample = next(iter(ds))
        clip = sample.waveform()
        self.assertEqual(clip.sample_rate, 16_000)
        self.assertEqual(clip.samples.dtype, np.float32)
        self.assertEqual(clip.samples.ndim, 1)
        self.assertAlmostEqual(clip.duration, 1.0, places=2)

    def test_waveform_duration_bounds_are_configurable(self) -> None:
        # Fixture clips are 1.0 s; a min bound above that must raise.
        ds = self.make_dataset()
        sample = next(iter(ds))
        with self.assertRaises(DurationValidationError):
            sample.waveform(min_duration_s=2.0)

    def test_waveform_missing_file_raises_audio_error(self) -> None:
        ds = self.make_dataset()
        sample = next(iter(ds))
        sample.audio_path.unlink()  # deleted after existence check
        from dataset.audio import AudioValidationError

        with self.assertRaises(AudioValidationError):
            sample.waveform()


class TestRealDataset(unittest.TestCase):
    """Smoke test against the actual local subset (read-only, skippable)."""

    REAL_ROOT = (Path(__file__).resolve().parents[1] / ".." / "data" / "mlaad_subset").resolve()

    def setUp(self) -> None:
        if not (self.REAL_ROOT / "splits.csv").is_file():
            self.skipTest("real MLAAD subset not present on this machine")

    def test_real_split_counts(self) -> None:
        expected = {"train": 700, "validation": 150, "test": 150}
        for split, count in expected.items():
            with self.subTest(split=split):
                self.assertEqual(len(MlaadDataset(self.REAL_ROOT, split=split)), count)

    def test_real_labels_are_canonical(self) -> None:
        ds = MlaadDataset(self.REAL_ROOT, split="train")
        labels = {s.label for s in ds}
        self.assertEqual(labels, {Label.BONAFIDE, Label.SPOOF})

    def test_one_real_sample_preprocesses(self) -> None:
        ds = MlaadDataset(self.REAL_ROOT, split="test")
        sample = next(iter(ds))
        clip = sample.waveform(min_duration_s=0.5, max_duration_s=None)
        self.assertEqual(clip.sample_rate, 16_000)
        self.assertEqual(clip.samples.dtype, np.float32)
        self.assertGreater(clip.num_samples, 0)
        self.assertLessEqual(clip.duration, 45.0)  # real clips run to ~38.5 s


if __name__ == "__main__":
    unittest.main()
