"""Unit tests for :mod:`dataset.audio`.

All audio is synthesized in memory (sine waves) — no dataset is
downloaded or required. Real-file loading is exercised with a tiny WAV
written to a temp dir and deleted afterwards.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.audio import (
    AudioClip,
    AudioValidationError,
    DurationValidationError,
    load_audio,
    preprocess,
    resample,
    to_mono,
    validate_duration,
)

SR = 16_000


def sine(seconds: float, sr: int = SR, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Synthesize a mono sine waveform (used everywhere below)."""
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * freq * t).astype(np.float32)


class TestAudioClipConstruction(unittest.TestCase):
    def test_from_array_enforces_invariants(self) -> None:
        clip = AudioClip.from_array(sine(0.1), sample_rate=SR)
        self.assertEqual(clip.samples.dtype, np.float32)
        self.assertEqual(clip.samples.ndim, 1)
        self.assertEqual(clip.num_samples, int(0.1 * SR))
        self.assertAlmostEqual(clip.duration, 0.1, places=3)

    def test_rejects_empty_nonfinite_and_bad_rate(self) -> None:
        with self.assertRaises(AudioValidationError):
            AudioClip.from_array(np.array([], dtype=np.float32), sample_rate=SR)
        with self.assertRaises(AudioValidationError):
            AudioClip.from_array(np.array([np.nan, 0.1], dtype=np.float32), sample_rate=SR)
        with self.assertRaises(AudioValidationError):
            AudioClip.from_array(np.array([np.inf, 0.1], dtype=np.float32), sample_rate=SR)
        with self.assertRaises(AudioValidationError):
            AudioClip.from_array(sine(0.1), sample_rate=0)
        with self.assertRaises(AudioValidationError):
            AudioClip.from_array(np.zeros((2, 100), dtype=np.float32), sample_rate=SR)

    def test_duration_and_num_samples_consistent(self) -> None:
        clip = AudioClip.from_array(sine(1.0, sr=8_000), sample_rate=8_000)
        self.assertEqual(clip.num_samples, 8_000)
        self.assertAlmostEqual(clip.duration, 1.0, places=6)


class TestToMono(unittest.TestCase):
    def test_mono_passthrough_is_copy(self) -> None:
        wave = sine(0.1)
        out = to_mono(wave)
        self.assertTrue(np.array_equal(wave, out))
        self.assertIsNot(wave, out)  # copy, not the same buffer

    def test_multichannel_averages_channels(self) -> None:
        left = sine(0.1, freq=440.0)
        right = sine(0.1, freq=880.0)
        stereo = np.stack([left, right])  # (channels, samples)
        mono = to_mono(stereo)
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(mono.shape[0], left.shape[0])
        self.assertTrue(np.allclose(mono, (left + right) / 2.0, atol=1e-6))

    def test_rejects_three_dimensional_input(self) -> None:
        with self.assertRaises(AudioValidationError):
            to_mono(np.zeros((2, 2, 10), dtype=np.float32))


class TestResample(unittest.TestCase):
    def test_same_rate_is_identity(self) -> None:
        wave = sine(0.1)
        out = resample(wave, orig_sr=SR, target_sr=SR)
        self.assertTrue(np.array_equal(wave, out))

    def test_downsample_and_upsample_lengths(self) -> None:
        wave = sine(1.0, sr=16_000)
        down = resample(wave, orig_sr=16_000, target_sr=8_000)
        self.assertEqual(down.shape[0], 8_000)
        up = resample(down, orig_sr=8_000, target_sr=32_000)
        self.assertEqual(up.shape[0], 32_000)

    def test_frequency_content_preserved(self) -> None:
        # A 440 Hz tone resampled 16k -> 8k stays a 440 Hz tone.
        wave = sine(0.5, sr=16_000, freq=440.0)
        down = resample(wave, orig_sr=16_000, target_sr=8_000)
        spectrum = np.abs(np.fft.rfft(down))
        peak_hz = float(np.argmax(spectrum)) * 8_000 / down.shape[0]
        self.assertAlmostEqual(peak_hz, 440.0, delta=5.0)

    def test_rejects_bad_sample_rates(self) -> None:
        with self.assertRaises(AudioValidationError):
            resample(sine(0.1), orig_sr=0, target_sr=8_000)
        with self.assertRaises(AudioValidationError):
            resample(sine(0.1), orig_sr=16_000, target_sr=-1)


class TestValidateDuration(unittest.TestCase):
    def test_within_range_passes(self) -> None:
        clip = AudioClip.from_array(sine(2.0), sample_rate=SR)
        self.assertAlmostEqual(validate_duration(clip), 2.0, places=3)

    def test_bounds_are_inclusive(self) -> None:
        clip = AudioClip.from_array(sine(1.0), sample_rate=SR)
        self.assertAlmostEqual(validate_duration(clip, 1.0, 1.0), 1.0, places=3)

    def test_too_short_and_too_long_raise(self) -> None:
        short = AudioClip.from_array(sine(0.1), sample_rate=SR)
        with self.assertRaises(DurationValidationError):
            validate_duration(short, min_duration_s=0.5, max_duration_s=None)
        long_clip = AudioClip.from_array(sine(2.0), sample_rate=SR)
        with self.assertRaises(DurationValidationError):
            validate_duration(long_clip, min_duration_s=None, max_duration_s=1.0)

    def test_none_disables_bounds(self) -> None:
        clip = AudioClip.from_array(sine(0.05), sample_rate=SR)
        self.assertAlmostEqual(
            validate_duration(clip, min_duration_s=None, max_duration_s=None), 0.05, places=3
        )


class TestLoadAudio(unittest.TestCase):
    def test_loads_temp_wav_and_validates(self) -> None:
        import soundfile as sf

        wave = sine(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            sf.write(str(path), wave, SR, subtype="PCM_16")
            clip = load_audio(path)
            self.assertEqual(clip.sample_rate, SR)
            self.assertAlmostEqual(clip.duration, 0.5, places=2)
            self.assertEqual(clip.samples.dtype, np.float32)

    def test_missing_file_raises_audio_validation_error(self) -> None:
        with self.assertRaises(AudioValidationError):
            load_audio("definitely/not/a/real/file.wav")

    def test_invalid_file_raises_audio_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "not_audio.wav"
            bad.write_bytes(b"this is not audio data")
            with self.assertRaises(AudioValidationError):
                load_audio(bad)


class TestPreprocess(unittest.TestCase):
    def test_file_path_full_pipeline(self) -> None:
        import soundfile as sf

        # 8 kHz stereo file, not yet 16 kHz mono — pipeline fixes both.
        left = sine(1.0, sr=8_000, freq=440.0)
        right = sine(1.0, sr=8_000, freq=440.0) * 0.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo_8k.wav"
            sf.write(str(path), np.stack([left, right]).T, 8_000, subtype="PCM_16")
            clip = preprocess(path)
            self.assertEqual(clip.sample_rate, 16_000)
            self.assertEqual(clip.samples.ndim, 1)
            self.assertAlmostEqual(clip.duration, 1.0, places=2)

    def test_array_input_requires_source_sample_rate(self) -> None:
        with self.assertRaises(AudioValidationError):
            preprocess(sine(1.0))  # missing source_sample_rate

    def test_array_input_full_pipeline(self) -> None:
        clip = preprocess(sine(1.0, sr=8_000), source_sample_rate=8_000)
        self.assertEqual(clip.sample_rate, 16_000)
        self.assertEqual(clip.samples.dtype, np.float32)
        self.assertAlmostEqual(clip.duration, 1.0, places=2)

    def test_duration_validation_applies(self) -> None:
        with self.assertRaises(DurationValidationError):
            preprocess(sine(0.1), source_sample_rate=SR)  # < default 0.5 s min
        ok = preprocess(sine(0.1), source_sample_rate=SR, min_duration_s=None)
        self.assertEqual(ok.sample_rate, SR)

    def test_custom_target_rate(self) -> None:
        clip = preprocess(sine(1.0, sr=8_000), source_sample_rate=8_000, sample_rate=22_050)
        self.assertEqual(clip.sample_rate, 22_050)


if __name__ == "__main__":
    unittest.main()
