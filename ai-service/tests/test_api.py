"""Unit tests for the FastAPI inference service.

Lightweight by design: a **fake model** is injected into the app state so
no checkpoint or encoder download is needed. The real-checkpoint smoke
test lives in ``test_api_integration.py`` and self-skips.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from api.main import create_app
from api.main import ModelRegistry
from api.settings import MAX_UPLOAD_BYTES, checkpoint_path
from dataset.audio import AudioClip

SR = 16_000


def wav_bytes(seconds: float = 1.0, freq: float = 440.0, sr: int = SR) -> bytes:
    """Synthesize a small mono WAV in memory."""
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    wave = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wave, sr, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def fake_model():
    """Deterministic stand-in exposing the predict()-consumed surface."""
    from model.classifier import VoiceGuardClassifier

    return VoiceGuardClassifier.with_tiny_config()


class FakePredict:
    """Callable replacing model.classifier.predict with fixed output."""

    def __init__(self, label: str = "spoof", spoof_probability: float = 0.87):
        self.label = label
        self.spoof_probability = spoof_probability
        self.calls = []

    def __call__(self, model, waveform):
        self.calls.append(waveform.shape)
        from model.classifier import SpoofPrediction

        p_spoof = self.spoof_probability
        return SpoofPrediction(
            label=self.label,
            confidence=p_spoof if self.label == "spoof" else 1.0 - p_spoof,
            probabilities={"bonafide": 1.0 - p_spoof, "spoof": p_spoof},
            spoof_probability=p_spoof,
        )


class ApiTestCase(unittest.TestCase):
    """Base: app with a fake model registry injected (no checkpoint IO)."""

    def setUp(self) -> None:
        self.app = create_app()
        self.app.state.registry = ModelRegistry(
            model=fake_model(), encoder_name="fake/wav2vec2"
        )
        self.client = TestClient(self.app)
        self.fake = FakePredict()
        # Patch where predict is looked up at call time (api.main namespace).
        patcher = patch("api.main.predict", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestHealth(unittest.TestCase):
    def test_health_ok_with_model(self) -> None:
        app = create_app()
        app.state.registry = ModelRegistry(model=fake_model(), encoder_name="fake")
        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["device"], "cpu")


class TestPredictValid(ApiTestCase):
    def test_valid_wav_returns_schema(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("clip.wav", wav_bytes(1.0), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["label"], {"SPOOF", "BONAFIDE"})
        self.assertGreaterEqual(body["spoofProbability"], 0.0)
        self.assertLessEqual(body["spoofProbability"], 1.0)
        self.assertEqual(body["sample_rate"], 16_000)
        self.assertGreater(body["duration_seconds"], 0.0)
        self.assertGreaterEqual(body["inference_time_ms"], 0.0)

    def test_existing_preprocessing_was_used(self) -> None:
        # The waveform handed to predict() must be the canonical clip
        # (float32, mono, 16 kHz) produced by dataset.audio.preprocess.
        self.client.post(
            "/predict",
            files={"file": ("clip.wav", wav_bytes(0.5), "audio/wav")},
        )
        received = self.fake.calls[0]
        self.assertEqual(received[0], int(0.5 * SR))

    def test_temp_file_is_deleted_after_inference(self) -> None:
        created = []
        original = Path

        class TrackingPath(original):
            def unlink(self, *args, **kwargs):
                created.append(str(self))
                return super().unlink(*args, **kwargs)

        with patch("api.main.Path", TrackingPath):
            self.client.post(
                "/predict",
                files={"file": ("clip.wav", wav_bytes(0.3), "audio/wav")},
            )
        self.assertTrue(any(name.endswith(".wav") for name in created))


class TestPredictRejections(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.app = create_app()
        self.app.state.registry = ModelRegistry(
            model=fake_model(), encoder_name="fake"
        )
        self.client = TestClient(self.app)
        patcher = patch("api.main.predict", FakePredict())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_file_field(self) -> None:
        response = self.client.post("/predict")  # no multipart body at all
        self.assertEqual(response.status_code, 422)

    def test_unsupported_extension(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_empty_wav_payload(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)

    def test_non_audio_bytes_with_wav_extension(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("fake.wav", b"this is not audio", "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("decode", response.json()["detail"])

    def test_oversized_upload_rejected_before_decode(self) -> None:
        payload = b"\x00" * (MAX_UPLOAD_BYTES + 1)
        response = self.client.post(
            "/predict",
            files={"file": ("big.wav", payload, "audio/wav")},
        )
        self.assertEqual(response.status_code, 413)

    def test_no_paths_leak_into_responses(self) -> None:
        response = self.client.post(
            "/predict",
            files={"file": ("clip.wav", wav_bytes(0.3), "audio/wav")},
        )
        body = response.json()
        serialized = str(body)
        self.assertNotIn(str(Path(__file__).parent), serialized)
        self.assertNotIn("models/trial", serialized)
        self.assertNotIn("C:\\", serialized)


class TestModelLoadingFailure(unittest.TestCase):
    def test_startup_fails_when_checkpoint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope" / "best_head.pt"
            with patch.dict("os.environ", {"VOICEGUARD_CHECKPOINT": str(missing)}):
                self.assertEqual(checkpoint_path(), missing)
                with self.assertRaises(RuntimeError) as ctx:
                    with TestClient(create_app()):
                        pass
                self.assertIn("checkpoint not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
