"""Integration test: real checkpoint inference through the FastAPI app.

Runs **only** when the trained trial checkpoint exists locally (and is
explicitly separated from the lightweight fake-model unit tests). Uses a
synthetic WAV — no dataset required.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import DEFAULT_CHECKPOINT

CHECKPOINT = DEFAULT_CHECKPOINT


@unittest.skipUnless(
    CHECKPOINT.is_file(),
    f"real checkpoint not present: {CHECKPOINT}",
)
class TestRealCheckpointIntegration(unittest.TestCase):
    def test_predict_with_real_trained_model(self) -> None:
        import io

        import numpy as np
        import soundfile as sf

        app = create_app()
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["model_loaded"])

            t = np.linspace(0.0, 1.0, 16_000, endpoint=False)
            wave = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            buffer = io.BytesIO()
            sf.write(buffer, wave, 16_000, subtype="PCM_16", format="WAV")
            buffer.seek(0)

            response = client.post(
                "/predict",
                files={"file": ("tone.wav", buffer.getvalue(), "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn(body["label"], {"SPOOF", "BONAFIDE"})
            self.assertGreaterEqual(body["spoofProbability"], 0.0)
            self.assertLessEqual(body["spoofProbability"], 1.0)
            self.assertEqual(body["sample_rate"], 16_000)
            self.assertAlmostEqual(body["duration_seconds"], 1.0, places=2)
            self.assertGreaterEqual(body["inference_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
