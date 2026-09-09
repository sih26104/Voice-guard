"""FastAPI inference service for VoiceGuard.

Serves the trained frozen-encoder + head model over HTTP:

    GET  /health   -> service availability
    POST /predict  -> multipart WAV upload -> BONAFIDE/SPOOF probability

Design:
    * The model is loaded **once** in the FastAPI lifespan (not per
      request) and fails fast at startup if the checkpoint is missing.
    * Audio preprocessing and inference are **reused** from
      ``dataset.audio`` and ``model`` — nothing duplicated here.
    * Uploaded audio lives only in memory/temp storage and is deleted
      right after inference; it is never persisted or logged.
    * CPU-only inference; the checkpoint path is configurable via the
      ``VOICEGUARD_CHECKPOINT`` environment variable.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status

# Current (non-deprecated) Starlette status names.
HTTP_413_TOO_LARGE = status.HTTP_413_CONTENT_TOO_LARGE
HTTP_422_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT

from dataset.audio import AudioValidationError, preprocess
from model import load_voiceguard_model
from model.classifier import VoiceGuardClassifier, predict

from .settings import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
    checkpoint_path,
)

__all__ = ["create_app"]

_CHECKPOINT_ENV_VAR = "VOICEGUARD_CHECKPOINT"


@dataclass
class ModelRegistry:
    """Holds the single model instance shared by all requests."""

    model: VoiceGuardClassifier
    encoder_name: str


def _load_registry(ckpt_path: Path) -> ModelRegistry:
    """Load the trained head into the pretrained-encoder model (once)."""
    from scripts.common import load_checkpoint  # local import: avoids cycle at module import

    def build_model(encoder_name: str) -> VoiceGuardClassifier:
        # Encoder weights come from the local HF cache (downloaded once
        # during model preparation) — no per-request downloads.
        return load_voiceguard_model(encoder_name=encoder_name)

    model, _config, _metadata = load_checkpoint(ckpt_path, build_model)
    model.eval()
    return ModelRegistry(model=model, encoder_name=model.encoder_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; nothing to clean up at shutdown."""
    ckpt = checkpoint_path(os.environ.get(_CHECKPOINT_ENV_VAR))
    if not ckpt.is_file():
        raise RuntimeError(
            f"VoiceGuard checkpoint not found at {ckpt}. Set "
            f"{_CHECKPOINT_ENV_VAR} to a trained best_head.pt file."
        )
    app.state.registry = _load_registry(ckpt)
    yield
    app.state.registry = None


def create_app() -> FastAPI:
    """Build the FastAPI application (separated for testing)."""
    app = FastAPI(
        title="VoiceGuard AI Service",
        description=(
            "BONAFIDE vs SPOOF voice-cloning detection prototype "
            "(SIH 2026 PS 26104). Prototype baseline — NOT production accuracy."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def _validate_upload(upload: UploadFile) -> None:
    """Reject missing/empty/oversized/unsupported uploads with clear 4xx."""
    if upload is None or upload.filename is None or not upload.filename.strip():
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail="No audio file provided. Attach a WAV file in the 'file' field.",
        )
    extension = Path(upload.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {extension or '(none)'!r}; "
                   f"expected one of {sorted(SUPPORTED_EXTENSIONS)}.",
        )


async def _read_audio_bytes(upload: UploadFile) -> bytes:
    """Read the upload, enforcing the size limit before decoding."""
    size = MAX_UPLOAD_BYTES
    buffer = bytearray()
    while chunk := await upload.read(1 << 20):  # 1 MiB chunks
        buffer.extend(chunk)
        if len(buffer) > size:
            raise HTTPException(
                status_code=HTTP_413_TOO_LARGE,
                detail=f"Audio file exceeds the {size // (1 << 20)} MiB limit.",
            )
    if not buffer:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE,
            detail="Uploaded file is empty.",
        )
    return bytes(buffer)


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict:
        ready = getattr(app.state, "registry", None) is not None
        return {
            "status": "ok" if ready else "starting",
            "service": "voiceguard-ai",
            "model_loaded": ready,
            "device": "cpu",
        }

    @app.post("/predict")
    async def predict_endpoint(file: UploadFile | None = File(default=None)) -> dict:
        _validate_upload(file)
        audio_bytes = await _read_audio_bytes(file)

        registry: ModelRegistry = app.state.registry
        if registry is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is still loading; retry shortly.",
            )

        # Temp file only for librosa/soundfile decoding — deleted in the
        # finally block; never persisted, never logged, never returned.
        tmp_path = None
        started = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(file.filename or "audio.wav").suffix.lower() or ".wav",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(audio_bytes)
            clip = preprocess(tmp_path)  # existing utilities: mono, 16 kHz, validated
            result = predict(registry.model, clip.samples)  # existing inference API
        except AudioValidationError as exc:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE,
                detail=f"Audio could not be decoded or validated: {exc}",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - keep internals out of responses
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failed; see server logs for details.",
            ) from exc
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

        inference_ms = (time.perf_counter() - started) * 1000.0
        return {
            "spoofProbability": result.spoof_probability,
            "label": result.label.upper(),  # "SPOOF" | "BONAFIDE"
            "duration_seconds": round(clip.duration, 3),
            "sample_rate": clip.sample_rate,
            "inference_time_ms": round(inference_ms, 1),
        }


app = create_app()
