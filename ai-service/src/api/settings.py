"""API configuration: upload limits, supported types, checkpoint location."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_EXTENSIONS",
    "CHECKPOINT_ENV_VAR",
    "DEFAULT_CHECKPOINT",
    "checkpoint_path",
]

#: Hard upload cap: a 60 s mono 16 kHz float32 WAV is ~1.9 MB, so 25 MiB
#: is generous for legitimate clips while blocking abusive payloads before
#: any expensive decoding happens.
MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

#: File extensions accepted by /predict (validated before decoding).
SUPPORTED_EXTENSIONS: frozenset = frozenset({".wav"})

#: Environment variable overriding the checkpoint location.
CHECKPOINT_ENV_VAR: str = "VOICEGUARD_CHECKPOINT"

#: Default trained checkpoint (existing local trial run).
DEFAULT_CHECKPOINT: Path = Path("models/trial/best_head.pt")


def checkpoint_path(value: str | None = None) -> Path:
    """Resolve the checkpoint path: explicit arg > env var > default."""
    if value:
        return Path(value)
    env_value = os.environ.get(CHECKPOINT_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_CHECKPOINT
