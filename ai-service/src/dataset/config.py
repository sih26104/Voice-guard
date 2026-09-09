"""Dataset path/label configuration for VoiceGuard.

Deliberately dataset-agnostic: nothing here encodes ASVspoof file
layouts or naming conventions. A concrete dataset is described entirely
by a config mapping, loaded from a JSON file via :func:`load_config`.

Expected config shape (all fields required unless noted)::

    {
        "name": "asvspoof2019-la",
        "root": "data/asvspoof2019/LA",
        "audio_extensions": [".flac", ".wav"],
        "splits": {
            "train": {
                "manifest": "data/asvspoof2019/LA/train_manifest.json",
                "labels": "data/asvspoof2019/LA/train_labels.json"
            },
            "dev":   { ... },
            "eval":  { ... }
        },
        "label_overrides": {            // optional
            "some/ambiguous/file.flac": "bonafide"
        }
    }

``manifest`` / ``labels`` are opaque string references for now — the
placeholder loader treats them as opaque keys, and a future concrete
loader decides what they point at (e.g. JSON/CSV files under ``root``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .labels import Label, normalize_label

__all__ = ["DatasetConfig", "SplitConfig", "load_config"]

_SPLITS_KEY = "splits"


@dataclass(frozen=True)
class SplitConfig:
    """Paths describing one dataset split (still dataset-agnostic)."""

    name: str
    manifest: str
    labels: str


@dataclass(frozen=True)
class DatasetConfig:
    """Validated, dataset-agnostic description of an audio dataset."""

    name: str
    root: Path
    audio_extensions: tuple[str, ...]
    splits: Mapping[str, SplitConfig]
    label_overrides: Mapping[str, Label] = field(default_factory=dict)

    def split(self, name: str) -> SplitConfig:
        """Return the named split or raise a clear KeyError."""
        try:
            return self.splits[name]
        except KeyError:
            known = ", ".join(sorted(self.splits)) or "<none defined>"
            raise KeyError(
                f"Unknown split {name!r} in dataset config {self.name!r}; "
                f"defined splits: {known}"
            ) from None


def _parse_split(name: str, raw: Any) -> SplitConfig:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Split {name!r} must be an object, got {type(raw).__name__}")
    missing = [key for key in ("manifest", "labels") if key not in raw]
    if missing:
        raise ValueError(f"Split {name!r} is missing required keys: {missing}")
    return SplitConfig(name=name, manifest=raw["manifest"], labels=raw["labels"])


def parse_config(raw: Mapping[str, Any]) -> DatasetConfig:
    """Validate an in-memory mapping into a :class:`DatasetConfig`."""
    missing = [key for key in ("name", "root", "audio_extensions", _SPLITS_KEY) if key not in raw]
    if missing:
        raise ValueError(f"Dataset config is missing required keys: {missing}")

    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a non-empty string")

    root = Path(str(raw["root"])).expanduser()

    raw_extensions = raw["audio_extensions"]
    if not isinstance(raw_extensions, (list, tuple)) or not raw_extensions:
        raise ValueError("'audio_extensions' must be a non-empty list like [' .flac', '.wav']")
    audio_extensions = tuple(str(ext).lower() for ext in raw_extensions)

    raw_splits = raw[_SPLITS_KEY]
    if not isinstance(raw_splits, Mapping) or not raw_splits:
        raise ValueError("'splits' must be a non-empty object")
    splits = {split_name: _parse_split(split_name, value) for split_name, value in raw_splits.items()}

    raw_overrides = raw.get("label_overrides") or {}
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("'label_overrides' must be an object mapping file keys to labels")
    label_overrides = {key: normalize_label(value) for key, value in raw_overrides.items()}

    return DatasetConfig(
        name=name,
        root=root,
        audio_extensions=audio_extensions,
        splits=splits,
        label_overrides=label_overrides,
    )


def load_config(path: str | Path) -> DatasetConfig:
    """Load and validate a dataset config from a JSON file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_config(raw)
