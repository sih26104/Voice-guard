"""Placeholder dataset loader interface for VoiceGuard.

This module defines *only* the contract that concrete dataset loaders
will implement later. It intentionally performs no filesystem discovery
and makes no assumptions about filenames, folder layout, or metadata
formats — concrete structure knowledge is added by future subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .config import DatasetConfig
from .labels import Label

__all__ = ["AudioSample", "BaseAudioDataset", "LoaderNotImplementedError"]


@dataclass(frozen=True)
class AudioSample:
    """A single labelled audio item.

    ``audio_path`` and ``label`` may be ``None`` while a concrete loader
    is still resolving the sample; the dataset-agnostic core never
    guesses either of them.
    """

    key: str
    audio_path: Optional[Path] = None
    label: Optional[Label] = None
    split: str = "train"
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class LoaderNotImplementedError(NotImplementedError):
    """Raised by placeholder loader methods awaiting a concrete implementation."""


class BaseAudioDataset(ABC):
    """Abstract, dataset-agnostic interface for BONAFIDE vs SPOOF data.

    Concrete loaders (e.g. a future ASVspoof loader) subclass this and
    implement :meth:`__len__` and :meth:`__iter__`. The base class only
    stores configuration — it never scans directories itself.
    """

    def __init__(self, config: DatasetConfig, split: str = "train") -> None:
        self._config = config
        self._split_name = split
        self._split = config.split(split)  # validates the split name eagerly

    @property
    def config(self) -> DatasetConfig:
        return self._config

    @property
    def split_name(self) -> str:
        return self._split_name

    @property
    def split(self):
        return self._split

    @property
    def audio_extensions(self) -> tuple[str, ...]:
        return self._config.audio_extensions

    def label_for(self, key: str) -> Optional[Label]:
        """Resolve a label for a sample key (overrides only, no guessing).

        Returns the configured override if present, otherwise ``None``:
        the placeholder layer has no other label source.
        """
        return self._config.label_overrides.get(key)

    def build_sample(self, key: str) -> AudioSample:
        """Construct a sample for ``key`` using only known config info."""
        return AudioSample(
            key=key,
            label=self.label_for(key),
            split=self._split_name,
        )

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples in this split. Must be implemented."""

    @abstractmethod
    def __iter__(self) -> Iterator[AudioSample]:
        """Iterate over resolved :class:`AudioSample` items. Must be implemented."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dataset={self._config.name!r}, "
            f"split={self._split_name!r})"
        )
