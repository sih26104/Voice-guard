"""Dataset-agnostic audio loading, preprocessing and validation.

This module owns the *audio* half of the pipeline (the dataset loaders
in this package stay separate and never preprocess audio themselves).
It knows nothing about any specific dataset: no filenames, no folder
layouts, no labels.

Pipeline contract (all functions are pure with respect to their inputs):

    load_audio(path)                     -> AudioClip (mono, float32, native SR)
    AudioClip.resampled(target_sr)       -> AudioClip (mono, float32, target SR)
    AudioClip.validated(min_s, max_s)    -> AudioClip (raises if out of range)
    preprocess(path_or_array, ...)       -> AudioClip (mono, float32, 16 kHz, valid)

The single consistent representation produced for downstream consumers
(feature extraction, model input) is :class:`AudioClip`: a 1-D float32
mono waveform plus its integer sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np

__all__ = [
    "TARGET_SAMPLE_RATE",
    "DEFAULT_MIN_DURATION_S",
    "DEFAULT_MAX_DURATION_S",
    "AudioClip",
    "AudioValidationError",
    "DurationValidationError",
    "load_audio",
    "to_mono",
    "resample",
    "validate_duration",
    "preprocess",
]

#: Canonical sample rate every processed clip is resampled to.
TARGET_SAMPLE_RATE: int = 16_000

#: Default minimum accepted duration in seconds (None disables the check).
DEFAULT_MIN_DURATION_S: float = 0.5

#: Default maximum accepted duration in seconds (None disables the check).
DEFAULT_MAX_DURATION_S: float = 30.0

_ArrayLike = Union[np.ndarray, "list[float]"]


class AudioValidationError(ValueError):
    """Raised when audio fails a structural or numeric validation check."""


class DurationValidationError(AudioValidationError):
    """Raised when a clip's duration falls outside the accepted range."""


@dataclass(frozen=True)
class AudioClip:
    """The canonical VoiceGuard audio representation.

    Invariants (enforced by :meth:`from_array`):
      * ``samples`` is a 1-D :class:`numpy.ndarray` of float32 in [-1, 1]
      * ``sample_rate`` is a positive int
      * audio is mono by the time it is wrapped here
    """

    samples: np.ndarray
    sample_rate: int

    @classmethod
    def from_array(cls, samples: _ArrayLike, sample_rate: int) -> "AudioClip":
        """Build and validate a clip from a raw mono waveform.

        Raises:
            AudioValidationError: if the waveform is empty, not 1-D,
                non-finite, or the sample rate is not a positive int.
        """
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1:
            raise AudioValidationError(
                f"Expected a 1-D mono waveform, got ndim={array.ndim}; "
                "collapse channels first with to_mono()."
            )
        if array.size == 0:
            raise AudioValidationError("Waveform is empty (0 samples).")
        if not np.all(np.isfinite(array)):
            raise AudioValidationError("Waveform contains NaN or infinite values.")
        rate = int(sample_rate)
        if rate <= 0:
            raise AudioValidationError(f"Sample rate must be a positive int, got {sample_rate!r}.")
        return cls(samples=array, sample_rate=rate)

    @property
    def num_samples(self) -> int:
        """Number of samples in the waveform."""
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        """Duration in seconds (``num_samples / sample_rate``)."""
        return self.num_samples / self.sample_rate

    def resampled(self, target_sr: int) -> "AudioClip":
        """Return a copy resampled to ``target_sr`` (no-op if already there)."""
        return AudioClip.from_array(
            resample(self.samples, orig_sr=self.sample_rate, target_sr=target_sr),
            sample_rate=target_sr,
        )

    def validated(
        self,
        min_duration_s: Optional[float] = DEFAULT_MIN_DURATION_S,
        max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S,
    ) -> "AudioClip":
        """Return self after a duration range check.

        Raises:
            DurationValidationError: if duration is outside the range.
        """
        validate_duration(self, min_duration_s=min_duration_s, max_duration_s=max_duration_s)
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AudioClip(num_samples={self.num_samples}, "
            f"sample_rate={self.sample_rate}, duration={self.duration:.3f}s)"
        )


def to_mono(samples: _ArrayLike) -> np.ndarray:
    """Collapse a multi-channel waveform to mono by averaging channels.

    Accepts shape ``(channels, n_samples)`` (channels first, matching
    ``librosa`` conventions) or an already-1-D mono signal. Returns a
    new 1-D float32 array; a mono input is returned as a copy.
    """
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        return array.copy()
    if array.ndim == 2:
        return librosa.to_mono(array)
    raise AudioValidationError(
        f"Expected 1-D mono or 2-D (channels, samples) input, got ndim={array.ndim}."
    )


def resample(samples: _ArrayLike, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono waveform from ``orig_sr`` to ``target_sr``.

    Uses a high-quality SOXR resampler (librosa's ``res_type='soxr_hq'``).
    Returns float32; length scales by ``target_sr / orig_sr``.
    """
    if int(orig_sr) <= 0 or int(target_sr) <= 0:
        raise AudioValidationError(
            f"Sample rates must be positive, got orig_sr={orig_sr!r}, target_sr={target_sr!r}."
        )
    array = np.asarray(samples, dtype=np.float32)
    if int(orig_sr) == int(target_sr):
        return array.copy()
    return librosa.resample(
        y=array, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq"
    ).astype(np.float32)


def validate_duration(
    clip: AudioClip,
    min_duration_s: Optional[float] = DEFAULT_MIN_DURATION_S,
    max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S,
) -> float:
    """Validate a clip's duration against an inclusive range.

    ``None`` disables the respective bound. Returns the duration on
    success.

    Raises:
        DurationValidationError: if duration is out of range.
    """
    duration = clip.duration
    if min_duration_s is not None and duration < min_duration_s:
        raise DurationValidationError(
            f"Clip too short: {duration:.3f}s < min {min_duration_s:.3f}s."
        )
    if max_duration_s is not None and duration > max_duration_s:
        raise DurationValidationError(
            f"Clip too long: {duration:.3f}s > max {max_duration_s:.3f}s."
        )
    return duration


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> AudioClip:
    """Load an audio file as a :class:`AudioClip`.

    Args:
        path: Path to any format readable by ``soundfile`` (wav, flac,
            ogg, ...). Never dataset-specific; the caller decides what
            file this is.
        sample_rate: Keep the file's native rate when ``None`` (default);
            otherwise resample to this rate while loading.
        mono: Downmix to mono (default True).

    Returns:
        AudioClip with a 1-D float32 waveform.

    Raises:
        AudioValidationError: wrapping load failures (missing file,
            unreadable/undecodable audio, empty result) so callers can
            catch one exception type.
    """
    file_path = Path(path)
    try:
        samples, sr = librosa.load(
            path=str(file_path),
            sr=sample_rate,
            mono=mono,
            dtype=np.float32,
        )
    except Exception as exc:  # noqa: BLE001 - normalize third-party errors
        raise AudioValidationError(
            f"Failed to load audio file {str(file_path)!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return AudioClip.from_array(samples, sample_rate=int(sr))


def preprocess(
    source: Union[str, Path, np.ndarray],
    sample_rate: int = TARGET_SAMPLE_RATE,
    source_sample_rate: Optional[int] = None,
    min_duration_s: Optional[float] = DEFAULT_MIN_DURATION_S,
    max_duration_s: Optional[float] = DEFAULT_MAX_DURATION_S,
) -> AudioClip:
    """Full preprocessing path to the canonical clip representation.

    Loads (when given a path), collapses to mono, resamples to
    ``sample_rate`` (default 16 kHz), and validates duration.

    Args:
        source: Audio file path, or an already-loaded waveform. For a
            multi-channel array use shape ``(channels, samples)``; for a
            mono array pass 1-D.
        sample_rate: Target sample rate (default 16 kHz).
        source_sample_rate: Required when ``source`` is a raw array.
            Ignored for file paths (their rate is read from the file).
        min_duration_s / max_duration_s: Duration bounds; ``None``
            disables a bound.

    Returns:
        AudioClip: mono, float32, resampled, duration-validated.

    Raises:
        AudioValidationError: load/structural failures or a missing
            ``source_sample_rate`` for array input.
        DurationValidationError: duration out of range.
    """
    if isinstance(source, (str, Path)):
        clip = load_audio(source)
    else:
        if source_sample_rate is None:
            raise AudioValidationError(
                "source_sample_rate is required when preprocessing a raw array."
            )
        clip = AudioClip.from_array(to_mono(source), sample_rate=int(source_sample_rate))

    mono_clip = AudioClip.from_array(to_mono(clip.samples), sample_rate=clip.sample_rate)
    target = mono_clip.resampled(sample_rate)
    return target.validated(min_duration_s=min_duration_s, max_duration_s=max_duration_s)
