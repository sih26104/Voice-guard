"""Canonical label taxonomy for VoiceGuard audio classification.

Only the two-class BONAFIDE vs SPOOF problem is defined. Keep the
serialized values lowercase-stable: they are the ground truth keys used
by config files, manifests, metrics and the API layer, so renaming a
value is a breaking change.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Label"]


class Label(str, Enum):
    """The two VoiceGuard classes."""

    BONAFIDE = "bonafide"
    SPOOF = "spoof"

    def __str__(self) -> str:
        return self.value


def normalize_label(raw: str) -> Label:
    """Parse a raw label string into a :class:`Label`.

    Accepts any casing. Raises ValueError for anything that is not a
    known label, so bad ground-truth data fails loudly instead of
    silently corrupting training/evaluation.
    """
    try:
        return Label(raw.strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"Unknown label {raw!r}; expected one of: "
            f"{[member.value for member in Label]}"
        ) from exc
