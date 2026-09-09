"""VoiceGuard spoof-detection model.

Wraps a frozen pretrained Wav2Vec2 encoder with a small trainable
classification head. Input contract: the canonical waveform produced by
``dataset.audio`` / ``MlaadDataset`` — a 1-D float32 mono waveform at
16 kHz (an ``(batch, time)`` batch of such waveforms also works).

Design:
    * Raw ``Wav2Vec2Model`` encoder (not ``...ForSequenceClassification``):
      transformers 5's ``SequenceClassifierOutput`` no longer exposes
      ``last_hidden_state``, and an explicit head is cleaner to train and
      inspect anyway. Pretrained weights load with zero warnings this way.
    * The encoder is frozen by default — only the small head trains,
      keeping CPU training feasible on a 1,000-sample prototype.
    * Mean-pooling over time yields a fixed-size embedding for any clip
      length. For padded batches the pooling is masked with the encoder's
      own frame-level attention mask, so variable-duration MLAAD clips
      contribute only their real frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from dataset.labels import Label

from .config import (
    DEFAULT_DROPOUT,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MODEL_SAMPLE_RATE,
    PRETRAINED_MODEL_NAME,
)
from .processor import input_should_normalize, normalize_waveform

__all__ = [
    "VoiceGuardClassifier",
    "SpoofPrediction",
    "load_voiceguard_model",
    "predict",
    "prepare_waveform",
    "prepare_batch",
]

WaveformLike = Union[np.ndarray, "list[float]", Tensor]


@dataclass(frozen=True)
class SpoofPrediction:
    """Single-sample inference result, ready for an API layer."""

    label: str                 # "bonafide" | "spoof" (canonical enum value)
    confidence: float          # probability of the predicted label
    probabilities: dict        # {"bonafide": p1, "spoof": p2}, sums to 1.0
    spoof_probability: float   # convenience copy of probabilities["spoof"]


class VoiceGuardClassifier(nn.Module):
    """Frozen Wav2Vec2 encoder + trainable 2-class head.

    Args:
        encoder_name: Hub id (or local path) of the pretrained checkpoint.
        dropout: Dropout used in the small head.
        freeze_encoder: Freeze all encoder weights (default True).
    """

    def __init__(
        self,
        encoder_name: str = PRETRAINED_MODEL_NAME,
        dropout: float = DEFAULT_DROPOUT,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.config = Wav2Vec2Config.from_pretrained(encoder_name)
        self.encoder = Wav2Vec2Model.from_pretrained(encoder_name)
        self.encoder.eval()  # encoder is inference-only while frozen
        self.normalize_input = input_should_normalize()
        self._encoder_frozen = bool(freeze_encoder)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            # Frozen encoder = inference-only component: make sure gradient
            # checkpointing is off (it exists to save activation memory for
            # backward passes the encoder will never run) and pin eval mode.
            self.encoder.gradient_checkpointing_disable()
            self.encoder.eval()

        hidden = self.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(Label)),
        )

    @classmethod
    def with_tiny_config(
        cls,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> "VoiceGuardClassifier":
        """Random-weights tiny model for tests (no download, CPU-fast).

        Mirrors the real architecture (encoder + head) at ~30k params so
        unit tests exercise shape/probability logic in milliseconds.
        """
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.encoder_name = "tiny-random"
        model.config = Wav2Vec2Config(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            intermediate_size=hidden_size * 2,
            conv_dim=(hidden_size, hidden_size),
            conv_stride=(5, 2),
            conv_kernel=(10, 3),
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=4,
            do_stable_layer_norm=False,
        )
        model.encoder = Wav2Vec2Model(model.config)
        model.encoder.eval()
        model.normalize_input = False  # tests must not touch the network
        model.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(Label)),
        )
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        model._encoder_frozen = True
        model.encoder.gradient_checkpointing_disable()
        model.encoder.eval()
        return model

    def train(self, mode: bool = True) -> "VoiceGuardClassifier":
        """Set training mode; the frozen encoder is pinned to eval.

        ``nn.Module.train(True)`` would otherwise re-enable dropout inside
        the frozen encoder during head training. The encoder is an
        inference-only feature extractor here, so it stays in eval mode
        regardless of ``mode``.
        """
        super().train(mode)
        if getattr(self, "_encoder_frozen", False):
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: WaveformLike,
        waveform_lengths: Optional[Union[Tensor, "list[int]", np.ndarray]] = None,
    ) -> Tensor:
        """Waveform(s) -> class logits ``(batch, 2)``.

        Accepts a 1-D waveform ``(time,)`` or a batch ``(batch, time)``.
        For padded batches pass ``waveform_lengths`` (real sample count
        per item); a mask is derived from them and pooling averages real
        frames only, so variable-length audio is handled correctly.
        Without padding the batch is processed as-is.
        """
        values, lengths = prepare_batch(waveform)
        if waveform_lengths is not None:
            lengths = torch.as_tensor(np.asarray(waveform_lengths), dtype=torch.long)
        attention_mask = None
        if lengths is not None and int(lengths.min().item()) < values.shape[1]:
            # Padded batch (any item shorter than the batch width):
            # (batch, time) bool mask, True on real samples.
            positions = torch.arange(values.shape[1]).unsqueeze(0)
            attention_mask = positions < lengths.clamp(min=1).unsqueeze(1)
        if self.normalize_input:
            values = normalize_waveform(values)
        if self._encoder_frozen:
            # Frozen encoder: skip building the autograd graph entirely
            # (no encoder activations retained, no encoder gradients).
            with torch.no_grad():
                outputs = self.encoder(input_values=values, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_values=values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, frames, hidden)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            # Downsample the mask with the encoder's own helper so pooling
            # averages real frames only (correct for variable lengths).
            frame_mask = self.encoder._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
            mask = frame_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

    def probabilities(
        self,
        waveform: WaveformLike,
        waveform_lengths: Optional[Union[Tensor, "list[int]", np.ndarray]] = None,
    ) -> Tensor:
        """Logits -> softmax probabilities ``(batch, 2)``."""
        logits = self.forward(waveform, waveform_lengths=waveform_lengths)
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Training bookkeeping (no training here — helpers for later)
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list:
        """Parameters that will receive gradients during head training."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_counts(self) -> dict:
        """Total vs trainable parameter counts (for logging/READMEs)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {"total": total, "trainable": trainable}


def load_voiceguard_model(
    encoder_name: str = PRETRAINED_MODEL_NAME,
    **kwargs,
) -> VoiceGuardClassifier:
    """Load the model in eval mode, ready for CPU inference."""
    model = VoiceGuardClassifier(encoder_name=encoder_name, **kwargs)
    model.eval()
    return model


def prepare_waveform(waveform: WaveformLike) -> Tensor:
    """Normalize arbitrary waveform input to ``(batch, time)`` float32.

    Validates the model contract: finite values, non-empty, 1-D or 2-D.
    A 16 kHz single-channel clip at 1 s is ``(16000,)`` — exactly what
    ``MlaadSample.waveform()`` and ``dataset.audio.preprocess`` produce.
    """
    values, _ = prepare_batch(waveform)
    return values


def prepare_batch(waveform: WaveformLike) -> tuple:
    """Waveform input -> ``((batch, time) tensor, per-item lengths)``.

    Lengths is ``None`` for a single 1-D clip (no padding concept) and a
    full-width tensor for a 2-D batch; callers that know the real
    per-item lengths (training collation) pass them explicitly via
    :meth:`VoiceGuardClassifier.forward`.
    """
    if isinstance(waveform, Tensor):
        input_ndim = waveform.dim()
        array = waveform.detach().cpu().numpy()
    else:
        input_ndim = np.asarray(waveform).ndim
        array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]  # (time,) -> (1, time)
    if array.ndim != 2:
        raise ValueError(
            f"Expected 1-D waveform or 2-D (batch, time) batch, got ndim={input_ndim}."
        )
    if array.shape[1] == 0:
        raise ValueError("Waveform is empty (0 samples).")
    if not np.all(np.isfinite(array)):
        raise ValueError("Waveform contains NaN or infinite values.")
    values = torch.from_numpy(array.astype(np.float32))
    if input_ndim == 1:
        return values, None
    lengths = torch.full((values.shape[0],), values.shape[1], dtype=torch.long)
    return values, lengths


@torch.no_grad()
def predict(
    model: VoiceGuardClassifier,
    waveform: WaveformLike,
) -> SpoofPrediction:
    """Clean single-sample inference interface for the future FastAPI layer.

    Args:
        model: A :class:`VoiceGuardClassifier` (eval mode expected).
        waveform: 1-D float32 mono waveform at 16 kHz — the exact output
            of ``MlaadSample.waveform()`` / ``dataset.audio.preprocess``.

    Returns:
        :class:`SpoofPrediction` with the predicted label, confidence and
        full BONAFIDE/SPOOF probability distribution (finite, summing to 1).

    Raises:
        ValueError: malformed/empty/non-finite waveform input.
    """
    probs = model.probabilities(waveform)[0]
    if not torch.isfinite(probs).all():
        raise ValueError("Model produced non-finite probabilities.")
    items = {label.value: float(probs[idx]) for label, idx in LABEL_TO_ID.items()}
    predicted_label = max(items, key=items.get)
    return SpoofPrediction(
        label=predicted_label,
        confidence=items[predicted_label],
        probabilities=items,
        spoof_probability=items[Label.SPOOF.value],
    )
