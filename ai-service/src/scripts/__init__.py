"""VoiceGuard training/evaluation scripts.

Run from ``ai-service/`` with the project venv::

    .venv/Scripts/python.exe -m src.scripts.train_head --help
    .venv/Scripts/python.exe -m src.scripts.evaluate --help
"""

from .common import (
    TrainingConfig,
    WaveformDataset,
    collate_variable_length,
    compute_metrics,
    evaluate_dataset,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    set_seeds,
)

__all__ = [
    "TrainingConfig",
    "WaveformDataset",
    "collate_variable_length",
    "compute_metrics",
    "evaluate_dataset",
    "evaluate_model",
    "load_checkpoint",
    "save_checkpoint",
    "set_seeds",
]
