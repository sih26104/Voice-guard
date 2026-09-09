# VoiceGuard AI Service

BONAFIDE vs SPOOF voice-classification prototype for SIH 2026 PS 26104.
Python 3.13, CPU-only PyTorch — no GPU required anywhere in this pipeline.

## Layout

```
src/dataset/    MLAAD loader + audio preprocessing (see src/dataset/README.md)
src/model/      frozen Wav2Vec2 encoder + trainable classification head
src/scripts/    train_head.py / evaluate.py CLIs
tests/          unit tests (synthetic audio + temp fixtures; real data read-only)
```

## Model: frozen encoder + trainable head

`facebook/wav2vec2-base` (95M params, 363 MB weights) provides the speech
representation; it is **frozen** (`requires_grad=False`, eval mode). Only the
small head trains — LayerNorm → 768→384 → GELU → 2 (~298k trainable params) —
which is what makes CPU training feasible on the 1,000-file subset. Input is
the canonical 16 kHz mono float32 waveform from `MlaadSample.waveform()`;
variable-length batches are padded and mean-pooled with a real-frames mask.

Label mapping reuses `dataset.labels.Label` (bonafide=0, spoof=1) — one
taxonomy across data, training, metrics and the future API.

This is a **prototype baseline, not production performance**.

## Training (PowerShell)

```powershell
cd ai-service
.venv\Scripts\python.exe -m src.scripts.train_head `
    --dataset ..\data\mlaad_subset `
    --train-split train `
    --validation-split validation `
    --epochs 3 `
    --batch-size 4 `
    --learning-rate 1e-4 `
    --output-dir models\head_only
```

Per epoch it reports train loss, validation loss, accuracy, precision, recall,
F1 and saves a checkpoint whenever validation F1 (or loss, with
`--best-metric loss`) improves. Seeded (default 26104) for reproducibility.

**CPU optimizations**: training clips longer than `--max-train-seconds`
(default 5 s) are deterministically cropped (stable per-file hash — same
segment every epoch; original WAVs untouched). Validation/test always run
full-length and uncropped. The frozen encoder never builds an autograd graph
(`torch.no_grad` forward, gradient checkpointing off, eval mode pinned even
inside `model.train()`), which roughly halves per-step memory and skips all
encoder backward work.

## Evaluation (PowerShell)

```powershell
cd ai-service
.venv\Scripts\python.exe -m src.scripts.evaluate `
    --checkpoint models\head_only\best_head.pt `
    --dataset ..\data\mlaad_subset `
    --split test `
    --output models\head_only\eval_test.json
```

## Expected output artifacts

| File | Contents |
|---|---|
| `models\head_only\best_head.pt` | head weights + encoder name + training config + label mapping + best-epoch metrics |
| `models\head_only\training_summary.json` | per-epoch history + best validation metrics |
| `models\head_only\eval_test.json` | test-split metrics report (with `--output`) |

Checkpoints contain **only head weights** — the encoder is re-downloaded from
its Hub id (cached after first use), so `.pt` files stay small (~1 MB).

## Limitations

- **CPU speed**: ~0.5–0.9 s per clip forward; a full 3-split pass is ~10 min.
  Training is capped at 5-second crops (~0.2–0.7 s per clip), keeping an
  epoch over 700 files in the low-tens-of-minutes range on CPU. Batch size 4
  and head-only training are the practical CPU regime; full fine-tuning is
  not realistic on this machine.
- **Test set is not group-aware**: the 1,000-file subset puts the *same TTS
  systems* in train and test. Evaluation therefore measures the pipeline, not
  unseen-TTS generalization. Treat all reported numbers as prototype baseline
  results, never production claims.
- `data\` is Git-ignored; training/evaluation read it locally only.
