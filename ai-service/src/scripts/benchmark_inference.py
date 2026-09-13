import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset.audio import preprocess
from model import load_voiceguard_model
from model.classifier import predict
from scripts.common import load_checkpoint

wav = ROOT.parent / "benchmark_test.wav"
checkpoint = ROOT / "models" / "trial" / "best_head.pt"

print("WAV:", wav)
print("Checkpoint:", checkpoint)

if not wav.is_file():
    raise FileNotFoundError(f"WAV not found: {wav}")

if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

model, _, _ = load_checkpoint(
    checkpoint,
    lambda name: load_voiceguard_model(encoder_name=name)
)

model.eval()

clip = preprocess(wav)

waveform = np.asarray(clip.samples, dtype=np.float32)

target_samples = 16000 * 3

if len(waveform) >= target_samples:
    waveform = waveform[:target_samples]
else:
    waveform = np.pad(
        waveform,
        (0, target_samples - len(waveform))
    )

print(f"Benchmark duration: {len(waveform) / 16000:.3f} seconds")

# Warm up the model
for _ in range(2):
    predict(model, waveform)

times = []
result = None

for i in range(5):
    start = time.perf_counter()
    result = predict(model, waveform)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"Run {i + 1}: {elapsed:.2f} ms")

median = float(np.median(times))

print()
print("=" * 50)
print("RESULT")
print("=" * 50)
print(f"Minimum model time  : {min(times):.2f} ms")
print(f"Median model time   : {median:.2f} ms")
print(f"Maximum model time  : {max(times):.2f} ms")
print(f"Label               : {result.label}")
print(f"Spoof probability   : {result.spoof_probability:.4f}")
print(f"Bonafide probability: {result.probabilities['bonafide']:.4f}")
print()
print("3-second target: <= 3000 ms")

if median <= 3000:
    print("STATUS: FAST ENOUGH")
else:
    print("STATUS: NEEDS OPTIMIZATION")
