/**
 * Upload analysis pipeline: windowed, sequential, real-data-only.
 *
 * A whole uploaded WAV is decoded in the browser (Web Audio API), converted
 * to mono at 16 kHz, split into sequential 5-second windows and every usable
 * window is analyzed by the EXISTING `analyzeAudio()` endpoint client
 * (POST /api/analyze) — one request per window, strictly sequential so the
 * backend is never overloaded. No backend change, no new endpoint.
 *
 * Aggregation is arithmetic: mean spoofProbability across analyzed windows,
 * riskScore = round(mean * 100), fixed risk bands, label by the 0.5
 * threshold, alert only for HIGH/CRITICAL, and summed inference time.
 * Every spoofProbability is the real value returned by the AI service —
 * nothing is faked, interpolated or defaulted.
 */

import { analyzeAudio } from './api';

const TARGET_SAMPLE_RATE = 16000; // Hz — matches the mic pipeline's WAV format
const WINDOW_SECONDS = 5; // sequential analysis window length
const MIN_FINAL_WINDOW_SECONDS = 3; // trailing partial window analyzed only if >= this

/** Risk bands (exact thresholds required by the product spec). */
function riskLevelFor(score) {
  if (score <= 40) return 'LOW';
  if (score <= 60) return 'MEDIUM';
  if (score <= 80) return 'HIGH';
  return 'CRITICAL';
}

/** Throw with a clear, user-facing message. */
function fail(message) {
  throw new Error(message);
}

/**
 * Decode the uploaded file with the browser Web Audio API.
 * Works for WAV (and anything else AudioContext can decode, though the
 * dashboard restricts selection to .wav).
 */
async function decodeToAudioBuffer(file) {
  if (typeof window === 'undefined') {
    fail('Audio decoding requires a browser environment.');
  }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    fail('This browser does not support the Web Audio API.');
  }

  let arrayBuffer;
  try {
    arrayBuffer = await file.arrayBuffer();
  } catch (err) {
    fail('The selected file could not be read. Please choose a valid WAV file.');
  }

  const context = new AudioContextClass();
  let buffer;
  try {
    buffer = await context.decodeAudioData(arrayBuffer);
  } catch (err) {
    fail('This file could not be decoded as audio. Please choose a valid WAV file.');
  } finally {
    // The context is only needed for decoding; release it immediately.
    if (context.close) context.close().catch(() => {});
  }

  if (!buffer || !Number.isFinite(buffer.duration) || buffer.duration <= 0 || buffer.length === 0) {
    fail('The selected file contains no audio data.');
  }
  return buffer;
}

/**
 * Mix every channel down to mono (simple arithmetic mean per sample frame).
 * Mono input returns a copy — callers may safely slice it.
 */
function toMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  if (channels <= 0) fail('The selected file contains no audio channels.');

  const length = audioBuffer.length;
  const mono = new Float32Array(length);

  for (let ch = 0; ch < channels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += data[i];
    }
  }
  if (channels > 1) {
    for (let i = 0; i < length; i++) {
      mono[i] /= channels;
    }
  }
  return mono;
}

/**
 * Linear-interpolation resampler to 16 kHz (deterministic, dependency-free,
 * same approach the mic pipeline uses for its WAV chunks).
 */
function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio = toRate / fromRate;
  const outLength = Math.max(1, Math.round(samples.length * ratio));
  const out = new Float32Array(outLength);
  const step = (samples.length - 1) / (outLength - 1 || 1);
  for (let i = 0; i < outLength; i++) {
    const pos = i * step;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = samples[idx];
    const b = samples[Math.min(idx + 1, samples.length - 1)];
    out[i] = a + (b - a) * frac;
  }
  return out;
}

/**
 * Encode a Float32 mono segment as a PCM16 WAV Blob at the given sample rate.
 * (Same header layout as the mic pipeline's encoder.)
 */
function encodeWavBlob(samples, sampleRate) {
  const bytesPerSample = 2; // PCM16
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Analyze an uploaded WAV file in sequential 5-second windows.
 *
 * @param {File} file WAV file selected on the upload dashboard
 * @returns {Promise<object>} result compatible with mapApiResponse()'s
 *   `api` input shape (spoofProbability, riskScore, riskLevel, alert, label,
 *   duration_seconds, sample_rate, inference_time_ms, message)
 */
export async function analyzeUploadedAudio(file) {
  if (!file) fail('No file provided for analysis.');

  const fileName = typeof file.name === 'string' ? file.name : 'upload.wav';

  // 1-2. Decode via Web Audio API (clear errors for invalid/empty input).
  const audioBuffer = await decodeToAudioBuffer(file);

  // 3-4. Mono + 16 kHz.
  const mono = toMono(audioBuffer);
  const pcm = resample(mono, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
  const samplesPerWindow = WINDOW_SECONDS * TARGET_SAMPLE_RATE;
  const totalWindows = Math.ceil(pcm.length / samplesPerWindow);

  // Collect usable windows: full 5 s windows, plus the trailing partial one
  // only when it is at least 3 s (shorter trailing audio is ignored).
  const windows = [];
  for (let w = 0; w < totalWindows; w++) {
    const start = w * samplesPerWindow;
    const end = Math.min(start + samplesPerWindow, pcm.length);
    const lengthSeconds = (end - start) / TARGET_SAMPLE_RATE;
    if (w === totalWindows - 1 && lengthSeconds < MIN_FINAL_WINDOW_SECONDS) {
      break; // 10-11. ignore a final partial window shorter than 3 s
    }
    if (lengthSeconds <= 0) break;
    windows.push(pcm.subarray(start, end));
  }

  // 22. Nothing usable (e.g. a 1–2 s file).
  if (windows.length === 0) {
    fail(
      'No usable audio: the file must contain at least 3 seconds of audio to be analyzed.',
    );
  }

  // 9. Analyze sequentially — one request at a time, real backend responses.
  let sumSpoofProbability = 0;
  let totalInferenceTimeMs = 0;
  let analyzedCount = 0;

  for (const windowSamples of windows) {
    const wavBlob = encodeWavBlob(windowSamples, TARGET_SAMPLE_RATE);
    const api = await analyzeAudio(wavBlob, fileName); // existing endpoint client
    sumSpoofProbability += api.spoofProbability;
    if (typeof api.inference_time_ms === 'number') {
      totalInferenceTimeMs += api.inference_time_ms; // 18. summed, not averaged
    }
    analyzedCount += 1;
  }

  // 13. Arithmetic mean of the real per-window probabilities.
  const meanSpoofProbability = sumSpoofProbability / analyzedCount;

  // 14-15. Risk score + fixed bands.
  const riskScore = Math.round(meanSpoofProbability * 100);
  const riskLevel = riskLevelFor(riskScore);

  // 16-17. Aggregate label + alert.
  const label = meanSpoofProbability >= 0.5 ? 'SPOOF' : 'BONAFIDE';
  const alert = riskLevel === 'HIGH' || riskLevel === 'CRITICAL';

  // 19. Compatible with the existing mapApiResponse() flow.
  return {
    spoofProbability: meanSpoofProbability,
    riskScore,
    riskLevel,
    alert,
    label,
    duration_seconds: audioBuffer.duration,
    sample_rate: TARGET_SAMPLE_RATE,
    inference_time_ms: totalInferenceTimeMs,
    message:
      windows.length === 1
        ? 'Analyzed as a single 5-second window.'
        : `Analyzed in ${windows.length} sequential 5-second windows (mean of per-window results).`,
    fileName,
  };
}
