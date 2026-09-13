// frontend/src/utils/speechActivity.js

/**
 * Speech activity measurement.
 *
 * IMPORTANT:
 *
 * This module measures whether a chunk appears
 * to contain audible speech.
 *
 * It should NOT be used as a hard security gate
 * for the deepfake detector.
 *
 * The microphone recorder always sends the
 * 3-second chunk to the AI.
 */

export const SPEECH_GATE_DEFAULTS = {
  /**
   * Chunk-level RMS threshold.
   */
  rmsThreshold: 0.006,

  /**
   * Per-frame RMS threshold.
   */
  frameRmsThreshold: 0.006,

  /**
   * 100 ms frames.
   */
  frameSeconds: 0.1,

  /**
   * Only 3% of frames need to contain
   * audible energy.
   *
   * This is deliberately permissive.
   */
  minSpeechFrameRatio: 0.03,
};

/**
 * Calculate speech activity metrics.
 *
 * @param {Float32Array} samples
 * @param {number} sampleRate
 * @param {object} overrides
 *
 * @returns {{
 *   isSpeech: boolean,
 *   rms: number,
 *   peak: number,
 *   zcr: number,
 *   speechFrameRatio: number,
 *   frames: number
 * }}
 */
export function isSpeech(samples, sampleRate, overrides = {}) {
  const cfg = {
    ...SPEECH_GATE_DEFAULTS,
    ...overrides,
  };

  const n = samples ? samples.length : 0;

  if (n === 0 || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    return {
      isSpeech: false,
      rms: 0,
      peak: 0,
      zcr: 0,
      speechFrameRatio: 0,
      frames: 0,
    };
  }

  // ------------------------------------------------------------
  // Whole-chunk RMS and peak
  // ------------------------------------------------------------

  let sumSquares = 0;

  let peak = 0;

  for (let i = 0; i < n; i++) {
    const value = Number.isFinite(samples[i]) ? samples[i] : 0;

    sumSquares += value * value;

    peak = Math.max(peak, Math.abs(value));
  }

  const rms = Math.sqrt(sumSquares / n);

  // ------------------------------------------------------------
  // Zero-crossing rate
  // ------------------------------------------------------------

  let zeroCrossings = 0;

  for (let i = 1; i < n; i++) {
    const previous = samples[i - 1] || 0;

    const current = samples[i] || 0;

    if ((previous >= 0 && current < 0) || (previous < 0 && current >= 0)) {
      zeroCrossings++;
    }
  }

  const zcr = n > 1 ? zeroCrossings / (n - 1) : 0;

  // ------------------------------------------------------------
  // Frame-level RMS
  // ------------------------------------------------------------

  const frameSize = Math.max(1, Math.round(cfg.frameSeconds * sampleRate));

  let frames = 0;

  let speechFrames = 0;

  for (let start = 0; start < n; start += frameSize) {
    const end = Math.min(n, start + frameSize);

    const length = end - start;

    if (length <= 0) {
      continue;
    }

    let frameSumSquares = 0;

    for (let i = start; i < end; i++) {
      const value = Number.isFinite(samples[i]) ? samples[i] : 0;

      frameSumSquares += value * value;
    }

    const frameRms = Math.sqrt(frameSumSquares / length);

    frames++;

    if (frameRms >= cfg.frameRmsThreshold) {
      speechFrames++;
    }
  }

  const speechFrameRatio = frames > 0 ? speechFrames / frames : 0;

  /**
   * This is deliberately permissive.
   */
  const isSpeech =
    rms >= cfg.rmsThreshold && speechFrameRatio >= cfg.minSpeechFrameRatio;

  return {
    isSpeech,

    rms,

    peak,

    zcr,

    speechFrameRatio,

    frames,
  };
}

export default isSpeech;
