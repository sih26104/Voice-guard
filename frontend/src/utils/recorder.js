// frontend/src/utils/recorder.js

import { isSpeech } from "./speechActivity";

const CHUNK_SECONDS = 3;
const TARGET_SAMPLE_RATE = 16000;

const MAX_ANALYSIS_QUEUE = 3;

const SCRIPT_PROCESSOR_BUFFER_SIZE = 4096;

const MIC_PERMISSION_TIMEOUT_MS = 60000;
const WATCHDOG_INTERVAL_MS = 1000;
const MAX_IDLE_MS = 10000;

/**
 * Convert microphone/browser errors into readable messages.
 */
export function describeMicError(err) {
  if (!err) {
    return "Microphone error. Please check your microphone and try again.";
  }

  if (err.name === "NotAllowedError" || err.name === "SecurityError") {
    return (
      "Microphone permission denied. Allow microphone access in your browser settings, " +
      "then start monitoring again."
    );
  }

  if (err.name === "NotFoundError") {
    return "No microphone was found on this device.";
  }

  if (err.name === "NotReadableError") {
    return (
      "The microphone is already being used by another application. " +
      "Close other microphone applications and try again."
    );
  }

  if (err.name === "OverconstrainedError") {
    return (
      "The selected microphone does not support the requested audio settings. " +
      "Try another microphone or browser input."
    );
  }

  if (err.name === "AbortError") {
    return "Microphone access was interrupted. Please try again.";
  }

  if (err.name === "TypeError") {
    return "This browser does not support microphone capture.";
  }

  return (
    err.message ||
    "Microphone error. Please check your microphone and try again."
  );
}

/**
 * Keep a sample in the valid floating-point audio range.
 */
function clampSample(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(-1, Math.min(1, value));
}

/**
 * Convert one or more microphone channels into mono.
 */
function mixToMono(channelData) {
  if (!channelData || channelData.length === 0) {
    return new Float32Array(0);
  }

  if (channelData.length === 1) {
    return new Float32Array(channelData[0]);
  }

  const length = channelData[0].length;

  const mono = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    let sum = 0;

    for (let channel = 0; channel < channelData.length; channel++) {
      sum += channelData[channel][i] || 0;
    }

    mono[i] = sum / channelData.length;
  }

  return mono;
}

/**
 * Remove DC offset from microphone input.
 *
 * This prevents a microphone with a small DC bias from
 * affecting RMS measurements and the detector input.
 */
function removeDcOffset(samples) {
  if (!samples || samples.length === 0) {
    return new Float32Array(0);
  }

  let sum = 0;

  for (let i = 0; i < samples.length; i++) {
    sum += Number.isFinite(samples[i]) ? samples[i] : 0;
  }

  const mean = sum / samples.length;

  const output = new Float32Array(samples.length);

  for (let i = 0; i < samples.length; i++) {
    output[i] = clampSample(
      (Number.isFinite(samples[i]) ? samples[i] : 0) - mean,
    );
  }

  return output;
}

/**
 * Resample microphone audio to the model's required 16 kHz.
 *
 * The browser may capture at 44.1 kHz or 48 kHz.
 */
function resampleLinear(samples, sourceRate, targetRate) {
  if (!samples || samples.length === 0) {
    return new Float32Array(0);
  }

  if (
    !Number.isFinite(sourceRate) ||
    sourceRate <= 0 ||
    !Number.isFinite(targetRate) ||
    targetRate <= 0
  ) {
    throw new Error("Invalid microphone sample rate.");
  }

  if (sourceRate === targetRate) {
    return new Float32Array(samples);
  }

  const outputLength = Math.max(
    1,
    Math.round((samples.length * targetRate) / sourceRate),
  );

  const output = new Float32Array(outputLength);

  const ratio = sourceRate / targetRate;

  for (let i = 0; i < outputLength; i++) {
    const position = i * ratio;

    const left = Math.floor(position);

    const right = Math.min(left + 1, samples.length - 1);

    const fraction = position - left;

    const leftValue = samples[left] || 0;

    const rightValue = samples[right] || 0;

    output[i] = leftValue + (rightValue - leftValue) * fraction;
  }

  return output;
}

/**
 * Conservative microphone level normalization.
 *
 * The purpose is to make very quiet laptop microphones
 * more comparable to the audio level used during training.
 *
 * Maximum gain is limited to 3x.
 * Clipping is prevented.
 */
function normalizeSpeechLevel(samples) {
  if (!samples || samples.length === 0) {
    return new Float32Array(0);
  }

  let sumSquares = 0;
  let peak = 0;

  for (let i = 0; i < samples.length; i++) {
    const value = Number.isFinite(samples[i]) ? samples[i] : 0;

    sumSquares += value * value;

    peak = Math.max(peak, Math.abs(value));
  }

  const rms = Math.sqrt(sumSquares / samples.length);

  if (!Number.isFinite(rms) || rms < 0.0005) {
    return new Float32Array(samples);
  }

  /**
   * Approximately -24.4 dBFS.
   */
  const targetRms = 0.06;

  let gain = targetRms / rms;

  /**
   * Never apply extreme amplification.
   */
  gain = Math.min(gain, 3.0);

  /**
   * Maintain headroom.
   */
  if (peak > 0 && peak * gain > 0.95) {
    gain = 0.95 / peak;
  }

  const output = new Float32Array(samples.length);

  for (let i = 0; i < samples.length; i++) {
    output[i] = clampSample(samples[i] * gain);
  }

  return output;
}

/**
 * Encode Float32 audio as:
 *
 * PCM
 * mono
 * 16-bit
 * 16 kHz
 *
 * WAV.
 */
function encodeWav(samples, sampleRate) {
  if (!samples || samples.length === 0) {
    throw new Error("Cannot encode an empty audio chunk.");
  }

  const channels = 1;

  const bitsPerSample = 16;

  const bytesPerSample = bitsPerSample / 8;

  const dataSize = samples.length * channels * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);

  const view = new DataView(buffer);

  function writeString(offset, text) {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  /**
   * RIFF
   */
  writeString(0, "RIFF");

  view.setUint32(4, 36 + dataSize, true);

  writeString(8, "WAVE");

  /**
   * fmt
   */
  writeString(12, "fmt ");

  view.setUint32(16, 16, true);

  /**
   * PCM format.
   */
  view.setUint16(20, 1, true);

  /**
   * Mono.
   */
  view.setUint16(22, channels, true);

  /**
   * Sample rate.
   */
  view.setUint32(24, sampleRate, true);

  const byteRate = sampleRate * channels * bytesPerSample;

  view.setUint32(28, byteRate, true);

  const blockAlign = channels * bytesPerSample;

  view.setUint16(32, blockAlign, true);

  view.setUint16(34, bitsPerSample, true);

  /**
   * data
   */
  writeString(36, "data");

  view.setUint32(40, dataSize, true);

  /**
   * Float32 → PCM16.
   */
  let offset = 44;

  for (let i = 0; i < samples.length; i++) {
    const sample = clampSample(samples[i]);

    let pcm;

    if (sample < 0) {
      pcm = Math.round(sample * 32768);
    } else {
      pcm = Math.round(sample * 32767);
    }

    view.setInt16(offset, pcm, true);

    offset += 2;
  }

  return new Blob([buffer], {
    type: "audio/wav",
  });
}

/**
 * Microphone monitoring.
 *
 * Pipeline:
 *
 * Browser microphone
 *        ↓
 * Raw PCM
 *        ↓
 * Mono
 *        ↓
 * 3-second chunk
 *        ↓
 * Speech activity measurement
 *        ↓
 * DC correction
 *        ↓
 * Level normalization
 *        ↓
 * 16 kHz
 *        ↓
 * PCM16 WAV
 *        ↓
 * FIFO
 *        ↓
 * AI
 *
 * IMPORTANT:
 *
 * Speech activity is now INFORMATIONAL.
 * It does NOT block AI analysis.
 *
 * This prevents quiet human speech from being
 * incorrectly discarded by the energy gate.
 */
export class MicMonitor {
  constructor() {
    this.phase = "IDLE";

    this.active = false;

    this.stopping = false;

    this.sessionId = 0;

    this.chunkNumber = 0;

    this.stream = null;

    this.audioContext = null;

    this.source = null;

    this.processor = null;

    this.silentGain = null;

    this.inputSampleRate = TARGET_SAMPLE_RATE;

    this.sampleBuffer = [];

    this.sampleBufferLength = 0;

    this.queue = [];

    this.processing = false;

    this.watchdogTimer = null;

    this.lastAudioCallbackAt = 0;

    /**
     * React callbacks.
     */
    this.onStateChange = null;

    this.onChunk = null;

    this.onChunkDropped = null;

    this.onNoSpeech = null;

    this.onQueueChange = null;

    this.onError = null;

    this.onStream = null;
  }

  isMonitoring() {
    return this.active;
  }

  emitState(nextPhase, sid = this.sessionId) {
    this.phase = nextPhase;

    if (typeof this.onStateChange === "function") {
      this.onStateChange(nextPhase, sid);
    }
  }

  emitError(error, sid = this.sessionId) {
    if (typeof this.onError === "function") {
      this.onError(error, sid);
    }
  }

  emitQueueChange(sid = this.sessionId) {
    if (typeof this.onQueueChange === "function") {
      this.onQueueChange({
        sessionId: sid,
        queued: this.queue.length,
      });
    }
  }

  /**
   * Start microphone.
   */
  async start() {
    if (this.active) {
      return this.sessionId;
    }

    const sid = ++this.sessionId;

    this.active = true;

    this.stopping = false;

    this.chunkNumber = 0;

    this.sampleBuffer = [];

    this.sampleBufferLength = 0;

    this.queue = [];

    this.processing = false;

    this.lastAudioCallbackAt = Date.now();

    this.emitState("REQUESTING_PERMISSION", sid);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("This browser does not support microphone capture.");
      }

      /**
       * Use IDEAL constraints rather than
       * mandatory constraints.
       *
       * This prevents some laptop microphones
       * from rejecting the request.
       */
      const stream = await Promise.race([
        navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: {
              ideal: 1,
            },

            sampleRate: {
              ideal: TARGET_SAMPLE_RATE,
            },

            sampleSize: {
              ideal: 16,
            },

            echoCancellation: false,

            noiseSuppression: false,

            autoGainControl: false,
          },
        }),

        new Promise((_, reject) => {
          setTimeout(() => {
            reject(
              new Error("Microphone permission timed out. Please try again."),
            );
          }, MIC_PERMISSION_TIMEOUT_MS);
        }),
      ]);

      if (sid !== this.sessionId || !this.active || this.stopping) {
        stream.getTracks().forEach((track) => track.stop());

        return sid;
      }

      this.stream = stream;

      const tracks = stream.getAudioTracks();

      if (tracks.length === 0) {
        throw new Error("No microphone audio track was provided.");
      }

      /**
       * Read actual browser microphone settings.
       */
      const settings = tracks[0].getSettings ? tracks[0].getSettings() : {};

      /**
       * Build audio graph.
       */
      const AudioContextClass =
        window.AudioContext || window.webkitAudioContext;

      if (!AudioContextClass) {
        throw new Error("Web Audio API is not supported by this browser.");
      }

      const audioContext = new AudioContextClass();

      this.audioContext = audioContext;

      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      this.inputSampleRate =
        audioContext.sampleRate || settings.sampleRate || TARGET_SAMPLE_RATE;

      this.source = audioContext.createMediaStreamSource(stream);

      /**
       * ScriptProcessorNode is used here because
       * it works reliably in the browser without
       * requiring an AudioWorklet file.
       */
      this.processor = audioContext.createScriptProcessor(
        SCRIPT_PROCESSOR_BUFFER_SIZE,
        2,
        1,
      );

      /**
       * Silent output node keeps the processor
       * active without playing microphone audio.
       */
      this.silentGain = audioContext.createGain();

      this.silentGain.gain.value = 0;

      this.processor.onaudioprocess = (event) => {
        if (!this.active || this.stopping || sid !== this.sessionId) {
          return;
        }

        this.lastAudioCallbackAt = Date.now();

        const input = event.inputBuffer;

        if (!input) {
          return;
        }

        const channels = input.numberOfChannels || 1;

        const channelData = [];

        for (let channel = 0; channel < channels; channel++) {
          channelData.push(input.getChannelData(channel));
        }

        const mono = mixToMono(channelData);

        this.appendSamples(mono, sid);
      };

      this.source.connect(this.processor);

      this.processor.connect(this.silentGain);

      this.silentGain.connect(audioContext.destination);

      if (typeof this.onStream === "function") {
        this.onStream(stream, sid);
      }

      this.emitState("RECORDING", sid);

      /**
       * Watchdog.
       */
      this.watchdogTimer = window.setInterval(() => {
        if (!this.active || this.stopping || sid !== this.sessionId) {
          return;
        }

        const idleFor = Date.now() - this.lastAudioCallbackAt;

        if (idleFor > MAX_IDLE_MS) {
          this.emitError(
            new Error(
              "Microphone audio stopped arriving. Check the selected microphone and try again.",
            ),
            sid,
          );
        }
      }, WATCHDOG_INTERVAL_MS);

      return sid;
    } catch (error) {
      this.active = false;

      this.stopping = true;

      this.cleanup();

      this.emitError(error, sid);

      throw error;
    }
  }

  /**
   * Accumulate PCM until a complete
   * 3-second chunk is available.
   */
  appendSamples(samples, sid) {
    if (!samples || samples.length === 0) {
      return;
    }

    this.sampleBuffer.push(samples);

    this.sampleBufferLength += samples.length;

    const sourceRate = this.inputSampleRate || TARGET_SAMPLE_RATE;

    const targetSamples = Math.round(CHUNK_SECONDS * sourceRate);

    while (this.sampleBufferLength >= targetSamples) {
      const chunk = this.takeSamples(targetSamples);

      this.handleCapturedChunk(chunk, sourceRate, sid);
    }
  }

  /**
   * Remove exactly N samples from
   * the internal buffer.
   */
  takeSamples(count) {
    const output = new Float32Array(count);

    let outputOffset = 0;

    while (outputOffset < count && this.sampleBuffer.length > 0) {
      const first = this.sampleBuffer[0];

      const remaining = count - outputOffset;

      if (first.length <= remaining) {
        output.set(first, outputOffset);

        outputOffset += first.length;

        this.sampleBuffer.shift();
      } else {
        output.set(first.subarray(0, remaining), outputOffset);

        this.sampleBuffer[0] = first.subarray(remaining);

        outputOffset = count;
      }
    }

    this.sampleBufferLength -= count;

    return output;
  }

  /**
   * Process one complete 3-second chunk.
   *
   * IMPORTANT:
   *
   * Speech detection is informational only.
   * Every chunk continues to the AI pipeline.
   */
  handleCapturedChunk(rawSamples, sourceSampleRate, sid) {
    if (sid !== this.sessionId || !this.active || this.stopping) {
      return;
    }

    const chunkNumber = ++this.chunkNumber;

    /**
     * ----------------------------------------
     * SPEECH ACTIVITY MEASUREMENT
     * ----------------------------------------
     *
     * We calculate it for diagnostics,
     * but DO NOT reject the chunk.
     */
    let speechMetrics = null;

    try {
      speechMetrics = isSpeech(rawSamples, sourceSampleRate);
    } catch {
      speechMetrics = null;
    }

    /**
     * IMPORTANT:
     *
     * We deliberately do NOT do:
     *
     * if (!speechMetrics.isSpeech) return;
     *
     * because that was causing your
     * "No speech detected" problem.
     *
     * The AI receives the chunk regardless.
     */

    if (
      speechMetrics &&
      typeof this.onNoSpeech === "function" &&
      !speechMetrics.isSpeech
    ) {
      /**
       * Keep this callback disabled from
       * controlling the pipeline.
       *
       * We intentionally don't call onNoSpeech.
       */
    }

    /**
     * ----------------------------------------
     * DC OFFSET REMOVAL
     * ----------------------------------------
     */
    const dcCorrected = removeDcOffset(rawSamples);

    /**
     * ----------------------------------------
     * LEVEL NORMALIZATION
     * ----------------------------------------
     */
    const normalized = normalizeSpeechLevel(dcCorrected);

    /**
     * ----------------------------------------
     * RESAMPLE TO 16 KHZ
     * ----------------------------------------
     */
    let resampled;

    try {
      resampled = resampleLinear(
        normalized,
        sourceSampleRate,
        TARGET_SAMPLE_RATE,
      );
    } catch (error) {
      this.emitError(error, sid);

      return;
    }

    /**
     * ----------------------------------------
     * WAV
     * ----------------------------------------
     */
    let blob;

    try {
      blob = encodeWav(resampled, TARGET_SAMPLE_RATE);
    } catch (error) {
      this.emitError(error, sid);

      return;
    }

    /**
     * ----------------------------------------
     * FIFO QUEUE
     * ----------------------------------------
     */
    this.enqueueChunk({
      blob,
      chunkNumber,
      sessionId: sid,
    });
  }

  /**
   * Add a chunk to the bounded FIFO.
   */
  enqueueChunk(item) {
    if (item.sessionId !== this.sessionId || !this.active || this.stopping) {
      return;
    }

    /**
     * If inference is busy,
     * retain up to 3 chunks.
     */
    if (this.queue.length >= MAX_ANALYSIS_QUEUE) {
      if (typeof this.onChunkDropped === "function") {
        this.onChunkDropped({
          sessionId: item.sessionId,

          chunkNumber: item.chunkNumber,

          reason: "Analysis queue is full.",
        });
      }

      return;
    }

    this.queue.push(item);

    this.emitQueueChange(item.sessionId);

    this.processQueue();
  }

  /**
   * Process queued chunks sequentially.
   */
  async processQueue() {
    if (this.processing || !this.active || this.stopping) {
      return;
    }

    const item = this.queue.shift();

    if (!item) {
      return;
    }

    this.processing = true;

    this.emitQueueChange(item.sessionId);

    this.emitState("ANALYZING", item.sessionId);

    try {
      if (typeof this.onChunk === "function") {
        await this.onChunk(item);
      }
    } catch (error) {
      this.emitError(error, item.sessionId);
    } finally {
      this.processing = false;

      if (item.sessionId === this.sessionId && this.active && !this.stopping) {
        this.emitQueueChange(item.sessionId);

        if (this.queue.length === 0) {
          this.emitState("RECORDING", item.sessionId);
        }

        this.processQueue();
      }
    }
  }

  /**
   * Stop microphone monitoring.
   */
  stop() {
    if (!this.active && !this.stream && !this.audioContext) {
      return;
    }

    const sid = this.sessionId;

    this.stopping = true;

    this.emitState("STOPPING", sid);

    this.active = false;

    this.cleanup();

    this.emitQueueChange(sid);

    this.emitState("IDLE", sid);
  }

  /**
   * Release all browser audio resources.
   */
  cleanup() {
    if (this.watchdogTimer !== null) {
      clearInterval(this.watchdogTimer);

      this.watchdogTimer = null;
    }

    if (this.processor) {
      try {
        this.processor.onaudioprocess = null;

        this.processor.disconnect();
      } catch {
        // Ignore cleanup errors.
      }

      this.processor = null;
    }

    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        // Ignore cleanup errors.
      }

      this.source = null;
    }

    if (this.silentGain) {
      try {
        this.silentGain.disconnect();
      } catch {
        // Ignore cleanup errors.
      }

      this.silentGain = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore.
        }
      });

      this.stream = null;
    }

    if (this.audioContext) {
      const context = this.audioContext;

      this.audioContext = null;

      if (context.state !== "closed") {
        context.close().catch(() => {});
      }
    }

    this.sampleBuffer = [];

    this.sampleBufferLength = 0;

    this.queue = [];

    this.processing = false;
  }
}
