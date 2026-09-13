/**
 * Deterministic, dependency-free tests for the browser mic pipeline.
 * Run: npm test   (node scripts/monitor-smoke-test.mjs)
 *
 * The browser media stack is stubbed (getUserMedia, AudioContext,
 * ScriptProcessorNode) so the recorder's real logic runs under Node.
 *
 * What is covered:
 *   1. WAV encoder produces a valid, standalone mono 16 kHz 16-bit PCM WAV.
 *   2. Speech gate: silence/quiet noise → no speech; speech-like audio → speech.
 *   3. SEQUENTIAL CHUNKS (#1 and #2) both finalize as independently valid
 *      WAVs — regression test for the old "chunk #2: Unable to decode audio
 *      data" MediaRecorder failure (headerless container fragments).
 *   4. No-speech chunks never reach onChunk (i.e. never POST /api/analyze),
 *      are reported as NO_SPEECH, and monitoring continues afterwards.
 *   5. stop() invalidates stale callbacks and cleans up (track stopped,
 *      context closed, handlers nulled); restart works on the same instance.
 *   6. FIFO analysis queue: a slow analysis does NOT skip the next chunk —
 *      chunks #2/#3 wait in a bounded FIFO and are analyzed in capture order;
 *      when the queue is full the newest chunk is released WITH a report.
 *
 * What CANNOT be tested here (needs a real browser, mic, and backend):
 *   - getUserMedia permission flow and real device sample rates, including
 *     the AudioContext({sampleRate:16000}) fallback path on Safari.
 *   - Real HTTP POST /api/analyze through Spring Boot → FastAPI.
 *   - React rendering of NO_SPEECH/COMPLETE states (no DOM test runner in
 *     this project; the hook's state mapping is exercised indirectly).
 */

import {
  MicMonitor,
  encodeWavPcm16,
  resampleTo16k,
  TARGET_SAMPLE_RATE
} from '../src/utils/recorder.js';
import { analyzeSpeechActivity, SPEECH_GATE_DEFAULTS } from '../src/utils/speechActivity.js';

// ---------------------------------------------------------------- stubs

const scriptProcessors = [];
const createdContexts = [];
let micTrackStopped = false;
let micTrackEndedListeners = [];

class FakeAudioContext {
  constructor(opts) {
    this.sampleRate = (opts && opts.sampleRate) || 48000;
    this.state = 'running';
    this.destination = { _: 'destination' };
    createdContexts.push(this);
  }
  async resume() {
    this.state = 'running';
  }
  async close() {
    this.state = 'closed';
  }
  createMediaStreamSource() {
    return { connect() {}, disconnect() {} };
  }
  createGain() {
    return { gain: { value: 1 }, connect() {}, disconnect() {} };
  }
  createScriptProcessor(bufferSize) {
    const node = { bufferSize, onaudioprocess: null, connect() {}, disconnect() {} };
    scriptProcessors.push(node);
    return node;
  }
}

/** The processor node of the most recently started session. */
function currentProcessor() {
  return scriptProcessors[scriptProcessors.length - 1];
}

globalThis.window = globalThis;
globalThis.AudioContext = FakeAudioContext;

const micTrack = {
  addEventListener(_t, fn) {
    micTrackEndedListeners.push(fn);
  },
  removeEventListener(_t, fn) {
    micTrackEndedListeners = micTrackEndedListeners.filter((l) => l !== fn);
  },
  stop() {
    micTrackStopped = true;
  }
};
const fakeStream = { getAudioTracks: () => [micTrack], getTracks: () => [micTrack] };

Object.defineProperty(globalThis, 'navigator', {
  value: { mediaDevices: { getUserMedia: async () => fakeStream } },
  configurable: true
});

function resetStubs() {
  scriptProcessors.length = 0;
  createdContexts.length = 0;
  micTrackStopped = false;
  micTrackEndedListeners = [];
}

// ---------------------------------------------------------------- helpers

const failureMessages = [];
const check = (cond, msg) => {
  if (!cond) failureMessages.push(msg);
};

/** Feed samples into the recorder through the stubbed ScriptProcessorNode. */
function feed(node, samples) {
  for (let i = 0; i < samples.length; i += node.bufferSize) {
    if (!node.onaudioprocess) return; // handler nulled by cleanup: session dead
    const buf = new Float32Array(node.bufferSize);
    buf.set(samples.subarray(i, Math.min(samples.length, i + node.bufferSize)));
    node.onaudioprocess({
      inputBuffer: {
        length: buf.length,
        numberOfChannels: 1,
        sampleRate: 16000,
        getChannelData: (c) => (c === 0 ? buf : new Float32Array(buf.length))
      }
    });
  }
}

/** Speech-like signal: 220 Hz tone with a 3 Hz syllable envelope. */
function speechLike(rate, seconds) {
  const n = Math.round(rate * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const t = i / rate;
    const env = 0.6 * (0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t));
    out[i] = 0.35 * env * Math.sin(2 * Math.PI * 220 * t);
  }
  return out;
}

const silence = (rate, seconds) => new Float32Array(Math.round(rate * seconds));

/** Parse a WAV blob's header fields with strict offset checks. */
async function parseWav(blob) {
  const ab = await blob.arrayBuffer();
  const v = new DataView(ab);
  const str = (o, n) => {
    let s = '';
    for (let i = 0; i < n; i += 1) s += String.fromCharCode(v.getUint8(o + i));
    return s;
  };
  return {
    byteLength: ab.byteLength,
    riff: str(0, 4),
    wave: str(8, 4),
    fmt: str(12, 4),
    dataTag: str(36, 4),
    fmtSize: v.getUint32(16, true),
    audioFormat: v.getUint16(20, true),
    channels: v.getUint16(22, true),
    sampleRate: v.getUint32(24, true),
    byteRate: v.getUint32(28, true),
    blockAlign: v.getUint16(32, true),
    bitsPerSample: v.getUint16(34, true),
    dataSize: v.getUint32(40, true)
  };
}

function makeMonitor(handlers = {}) {
  const m = new MicMonitor(handlers.options || {});
  m.onChunk = handlers.onChunk || null;
  m.onChunkDropped = handlers.onChunkDropped || null;
  m.onNoSpeech = handlers.onNoSpeech || null;
  m.onError = handlers.onError || null;
  m.onStateChange = handlers.onStateChange || null;
  return m;
}

// ---------------------------------------------------------------- 1. WAV encoder

{
  const blob = encodeWavPcm16(new Float32Array([0, 0.5, -0.5, 1]), 16000);
  check(blob.type === 'audio/wav', `WAV blob type is ${blob.type}`);
  const w = await parseWav(blob);
  check(w.riff === 'RIFF' && w.wave === 'WAVE' && w.fmt === 'fmt ' && w.dataTag === 'data', 'WAV tags missing');
  check(w.fmtSize === 16, 'fmt chunk size must be 16');
  check(w.audioFormat === 1, 'audioFormat must be PCM (1)');
  check(w.channels === 1, 'WAV must be mono');
  check(w.sampleRate === 16000, `WAV sample rate must be 16000, got ${w.sampleRate}`);
  check(w.bitsPerSample === 16, 'WAV must be 16-bit');
  check(w.byteRate === 32000 && w.blockAlign === 2, 'byteRate/blockAlign wrong');
  check(w.dataSize === 8 && w.byteLength === 44 + 8, 'RIFF/data sizes inconsistent');
  const ab = await blob.arrayBuffer();
  const s = new DataView(ab).getInt16(44 + 2, true); // 0.5 → ±16383.5
  check(s === 16383 || s === 16384, `0.5 sample encoded as ${s}`);

  let threw = false;
  try {
    encodeWavPcm16(new Float32Array(0), 16000);
  } catch {
    threw = true;
  }
  check(threw, 'encoding empty samples must throw (never send empty chunks)');
  console.log('1. WAV encoder (mono, 16 kHz, 16-bit, standalone): PASS');
}

// ---------------------------------------------------------------- 2. speech gate

{
  check(analyzeSpeechActivity(new Float32Array(48000), 16000).isSpeech === false, 'silence classified as speech');
  check(analyzeSpeechActivity(new Float32Array(0), 16000).isSpeech === false, 'empty input classified as speech');
  const sine = speechLike(16000, 3);
  const speech = analyzeSpeechActivity(sine, 16000);
  check(speech.isSpeech === true, 'speech-like audio not detected as speech');
  check(speech.rms > 0 && speech.frames === 30, `unexpected metrics: ${JSON.stringify(speech)}`);
  const quiet = sine.map((v) => v * (0.02 / 0.35));
  check(analyzeSpeechActivity(quiet, 16000).isSpeech === false, 'room-tone-level audio classified as speech');
  check(
    analyzeSpeechActivity(sine, 16000, { rmsThreshold: 0.9 }).isSpeech === false,
    'config override (rmsThreshold) has no effect'
  );
  check(typeof SPEECH_GATE_DEFAULTS.rmsThreshold === 'number', 'SPEECH_GATE_DEFAULTS missing');
  console.log('2. Speech gate (silence → no speech, speech → speech, configurable): PASS');
}

// ---------------------------------------------------------------- 3. sequential chunks

{
  resetStubs();
  const chunks = [];
  const phases = [];
  const m = makeMonitor({
    onChunk: async ({ blob }) => {
      chunks.push(blob);
    },
    onStateChange: (p) => phases.push(p)
  });
  await m.start();
  check(scriptProcessors.length === 1, 'expected exactly one capture node');
  // 2 × 3.2 s of speech at 16 kHz → chunks #1 and #2, with partial remainders.
  // A macrotask flush between the two mirrors real browser timing: real
  // onaudioprocess events are separate macrotasks 3 s apart, so in-flight
  // analysis always settles between chunk boundaries (unless it is genuinely
  // slower than 3 s, which is the tested-and-reported drop path in scenario 6).
  feed(currentProcessor(), speechLike(16000, 3.2));
  await new Promise((r) => setTimeout(r, 0));
  feed(currentProcessor(), speechLike(16000, 3.2));
  await new Promise((r) => setTimeout(r, 0));
  check(chunks.length === 2, `expected chunks #1 and #2, got ${chunks.length}`);
  for (let i = 0; i < chunks.length; i += 1) {
    const w = await parseWav(chunks[i]);
    check(w.riff === 'RIFF' && w.wave === 'WAVE', `chunk #${i + 1} not a standalone WAV`);
    check(w.audioFormat === 1 && w.channels === 1, `chunk #${i + 1} not mono PCM`);
    check(w.sampleRate === 16000, `chunk #${i + 1} sampleRate ${w.sampleRate}`);
    check(w.bitsPerSample === 16, `chunk #${i + 1} not 16-bit`);
    check(w.dataSize === 48000 * 2, `chunk #${i + 1} data size ${w.dataSize}, expected 96000`);
    check(w.dataSize === w.byteLength - 44, `chunk #${i + 1} header/data size mismatch`);
  }
  check(phases.includes('ANALYZING') && phases.includes('RECORDING'), 'missing ANALYZING/RECORDING states');
  m.stop();
  console.log('3. Sequential chunks #1 AND #2 are independently valid 16 kHz mono WAVs: PASS');
}

// ---------------------------------------------------------------- 4. NO_SPEECH gating

{
  resetStubs();
  const chunks = [];
  const noSpeech = [];
  const phases = [];
  const m = makeMonitor({
    onChunk: async ({ blob }) => {
      chunks.push(blob);
    },
    onNoSpeech: (e) => noSpeech.push(e),
    onStateChange: (p) => phases.push(p)
  });
  await m.start();
  feed(currentProcessor(), silence(16000, 3.2));
  check(noSpeech.length === 1, 'silence chunk not reported as NO_SPEECH');
  check(chunks.length === 0, 'silence chunk reached onChunk (would POST /api/analyze)');
  check(phases.includes('NO_SPEECH'), 'NO_SPEECH state not emitted');
  check(typeof noSpeech[0].metrics.rms === 'number', 'no-speech report missing RMS metric');
  // Monitoring must continue normally afterwards.
  feed(currentProcessor(), speechLike(16000, 3.2));
  check(chunks.length === 1, 'monitoring did not continue after a NO_SPEECH chunk');
  m.stop();
  console.log('4. NO_SPEECH: silence never analyzed, monitoring continues: PASS');
}

// ---------------------------------------------------------------- 5. stop/restart lifecycle

{
  resetStubs();
  const chunks = [];
  let errored = null;
  const m = makeMonitor({
    onChunk: async ({ blob }) => {
      chunks.push(blob);
    },
    onError: (e) => {
      errored = e;
    }
  });
  await m.start();
  const sid = m.sessionId;
  feed(currentProcessor(), speechLike(16000, 1.0)); // partial chunk only
  m.stop();
  m.stop(); // idempotent
  check(micTrackStopped, 'mic track not stopped after stop()');
  check(createdContexts.length === 1 && createdContexts[0].state === 'closed', 'capture AudioContext not closed');
  check(scriptProcessors[0].onaudioprocess === null, 'audio handler not nulled after stop()');
  check(!m.isMonitoring(), 'isMonitoring() true after stop()');
  check(!errored, `unexpected error on clean stop: ${errored && errored.message}`);
  const before = chunks.length;
  feed(scriptProcessors[0], speechLike(16000, 3.2)); // stale audio for a dead session
  check(chunks.length === before, 'stale audio produced a chunk after stop()');

  // Restart on the same instance: new generation, fresh capture, chunks flow.
  await m.start();
  check(m.sessionId > sid, 'sessionId not incremented on restart');
  feed(currentProcessor(), speechLike(16000, 3.2));
  check(chunks.length === before + 1, 'restart did not produce a new chunk');
  m.stop();
  console.log('5. stop() cleanup + stale-callback invalidation + restart: PASS');
}

// ---------------------------------------------------------------- 6. FIFO analysis queue

{
  resetStubs();
  const order = [];
  const dropped = [];
  let release;
  let heldOnce = false;
  const m = makeMonitor({
    onChunk: ({ chunkNumber }) =>
      new Promise((resolve) => {
        order.push(chunkNumber);
        if (!heldOnce) {
          heldOnce = true; // hold ONLY chunk #1 to simulate slow CPU inference
          release = resolve;
        } else {
          resolve();
        }
      }),
    onChunkDropped: (e) => dropped.push(e)
  });
  await m.start();
  // Chunk #1 starts analyzing and is held in flight.
  feed(currentProcessor(), speechLike(16000, 3.2));
  check(order.length === 1 && order[0] === 1, `chunk #1 not analyzed first: ${JSON.stringify(order)}`);
  // Chunk #2 arrives while #1 is in flight: it MUST be queued, never dropped.
  feed(currentProcessor(), speechLike(16000, 3.2));
  check(dropped.length === 0, 'chunk #2 was dropped while chunk #1 was still being analyzed');
  check(m.analysisQueue.length === 1, `chunk #2 was not queued (queue=${m.analysisQueue.length})`);
  // Chunk #3 also queues behind #2.
  feed(currentProcessor(), speechLike(16000, 3.2));
  check(m.analysisQueue.length === 2, `chunk #3 was not queued (queue=${m.analysisQueue.length})`);
  // Finish chunk #1: the drain loop must continue with #2 then #3, in order.
  release();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  check(
    order.length === 3 && order.join(',') === '1,2,3',
    `FIFO order violated: [${order.join(', ')}]`
  );
  check(dropped.length === 0, `unexpected drops during FIFO flow: ${JSON.stringify(dropped)}`);
  check(m.analysisQueue.length === 0, 'queue not empty after drain');
  m.stop();
  console.log('6. FIFO queue: slow analysis NEVER skips the next chunk (1→2→3 in order): PASS');
}

// ---------------------------------------------------------------- 6b. bounded backpressure

{
  resetStubs();
  const order = [];
  const dropped = [];
  let release;
  const m = makeMonitor({
    onChunk: ({ chunkNumber }) =>
      new Promise((resolve) => {
        order.push(chunkNumber);
        if (!release) release = resolve; // hold chunk #1 forever until released below
        else resolve();
      }),
    onChunkDropped: (e) => dropped.push(e)
  });
  await m.start();
  feed(currentProcessor(), speechLike(16000, 3.2)); // #1 analyzing (held)
  feed(currentProcessor(), speechLike(16000, 3.2)); // #2 queued (1/3)
  feed(currentProcessor(), speechLike(16000, 3.2)); // #3 queued (2/3)
  feed(currentProcessor(), speechLike(16000, 3.2)); // #4 queued (3/3 = full)
  check(m.analysisQueue.length === 3, `expected full queue of 3, got ${m.analysisQueue.length}`);
  feed(currentProcessor(), speechLike(16000, 3.2)); // #5 → released WITH a report
  check(dropped.length === 1, 'queue-full chunk was not reported (silent drop)');
  check(
    dropped[0] && dropped[0].chunkNumber === 5,
    `wrong chunk reported as released: ${JSON.stringify(dropped)}`
  );
  check(typeof dropped[0].reason === 'string' && dropped[0].reason.length > 0, 'drop report missing reason');
  // Release #1: remaining queued chunks #2–#4 drain strictly in order.
  release();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  check(order.join(',') === '1,2,3,4', `FIFO order violated under backpressure: [${order.join(', ')}]`);
  check(dropped.length === 1, 'extra unexpected drops after drain');
  m.stop();
  console.log('6b. Bounded queue (max 3): full queue releases newest chunk WITH report, order kept: PASS');
}

// ---------------------------------------------------------------- 7. fallback resampler

{
  const src = new Float32Array(48000).fill(0.25);
  const out = resampleTo16k(src, 48000);
  check(out.length === 16000, `resample length ${out.length}, expected 16000`);
  check(Math.abs(out[100] - 0.25) < 1e-6, 'resample changed constant signal level');
  check(resampleTo16k(src, TARGET_SAMPLE_RATE) === src, 'same-rate passthrough must return input');
  console.log('7. Fallback resampler (48 kHz → 16 kHz): PASS');
}

// ---------------------------------------------------------------- summary

if (failureMessages.length === 0) {
  console.log('\nALL TESTS PASSED');
  process.exit(0);
} else {
  console.error(`\n${failureMessages.length} CHECK(S) FAILED:`);
  for (const msg of failureMessages) console.error(` - ${msg}`);
  process.exit(1);
}
