// frontend/src/hooks/useMonitor.js

import { useState, useEffect, useRef, useCallback } from "react";

import { MicMonitor, describeMicError } from "../utils/recorder";

import { analyzeAudio } from "../utils/api";

import { mapApiResponse } from "../utils/resultMapper";

import { speakHighRiskWarning } from "../utils/warning";

/**
 * Monitoring states.
 */
const MONITOR_STATES = [
  "IDLE",
  "REQUESTING_PERMISSION",
  "RECORDING",
  "ANALYZING",
  "NO_SPEECH",
  "COMPLETE",
  "STOPPING",
  "ERROR",
];

/**
 * Keep only the latest 50 analyzed chunks.
 */
const MAX_HISTORY = 50;

/**
 * EMA smoothing.
 *
 * Higher = more responsive.
 * Lower = smoother.
 */
const EMA_ALPHA = 0.35;

/**
 * Require two consecutive high-risk
 * results before triggering an alert.
 */
const PERSISTENT_HIGH_RISK_COUNT = 2;

/**
 * Risk levels:
 *
 * 0–40   LOW
 * 41–60  MEDIUM
 * 61–80  HIGH
 * 81–100 CRITICAL
 */
const HIGH_RISK_SCORE = 61;

/**
 * Main monitoring hook.
 */
export function useMonitor() {
  // ------------------------------------------------------------
  // React state
  // ------------------------------------------------------------

  const [phase, setPhase] = useState("IDLE");

  const [error, setError] = useState(null);

  const [currentResult, setCurrentResult] = useState(null);

  const [history, setHistory] = useState([]);

  const [droppedInfo, setDroppedInfo] = useState(null);

  /**
   * Kept for compatibility with DashboardPage.
   *
   * The recorder no longer blocks AI analysis
   * because of this value.
   */
  const [noSpeechInfo, setNoSpeechInfo] = useState(null);

  const [queueDepth, setQueueDepth] = useState(0);

  const [session, setSession] = useState(0);

  const [liveAnalyser, setLiveAnalyser] = useState(null);

  // ------------------------------------------------------------
  // Refs
  // ------------------------------------------------------------

  const monitorRef = useRef(null);

  const sessionRef = useRef(0);

  const unmountedRef = useRef(false);

  const phaseRef = useRef("IDLE");

  const audioContextRef = useRef(null);

  /**
   * Current EMA value.
   */
  const smoothedRiskRef = useRef(null);

  /**
   * Number of consecutive
   * high-risk results.
   */
  const persistentHighRiskRef = useRef(0);

  /**
   * Prevent repeated alerts during
   * the same high-risk period.
   */
  const warningTriggeredRef = useRef(false);

  // ------------------------------------------------------------
  // Create microphone monitor
  // ------------------------------------------------------------

  if (!monitorRef.current) {
    monitorRef.current = new MicMonitor();
  }

  const monitor = monitorRef.current;

  // ------------------------------------------------------------
  // Phase synchronization
  // ------------------------------------------------------------

  const setPhaseSynced = useCallback((nextPhase) => {
    const resolved =
      typeof nextPhase === "function" ? nextPhase(phaseRef.current) : nextPhase;

    phaseRef.current = resolved;

    setPhase(resolved);
  }, []);

  // ------------------------------------------------------------
  // Session check
  // ------------------------------------------------------------

  const isCurrent = useCallback((sid) => {
    return sid === sessionRef.current && !unmountedRef.current;
  }, []);

  // ------------------------------------------------------------
  // Analyzer cleanup
  // ------------------------------------------------------------

  const teardownAnalyser = useCallback(() => {
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
    }

    audioContextRef.current = null;

    setLiveAnalyser(null);
  }, []);

  // ------------------------------------------------------------
  // EMA smoothing
  // ------------------------------------------------------------

  const calculateSmoothedRisk = useCallback((rawRisk) => {
    if (typeof rawRisk !== "number" || !Number.isFinite(rawRisk)) {
      return null;
    }

    /**
     * First result initializes
     * the EMA.
     */
    if (smoothedRiskRef.current === null) {
      smoothedRiskRef.current = rawRisk;

      return rawRisk;
    }

    const smoothed =
      EMA_ALPHA * rawRisk + (1 - EMA_ALPHA) * smoothedRiskRef.current;

    const clamped = Math.max(0, Math.min(100, smoothed));

    smoothedRiskRef.current = clamped;

    return clamped;
  }, []);

  // ------------------------------------------------------------
  // Persistent high-risk detection
  // ------------------------------------------------------------

  const updatePersistentRisk = useCallback((smoothedRisk) => {
    if (typeof smoothedRisk !== "number" || !Number.isFinite(smoothedRisk)) {
      return false;
    }

    if (smoothedRisk >= HIGH_RISK_SCORE) {
      persistentHighRiskRef.current += 1;
    } else {
      persistentHighRiskRef.current = 0;

      /**
       * Allow a future alert
       * if risk becomes high again.
       */
      warningTriggeredRef.current = false;
    }

    if (
      persistentHighRiskRef.current >= PERSISTENT_HIGH_RISK_COUNT &&
      !warningTriggeredRef.current
    ) {
      warningTriggeredRef.current = true;

      return true;
    }

    return false;
  }, []);

  // ------------------------------------------------------------
  // AI chunk analysis
  // ------------------------------------------------------------

  const handleChunk = useCallback(
    async ({ blob, chunkNumber, sessionId }) => {
      if (!isCurrent(sessionId)) {
        return;
      }

      try {
        /**
         * Send actual microphone WAV
         * to the existing backend.
         */
        const api = await analyzeAudio(blob, "microphone-chunk.wav");

        /**
         * Ignore results from
         * old sessions.
         */
        if (
          !isCurrent(sessionId) ||
          phaseRef.current === "IDLE" ||
          phaseRef.current === "STOPPING" ||
          phaseRef.current === "ERROR"
        ) {
          return;
        }

        /**
         * Map actual backend result.
         */
        const mapped = mapApiResponse(api, {
          source: "mic",
          chunkNumber,
        });

        /**
         * IMPORTANT:
         *
         * This is the actual backend
         * spoof probability.
         *
         * We do NOT fabricate it.
         */
        const rawSpoofPercent = mapped.aiScore;

        /**
         * EMA smoothing.
         */
        const smoothedSpoofPercent = calculateSmoothedRisk(rawSpoofPercent);

        if (smoothedSpoofPercent === null) {
          return;
        }

        /**
         * Risk score.
         */
        const smoothedRiskScore = Math.round(smoothedSpoofPercent);

        /**
         * Risk level.
         */
        let smoothedRiskLevel = "LOW";

        if (smoothedRiskScore >= 81) {
          smoothedRiskLevel = "CRITICAL";
        } else if (smoothedRiskScore >= 61) {
          smoothedRiskLevel = "HIGH";
        } else if (smoothedRiskScore >= 41) {
          smoothedRiskLevel = "MEDIUM";
        }

        // ------------------------------------------------------
        // Human / synthetic classification
        // ------------------------------------------------------

        let liveClassification;

        let liveDirective;

        /**
         * 81–100
         */
        if (smoothedSpoofPercent >= 81) {
          liveClassification = "SYNTHETIC VOICE DETECTED";

          liveDirective =
            "High probability of synthetic speech. Do not act on this audio; verify the speaker through a separate, trusted channel.";
        } else if (smoothedSpoofPercent >= 61) {
          /**
           * 61–80
           */
          liveClassification = "SYNTHETIC VOICE LIKELY";

          liveDirective =
            "Elevated probability of synthetic speech. Continue with additional verification before trusting this audio.";
        } else if (smoothedSpoofPercent >= 50) {
          /**
           * 50–60
           */
          liveClassification = "INCONCLUSIVE VOICE SIGNAL";

          liveDirective =
            "The analysis is inconclusive. Treat this result cautiously and verify through a second channel if the decision is high-stakes.";
        } else if (smoothedSpoofPercent > 40) {
          /**
           * 40.1–49.9
           */
          liveClassification = "LIKELY HUMAN VOICE";

          liveDirective =
            "The audio is more likely to be human, but the medium risk score indicates some uncertainty. Continue with standard verification practices.";
        } else {
          /**
           * 0–40
           */
          liveClassification = "AUTHENTIC HUMAN VOICE";

          liveDirective =
            "No strong synthetic-speech signals detected. Standard verification practices still apply.";
        }

        // ------------------------------------------------------
        // Persistent alert
        // ------------------------------------------------------

        const persistentHighRisk = updatePersistentRisk(smoothedRiskScore);

        // ------------------------------------------------------
        // Main result
        // ------------------------------------------------------

        const smoothedResult = {
          ...mapped,

          /**
           * Smoothed display values.
           */
          aiScore: smoothedSpoofPercent,

          spoofProbability: smoothedSpoofPercent / 100,

          riskScore: smoothedRiskScore,

          riskLevel: smoothedRiskLevel,

          classification: liveClassification,

          directive: liveDirective,

          /**
           * Only true after
           * persistent high risk.
           */
          alert: persistentHighRisk,
        };

        setCurrentResult(smoothedResult);

        // ------------------------------------------------------
        // RAW HISTORY
        // ------------------------------------------------------

        setHistory((previous) => {
          const next = [
            ...previous,
            {
              chunkNumber: mapped.chunkNumber,

              /**
               * REAL raw backend score.
               */
              spoofPercent: rawSpoofPercent,

              riskScore: mapped.riskScore,

              riskLevel: mapped.riskLevel,

              alert: mapped.alert,

              label: mapped.label,

              message: mapped.message,

              durationSeconds: mapped.durationSeconds,

              sampleRate: mapped.sampleRate,

              inferenceTimeMs: mapped.inferenceTimeMs,
            },
          ];

          return next.slice(-MAX_HISTORY);
        });

        /**
         * Voice warning only after
         * two consecutive high-risk
         * results.
         */
        if (persistentHighRisk) {
          try {
            speakHighRiskWarning();
          } catch {
            /**
             * Browser speech synthesis
             * is optional.
             */
          }
        }
      } catch (err) {
        if (!isCurrent(sessionId)) {
          return;
        }

        const message =
          err && err.message
            ? err.message
            : "Chunk analysis failed. Check that the AI and backend services are running.";

        monitor.stop();

        setError(message);

        phaseRef.current = "ERROR";

        setPhase("ERROR");
      }
    },
    [monitor, isCurrent, calculateSmoothedRisk, updatePersistentRisk],
  );

  // ------------------------------------------------------------
  // Dropped chunk
  // ------------------------------------------------------------

  const handleChunkDropped = useCallback(
    ({ sessionId, chunkNumber, reason }) => {
      if (!isCurrent(sessionId)) {
        return;
      }

      setDroppedInfo({
        chunkNumber,
        reason,
        at: Date.now(),
      });
    },
    [isCurrent],
  );

  // ------------------------------------------------------------
  // Queue depth
  // ------------------------------------------------------------

  const handleQueueChange = useCallback(
    ({ sessionId, queued }) => {
      if (!isCurrent(sessionId)) {
        return;
      }

      setQueueDepth(queued);
    },
    [isCurrent],
  );

  // ------------------------------------------------------------
  // No-speech callback
  // ------------------------------------------------------------

  const handleNoSpeech = useCallback(
    ({ sessionId, chunkNumber, metrics }) => {
      if (!isCurrent(sessionId)) {
        return;
      }

      /**
       * We keep this only for
       * compatibility.
       *
       * The recorder does not call it
       * anymore because no-speech chunks
       * are now analyzed.
       */
      setNoSpeechInfo({
        chunkNumber,

        rms: metrics && typeof metrics.rms === "number" ? metrics.rms : null,

        at: Date.now(),
      });
    },
    [isCurrent],
  );

  // ------------------------------------------------------------
  // Microphone errors
  // ------------------------------------------------------------

  const handleError = useCallback(
    (err, sid) => {
      if (!isCurrent(sid)) {
        return;
      }

      const message =
        err instanceof Error
          ? err.message
          : describeMicError(err) || "Microphone error.";

      setError(message);

      phaseRef.current = "ERROR";

      setPhaseSynced("ERROR");
    },
    [isCurrent, setPhaseSynced],
  );

  // ------------------------------------------------------------
  // Live waveform
  // ------------------------------------------------------------

  const handleStream = useCallback(
    (stream, sid) => {
      if (!isCurrent(sid)) {
        return;
      }

      try {
        const AudioContextClass =
          window.AudioContext || window.webkitAudioContext;

        if (!AudioContextClass) {
          return;
        }

        const audioContext = new AudioContextClass();

        const source = audioContext.createMediaStreamSource(stream);

        const analyser = audioContext.createAnalyser();

        analyser.fftSize = 256;

        analyser.smoothingTimeConstant = 0.8;

        source.connect(analyser);

        audioContextRef.current = audioContext;

        setLiveAnalyser(analyser);
      } catch {
        setLiveAnalyser(null);
      }
    },
    [isCurrent],
  );

  // ------------------------------------------------------------
  // Wire recorder callbacks
  // ------------------------------------------------------------

  useEffect(() => {
    unmountedRef.current = false;

    monitor.onChunk = handleChunk;

    monitor.onChunkDropped = handleChunkDropped;

    monitor.onNoSpeech = handleNoSpeech;

    monitor.onQueueChange = handleQueueChange;

    monitor.onError = handleError;

    monitor.onStateChange = (nextPhase, sid) => {
      if (!isCurrent(sid)) {
        return;
      }

      if (MONITOR_STATES.includes(nextPhase)) {
        phaseRef.current = nextPhase;

        setPhase(nextPhase);
      }
    };

    monitor.onStream = handleStream;

    return () => {
      unmountedRef.current = true;

      monitor.onChunk = null;

      monitor.onChunkDropped = null;

      monitor.onNoSpeech = null;

      monitor.onQueueChange = null;

      monitor.onError = null;

      monitor.onStateChange = null;

      monitor.onStream = null;

      monitor.stop();

      teardownAnalyser();
    };
  }, [
    monitor,
    handleChunk,
    handleChunkDropped,
    handleNoSpeech,
    handleQueueChange,
    handleError,
    handleStream,
    isCurrent,
    teardownAnalyser,
  ]);

  // ------------------------------------------------------------
  // Analyzer cleanup when capture stops
  // ------------------------------------------------------------

  useEffect(() => {
    if (phase === "IDLE" || phase === "ERROR" || phase === "STOPPING") {
      teardownAnalyser();
    }
  }, [phase, teardownAnalyser]);

  // ------------------------------------------------------------
  // Start monitoring
  // ------------------------------------------------------------

  const startMonitoring = useCallback(async () => {
    if (monitor.isMonitoring()) {
      return;
    }

    /**
     * Clear previous session.
     */
    setError(null);

    setCurrentResult(null);

    setHistory([]);

    setDroppedInfo(null);

    setNoSpeechInfo(null);

    setQueueDepth(0);

    /**
     * Reset smoothing.
     */
    smoothedRiskRef.current = null;

    persistentHighRiskRef.current = 0;

    warningTriggeredRef.current = false;

    /**
     * Reset analyser.
     */
    teardownAnalyser();

    setPhaseSynced("REQUESTING_PERMISSION");

    try {
      /**
       * Recorder increments its
       * session synchronously.
       */
      sessionRef.current = monitor.sessionId + 1;

      const sid = await monitor.start();

      sessionRef.current = sid;

      setSession(sid);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : describeMicError(err) || "Could not start monitoring.";

      setError(message);

      phaseRef.current = "ERROR";

      setPhase("ERROR");
    }
  }, [monitor, setPhaseSynced, teardownAnalyser]);

  // ------------------------------------------------------------
  // Stop monitoring
  // ------------------------------------------------------------

  const stopMonitoring = useCallback(() => {
    setPhaseSynced((previous) => {
      if (
        previous === "IDLE" ||
        previous === "COMPLETE" ||
        previous === "ERROR"
      ) {
        return previous;
      }

      return "STOPPING";
    });

    monitor.stop();

    teardownAnalyser();

    if (!monitor.isMonitoring()) {
      setPhaseSynced((previous) =>
        previous === "STOPPING" ? "COMPLETE" : previous,
      );
    }
  }, [monitor, setPhaseSynced, teardownAnalyser]);

  // ------------------------------------------------------------
  // New session
  // ------------------------------------------------------------

  const startNewSession = useCallback(() => {
    if (monitor.isMonitoring()) {
      return;
    }

    setError(null);

    setCurrentResult(null);

    setHistory([]);

    setDroppedInfo(null);

    setNoSpeechInfo(null);

    setQueueDepth(0);

    /**
     * Reset EMA.
     */
    smoothedRiskRef.current = null;

    persistentHighRiskRef.current = 0;

    warningTriggeredRef.current = false;

    teardownAnalyser();

    setPhaseSynced("IDLE");
  }, [monitor, setPhaseSynced, teardownAnalyser]);

  // ------------------------------------------------------------
  // Clear error
  // ------------------------------------------------------------

  const clearError = useCallback(() => {
    setError(null);

    if (phaseRef.current === "ERROR") {
      setPhaseSynced("IDLE");
    }
  }, [setPhaseSynced]);

  // ------------------------------------------------------------
  // Return
  // ------------------------------------------------------------

  return {
    phase,

    isMonitoring: phase !== "IDLE" && phase !== "ERROR" && phase !== "COMPLETE",

    startMonitoring,

    stopMonitoring,

    startNewSession,

    currentResult,

    history,

    droppedInfo,

    noSpeechInfo,

    queueDepth,

    error,

    session,

    liveAnalyser,

    clearError,
  };
}
