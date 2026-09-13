import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import {
  Mic,
  MicOff,
  ShieldAlert,
  AlertTriangle,
  BarChart3,
  Activity,
  Cpu,
  X,
  CheckCircle2,
  RotateCcw,
  Lock,
  VolumeX,
  FileUp,
  FileAudio,
  Loader2,
} from "lucide-react";
import ConfidenceMeter from "../components/ConfidenceMeter";
import AudioWaveform from "../components/AudioWaveform";
import DashboardBackground from "../components/DashboardBackground";
import { mapApiResponse } from "../utils/resultMapper";
import { analyzeUploadedAudio } from "../utils/uploadAnalyzer";
import { takeHandoffFile } from "../utils/heroFileHandoff";
import { useMonitor } from "../hooks/useMonitor";

const HIGH_RISK_LEVELS = ["HIGH", "CRITICAL"];
const UPLOAD_MAX_BYTES = 15 * 1024 * 1024; // 15 MB guard — mirrors a sane FastAPI upload limit

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "--";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Looks-like-WAV check: extension or MIME variant. The backend remains the strict validator. */
function looksLikeWav(file) {
  const name = (file.name || "").toLowerCase();
  const type = (file.type || "").toLowerCase();
  return (
    name.endsWith(".wav") ||
    ["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"].includes(type)
  );
}

export default function DashboardPage() {
  const location = useLocation();

  const {
    phase,
    isMonitoring,
    startMonitoring,
    stopMonitoring,
    startNewSession,
    currentResult,
    history,
    droppedInfo,
    noSpeechInfo,
    queueDepth,
    error,
    liveAnalyser,
    clearError,
  } = useMonitor();

  // ---- Upload vs monitoring view (Upload Audio button on Home activates upload)
  const [mode, setMode] = useState(
    location.state?.mode === "upload" ? "upload" : "mic",
  );
  useEffect(() => {
    if (location.state?.mode === "upload") setMode("upload");
  }, [location.state]);

  // ---- Upload flow state (independent of the mic pipeline; same backend endpoint)
  // A WAV picked on the homepage hero arrives via the in-memory handoff;
  // it goes through the exact same real validation + /api/analyze flow.
  const [file, setFile] = useState(() => takeHandoffFile());
  const [uploadPhase, setUploadPhase] = useState("IDLE"); // IDLE | ANALYZING | COMPLETE | ERROR
  const [uploadError, setUploadError] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const acceptFile = (f) => {
    setUploadError(null);
    setUploadResult(null);
    setUploadPhase("IDLE");
    if (!f) {
      setFile(null);
      return;
    }
    if (!looksLikeWav(f)) {
      setFile(null);
      setUploadError(
        "Unsupported file. Please choose a .wav audio file (16 kHz mono is ideal).",
      );
      return;
    }
    if (f.size <= 0) {
      setFile(null);
      setUploadError("That file appears to be empty.");
      return;
    }
    if (f.size > UPLOAD_MAX_BYTES) {
      setFile(null);
      setUploadError(
        `File is too large (${formatBytes(f.size)}). Maximum size is 15 MB.`,
      );
      return;
    }
    setFile(f);
  };

  const analyzeUpload = async () => {
    if (!file || uploadPhase === "ANALYZING") return;
    setUploadError(null);
    setUploadResult(null);
    setUploadPhase("ANALYZING");
    try {
      // Real backend calls — windows are analyzed sequentially through the
      // same POST /api/analyze endpoint the mic uses (see uploadAnalyzer.js).
      const api = await analyzeUploadedAudio(file);
      setUploadResult(
        mapApiResponse(api, { fileName: file.name, source: "upload" }),
      );
      setUploadPhase("COMPLETE");
    } catch (err) {
      setUploadError(
        err && err.message
          ? err.message
          : "Upload analysis failed. Check that the backend is running and try again.",
      );
      setUploadPhase("ERROR");
    }
  };

  const clearUpload = () => {
    setFile(null);
    setUploadResult(null);
    setUploadError(null);
    setUploadPhase("IDLE");
  };

  // ---- Status pill: one explicit source of truth (the phase)
  const statusPill = {
    IDLE: { color: "text-slate-500", dot: "bg-slate-400" },
    REQUESTING_PERMISSION: { color: "text-amber-600", dot: "bg-amber-400" },
    RECORDING: { color: "text-emerald-600", dot: "bg-shield-human" },
    ANALYZING: { color: "text-amber-600", dot: "bg-amber-400" },
    NO_SPEECH: { color: "text-tech", dot: "bg-tech" },
    COMPLETE: { color: "text-emerald-600", dot: "bg-emerald-500" },
    STOPPING: { color: "text-slate-500", dot: "bg-slate-400" },
    ERROR: { color: "text-red-600", dot: "bg-red-500" },
  }[phase] || { color: "text-slate-500", dot: "bg-slate-400" };

  const canStart =
    phase === "IDLE" || phase === "ERROR" || phase === "COMPLETE";
  const canStop =
    isMonitoring && phase !== "STOPPING" && phase !== "REQUESTING_PERMISSION";
  const canNewSession =
    phase === "IDLE" || phase === "ERROR" || phase === "COMPLETE";

  // ---- Warning area: shown ONLY when the real backend response has alert === true
  const showAlert = Boolean(currentResult && currentResult.alert);
  const isSevereAlert =
    showAlert && HIGH_RISK_LEVELS.includes(currentResult.riskLevel);
  const [alertDismissed, setAlertDismissed] = React.useState(false);
  React.useEffect(() => {
    if (currentResult) setAlertDismissed(false); // re-arm on every new result
  }, [currentResult]);

  // ---- Probabilities: real data or "--" (never a fabricated default)
  const synthProb = currentResult ? currentResult.aiScore : null;
  const humanProb = currentResult ? 100 - currentResult.aiScore : null;
  const fmtProb = (v) => (v != null ? `${v.toFixed(1)}%` : "--");

  // ---- Live risk history plot (real session analyses only)
  const plotPoints = history.filter((h) => typeof h.spoofPercent === "number");
  const W = 600;
  const H = 140;
  const plotPath =
    plotPoints.length > 0
      ? plotPoints
          .map((h, i) => {
            const x =
              plotPoints.length === 1
                ? W / 2
                : (i / (plotPoints.length - 1)) * (W - 20) + 10;
            const y =
              H -
              15 -
              (Math.min(100, Math.max(0, h.spoofPercent)) / 100) * (H - 30);
            return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
          })
          .join(" ")
      : null;

  const historyRowColor = (level) =>
    HIGH_RISK_LEVELS.includes(level)
      ? "text-red-700 bg-red-50 border-red-300"
      : level === "MEDIUM"
        ? "text-amber-700 bg-amber-50 border-amber-300"
        : "text-emerald-700 bg-emerald-50 border-emerald-300";

  return (
    <div className="relative min-h-screen">
      <DashboardBackground />

      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-8 pt-8 pb-16">
        {/* ============ HEADER ============ */}
        <div className="mb-5 dash-rise">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-wine tracking-tight">
            Echo<span className="es-hero-gradient">Shield</span> Dashboard
          </h1>
        </div>

        {/* ============ MODE TABS ============ */}
        <div
          className="es-card inline-flex gap-1.5 p-1.5 mb-6 dash-rise"
          role="tablist"
          aria-label="Analysis mode"
          style={{ animationDelay: ".05s" }}
        >
          <button
            role="tab"
            aria-selected={mode === "mic"}
            onClick={() => setMode("mic")}
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 cursor-pointer ${
              mode === "mic"
                ? "bg-rose text-white shadow-btn"
                : "text-wine hover:bg-blush/60"
            }`}
          >
            <Mic className="w-4 h-4" />
            Microphone Monitoring
          </button>
          <button
            role="tab"
            aria-selected={mode === "upload"}
            onClick={() => setMode("upload")}
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 cursor-pointer ${
              mode === "upload"
                ? "bg-rose text-white shadow-btn"
                : "text-wine hover:bg-blush/60"
            }`}
          >
            <FileUp className="w-4 h-4" />
            Upload File
          </button>
        </div>

        {/* ================================================================ */}
        {/* ===================== MICROPHONE MONITORING ==================== */}
        {/* ================================================================ */}
        {mode === "mic" && (
          <>
            {/* ============ WARNING AREA (only when backend alert === true) ============ */}
            {showAlert && !alertDismissed && (
              <div
                className={`flex items-start justify-between gap-3 p-4 rounded-2xl mb-5 border ${
                  isSevereAlert
                    ? "bg-red-50 border-red-300"
                    : "bg-amber-50 border-amber-300"
                }`}
              >
                <div className="flex items-start space-x-2.5">
                  <ShieldAlert
                    className={`w-5 h-5 shrink-0 mt-0.5 ${isSevereAlert ? "text-red-600" : "text-amber-600"}`}
                  />
                  <div>
                    <div
                      className={`text-sm font-bold font-mono ${
                        isSevereAlert ? "text-red-700" : "text-amber-700"
                      }`}
                    >
                      ALERT FLAGGED — RISK LEVEL: {currentResult.riskLevel}
                    </div>
                    <p className="text-xs text-shield-muted mt-1">
                      {currentResult.message ||
                        "The audio was flagged by the analysis engine."}
                      {synthProb != null && (
                        <span className="font-mono">
                          {" "}
                          (synthetic probability {synthProb.toFixed(1)}%)
                        </span>
                      )}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setAlertDismissed(true)}
                  className="p-1.5 rounded-lg text-shield-muted hover:text-wine hover:bg-blush transition-colors cursor-pointer shrink-0"
                  aria-label="Dismiss alert"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* ============ MONITORING STATUS + CONTROLS ============ */}
            <div
              className="es-card es-card-hover p-5 sm:p-6 mb-5 dash-rise"
              style={{ animationDelay: ".1s" }}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                  <span className="text-xs font-mono uppercase tracking-widest text-rose-hover font-bold">
                    Microphone Monitoring
                  </span>
                  <span
                    className={`inline-flex items-center gap-2 text-xs font-mono px-3 py-1 rounded-full bg-blush/50 border border-blush ${statusPill.color}`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full ${statusPill.dot} ${
                        phase === "RECORDING" || phase === "ANALYZING"
                          ? "animate-pulse"
                          : ""
                      }`}
                    />
                    {phase}
                  </span>
                  {isMonitoring && (
                    <span
                      className="inline-flex dash-eq items-end gap-[3px] h-3.5"
                      aria-hidden="true"
                    >
                      <span style={{ height: "60%", animationDelay: "0s" }} />
                      <span
                        style={{ height: "100%", animationDelay: ".18s" }}
                      />
                      <span style={{ height: "45%", animationDelay: ".36s" }} />
                      <span style={{ height: "85%", animationDelay: ".54s" }} />
                      <span style={{ height: "55%", animationDelay: ".72s" }} />
                    </span>
                  )}
                </div>
                <span className="text-[11px] font-mono text-shield-muted">
                  {isMonitoring ? "Session active" : "No active session"} •{" "}
                  {history.length} chunk{history.length === 1 ? "" : "s"}{" "}
                  analyzed
                  {isMonitoring && queueDepth > 0 && (
                    <span className="text-tech font-bold">
                      {" "}
                      • {queueDepth} in queue
                    </span>
                  )}
                </span>
              </div>

              {/* Buttons: disabled states follow the single lifecycle phase */}
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => startMonitoring().catch(() => {})}
                  disabled={!canStart}
                  className={`flex-1 py-3.5 px-6 rounded-2xl text-sm font-bold tracking-wide uppercase transition-all duration-300 flex items-center justify-center space-x-2.5 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed ${
                    canStart ? "btn-primary" : "bg-blush/60 text-wine/50"
                  }`}
                >
                  {phase === "REQUESTING_PERMISSION" ? (
                    <>
                      <Cpu className="w-5 h-5 animate-pulse" />
                      <span>Requesting Mic Access…</span>
                    </>
                  ) : phase === "RECORDING" ? (
                    <>
                      <Cpu className="w-5 h-5 animate-pulse" />
                      <span>Recording…</span>
                    </>
                  ) : phase === "ANALYZING" ? (
                    <>
                      <Cpu className="w-5 h-5 animate-pulse" />
                      <span>Analyzing Chunk…</span>
                    </>
                  ) : (
                    <>
                      <Mic className="w-5 h-5" />
                      <span>Start Monitoring</span>
                    </>
                  )}
                </button>

                <button
                  onClick={stopMonitoring}
                  disabled={!canStop}
                  className="flex-1 py-3.5 px-6 rounded-2xl text-sm font-bold tracking-wide uppercase bg-red-500 hover:bg-red-600 disabled:bg-blush/60 disabled:text-wine/50 text-white transition-all duration-300 flex items-center justify-center space-x-2.5 cursor-pointer disabled:cursor-not-allowed"
                >
                  <MicOff className="w-5 h-5" />
                  <span>
                    {phase === "STOPPING" ? "Stopping…" : "Stop Monitoring"}
                  </span>
                </button>

                <button
                  onClick={startNewSession}
                  disabled={!canNewSession}
                  title="Clear results and start a fresh session (no page reload)"
                  className="btn-secondary py-3.5 px-5 rounded-2xl text-sm font-bold uppercase flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span>New Session</span>
                </button>
              </div>

              {/* Bounded-pipeline drop notice (real event, never silent) */}
              {droppedInfo && (
                <div className="flex items-start space-x-2 p-3 rounded-2xl bg-amber-50 border border-amber-300 text-[11px] font-mono text-amber-700 mt-3">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
                  <span>
                    Chunk #{droppedInfo.chunkNumber} skipped —{" "}
                    {droppedInfo.reason}
                  </span>
                </div>
              )}

              {/* No-speech notice: informational only — NOT a deepfake result. A
                  gated chunk is never sent for analysis, so no spoof/human
                  probability exists for it. */}
              {noSpeechInfo && (
                <div className="flex items-start space-x-2 p-3 rounded-2xl bg-blue-50 border border-tech/30 text-[11px] font-mono text-tech mt-3">
                  <VolumeX className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>
                    Chunk #{noSpeechInfo.chunkNumber}: No speech detected —
                    monitoring continues.
                  </span>
                </div>
              )}

              {/* Error banner (permission, capture, conversion, API failures) */}
              {error && (
                <div className="flex items-start justify-between gap-3 p-3.5 rounded-2xl bg-red-50 border border-red-300 text-xs font-mono text-red-700 mt-3">
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </div>
                  <button
                    onClick={clearError}
                    className="p-1 rounded text-red-600 hover:text-red-800 hover:bg-red-100 cursor-pointer shrink-0"
                    aria-label="Clear error"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* ============ CURRENT VOICE ANALYSIS (directly below controls) ============ */}
            <div
              className={`es-card p-5 sm:p-6 mb-5 dash-rise ${
                showAlert && !alertDismissed ? "dash-alert-glow" : ""
              }`}
              style={{ animationDelay: ".15s" }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm sm:text-base font-extrabold text-wine tracking-tight flex items-center gap-2">
                  <Activity className="w-5 h-5 text-rose" />
                  Current Voice Analysis
                </h2>
                {currentResult ? (
                  <span className="text-[11px] font-mono text-shield-muted">
                    Chunk #{currentResult.chunkNumber}
                  </span>
                ) : (
                  <span className="text-[11px] font-mono text-shield-muted">
                    Awaiting first chunk
                  </span>
                )}
              </div>

              {/* Probabilities: real values or "--" */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="p-4 rounded-2xl bg-blush/40 border border-blush text-center">
                  <div className="text-[10px] font-mono text-shield-muted uppercase">
                    Synthetic Probability
                  </div>
                  <div
                    className={`text-3xl font-extrabold font-mono mt-1 ${
                      synthProb != null
                        ? "text-red-600"
                        : "text-shield-muted/70"
                    }`}
                  >
                    {fmtProb(synthProb)}
                  </div>
                </div>
                <div className="p-4 rounded-2xl bg-blush/40 border border-blush text-center">
                  <div className="text-[10px] font-mono text-shield-muted uppercase">
                    Human Probability
                  </div>
                  <div
                    className={`text-3xl font-extrabold font-mono mt-1 ${
                      humanProb != null
                        ? "text-emerald-600"
                        : "text-shield-muted/70"
                    }`}
                  >
                    {fmtProb(humanProb)}
                  </div>
                </div>
              </div>

              {/* Risk score + level (real backend values) */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
                <div className="p-3.5 rounded-2xl bg-blush/40 border border-blush sm:col-span-2">
                  <div className="flex items-center justify-between text-[11px] font-mono mb-2">
                    <span className="text-shield-muted uppercase">
                      Risk Score
                    </span>
                    <span className="font-bold text-wine">
                      {currentResult
                        ? `${currentResult.riskScore} / 100`
                        : "--"}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-blush rounded-full overflow-hidden">
                    {currentResult && (
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${
                          HIGH_RISK_LEVELS.includes(currentResult.riskLevel)
                            ? "bg-red-500"
                            : currentResult.riskLevel === "MEDIUM"
                              ? "bg-amber-500"
                              : "bg-shield-human"
                        }`}
                        style={{
                          width: `${Math.min(100, Math.max(0, currentResult.riskScore))}%`,
                        }}
                      />
                    )}
                  </div>
                </div>
                <div className="p-3.5 rounded-2xl bg-blush/40 border border-blush">
                  <div className="text-[10px] font-mono text-shield-muted uppercase">
                    Risk Level
                  </div>
                  <div
                    className={`text-base font-extrabold font-mono mt-1 ${
                      currentResult
                        ? HIGH_RISK_LEVELS.includes(currentResult.riskLevel)
                          ? "text-red-600"
                          : currentResult.riskLevel === "MEDIUM"
                            ? "text-amber-600"
                            : "text-emerald-600"
                        : "text-shield-muted/70"
                    }`}
                  >
                    {currentResult ? currentResult.riskLevel : "--"}
                  </div>
                </div>
              </div>

              {/* Confidence visualization (real data or honest empty state) */}
              <ConfidenceMeter
                aiScore={synthProb}
                hasData={Boolean(currentResult)}
                classification={
                  currentResult
                    ? currentResult.classification
                    : "WAITING FOR INPUT"
                }
                riskLevel={currentResult ? currentResult.riskLevel : "IDLE"}
              />

              {/* Real mic signal while the stream is open; honest idle baseline otherwise */}
              <div className="mt-4">
                <AudioWaveform
                  analyser={liveAnalyser}
                  isActive={Boolean(liveAnalyser)}
                  colorScheme={currentResult ? currentResult.type : "blue"}
                  height={110}
                />
              </div>

              {/* Verdict label + message (real fields, no internal wiring details) */}
              {currentResult && (
                <div className="flex flex-col sm:flex-row sm:items-center gap-2 mt-4 text-[11px] font-mono">
                  <span
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border w-fit ${
                      currentResult.alert
                        ? "bg-red-50 text-red-700 border-red-300"
                        : "bg-emerald-50 text-emerald-700 border-emerald-300"
                    }`}
                  >
                    {currentResult.alert ? (
                      <ShieldAlert className="w-3.5 h-3.5" />
                    ) : (
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    )}
                    Label: {currentResult.label}
                  </span>
                  <span className="text-shield-muted bg-blush/40 p-2 rounded-xl border border-blush flex-1">
                    <span className="text-wine font-bold">Result: </span>
                    {currentResult.message || "—"}
                  </span>
                </div>
              )}
            </div>

            {/* ============ LIVE RISK HISTORY ============ */}
            <div
              className="es-card es-card-hover p-5 sm:p-6 mb-5 dash-rise"
              style={{ animationDelay: ".2s" }}
            >
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xs font-mono uppercase tracking-widest text-rose-hover font-bold flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Live Risk History
                </h2>
                <span className="text-[11px] font-mono text-shield-muted">
                  {plotPoints.length} real analysis point
                  {plotPoints.length === 1 ? "" : "s"}
                </span>
              </div>

              {plotPath ? (
                <svg
                  viewBox={`0 0 ${W} ${H}`}
                  className="w-full block"
                  preserveAspectRatio="none"
                  style={{ height: 140 }}
                >
                  {/* Risk band reference (0-40 LOW / 41-60 MEDIUM / 61-80 HIGH / 81-100 CRITICAL) */}
                  <rect
                    x="0"
                    y={H - 15 - 0.4 * (H - 30)}
                    width={W}
                    height={0.4 * (H - 30)}
                    fill="rgba(34,197,94,0.08)"
                  />
                  <rect
                    x="0"
                    y={H - 15 - 0.6 * (H - 30)}
                    width={W}
                    height={0.2 * (H - 30)}
                    fill="rgba(245,158,11,0.08)"
                  />
                  <rect
                    x="0"
                    y={H - 15 - 0.8 * (H - 30)}
                    width={W}
                    height={0.2 * (H - 30)}
                    fill="rgba(239,68,68,0.08)"
                  />
                  <rect
                    x="0"
                    y="0"
                    width={W}
                    height={0.2 * (H - 30)}
                    fill="rgba(239,68,68,0.12)"
                  />
                  {/* Real data line */}
                  <path
                    d={plotPath}
                    fill="none"
                    stroke="#3B82F6"
                    strokeWidth="2.5"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  {plotPoints.map((h, i) => {
                    const x =
                      plotPoints.length === 1
                        ? W / 2
                        : (i / (plotPoints.length - 1)) * (W - 20) + 10;
                    const y =
                      H -
                      15 -
                      (Math.min(100, Math.max(0, h.spoofPercent)) / 100) *
                        (H - 30);
                    return (
                      <circle
                        key={h.chunkNumber ?? i}
                        cx={x}
                        cy={y}
                        r="4"
                        fill={
                          HIGH_RISK_LEVELS.includes(h.riskLevel)
                            ? "#EF4444"
                            : h.riskLevel === "MEDIUM"
                              ? "#F59E0B"
                              : "#22C55E"
                        }
                      />
                    );
                  })}
                </svg>
              ) : (
                <div className="p-6 rounded-2xl bg-blush/40 border border-blush text-center">
                  <p className="text-sm font-mono text-shield-muted">
                    No chunk history yet.
                  </p>
                  <p className="text-xs text-shield-muted/80 mt-1">
                    Start monitoring to see analysis results.
                  </p>
                </div>
              )}

              <p className="text-[10px] font-mono text-shield-muted/80 mt-2">
                One point per analyzed chunk. No simulated points are shown.
              </p>
            </div>

            {/* ============ CHUNK ANALYSIS HISTORY ============ */}
            <div
              className="es-card es-card-hover p-5 sm:p-6 mb-5 dash-rise"
              style={{ animationDelay: ".25s" }}
            >
              <h2 className="text-xs font-mono uppercase tracking-widest text-rose-hover font-bold flex items-center gap-2 mb-3">
                <BarChart3 className="w-4 h-4" />
                Chunk Analysis History
              </h2>

              {history.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-blush text-shield-muted text-[10px] uppercase">
                        <th className="py-2 px-2">Chunk #</th>
                        <th className="py-2 px-2">Synthetic %</th>
                        <th className="py-2 px-2">Risk</th>
                        <th className="py-2 px-2">Label</th>
                        <th className="py-2 px-2">Processing time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-blush">
                      {[...history].reverse().map((row) => (
                        <tr
                          key={row.chunkNumber}
                          className="hover:bg-blush/40 transition-colors"
                        >
                          <td className="py-2.5 px-2 text-wine font-bold">
                            #{row.chunkNumber}
                          </td>
                          <td className="py-2.5 px-2 text-wine">
                            {row.spoofPercent != null
                              ? `${row.spoofPercent.toFixed(1)}%`
                              : "--"}
                          </td>
                          <td className="py-2.5 px-2">
                            <span
                              className={`px-2 py-0.5 rounded border ${historyRowColor(row.riskLevel)}`}
                            >
                              {row.riskLevel}
                            </span>
                          </td>
                          <td className="py-2.5 px-2 text-shield-muted">
                            {row.label}
                          </td>
                          <td className="py-2.5 px-2 text-shield-muted">
                            {row.inferenceTimeMs != null
                              ? `${row.inferenceTimeMs.toFixed(0)} ms`
                              : "--"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-5 rounded-2xl bg-blush/40 border border-blush text-center">
                  <p className="text-sm font-mono text-shield-muted">
                    No chunk history yet.
                  </p>
                  <p className="text-xs text-shield-muted/80 mt-1">
                    Completed analyses will be listed here — one row per
                    analyzed chunk.
                  </p>
                </div>
              )}
            </div>

            {/* ============ PRIVACY ============ */}
            <div
              className="es-card es-card-hover p-4 sm:p-5 flex items-start space-x-3 dash-rise"
              style={{ animationDelay: ".3s" }}
            >
              <Lock className="w-4 h-4 text-wine shrink-0 mt-0.5" />
              <div>
                <div className="text-[11px] font-mono uppercase tracking-widest text-rose-hover font-bold mb-1">
                  Privacy
                </div>
                <p className="text-xs text-shield-muted leading-relaxed">
                  Audio is processed temporarily for voice analysis. Chunks are
                  converted to WAV in the browser, analyzed in memory, and are
                  not stored.
                </p>
              </div>
            </div>
          </>
        )}

        {/* ================================================================ */}
        {/* ========================= UPLOAD FILE ========================== */}
        {/* ================================================================ */}
        {mode === "upload" && (
          <div
            className="es-card es-card-hover p-5 sm:p-8 dash-rise"
            style={{ animationDelay: ".1s" }}
          >
            <div className="flex items-center mb-1">
              <h2 className="text-sm sm:text-base font-extrabold text-wine tracking-tight flex items-center gap-2">
                <FileUp className="w-5 h-5 text-rose" />
                Upload a WAV File
              </h2>
            </div>
            <p className="text-xs text-shield-muted mb-6">
              Analyze a pre-recorded WAV audio file for synthetic voice
              characteristics. The file is processed in memory and not stored.
            </p>

            {/* Drop zone / file chooser (label-wrapped input = accessible) */}
            {!file && uploadPhase !== "COMPLETE" && (
              <label
                className={`block cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
                  dragOver
                    ? "border-rose bg-blush/50"
                    : "border-blush bg-blush/20 hover:bg-blush/40"
                }`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  acceptFile(e.dataTransfer.files && e.dataTransfer.files[0]);
                }}
              >
                <input
                  type="file"
                  accept=".wav,audio/wav,audio/x-wav"
                  className="sr-only"
                  onChange={(e) =>
                    acceptFile(e.target.files && e.target.files[0])
                  }
                />
                <FileAudio className="w-10 h-10 text-wine mx-auto" />
                <p className="mt-3 text-sm font-bold text-wine">
                  Click to choose a WAV file, or drag &amp; drop it here
                </p>
                <p className="mt-1 text-xs text-shield-muted">
                  WAV audio only • up to 15 MB
                </p>
              </label>
            )}

            {/* Selected file */}
            {file && uploadPhase !== "COMPLETE" && (
              <div className="mt-4">
                <div className="flex items-center gap-3 p-4 rounded-2xl bg-blush/40 border border-blush">
                  <div className="w-11 h-11 rounded-xl bg-white border border-blush flex items-center justify-center shrink-0">
                    <FileAudio className="w-5 h-5 text-wine" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-bold text-wine truncate">
                      {file.name}
                    </div>
                    <div className="text-[11px] font-mono text-shield-muted">
                      {formatBytes(file.size)}
                    </div>
                  </div>
                  <button
                    onClick={clearUpload}
                    className="btn-secondary px-3 py-2 rounded-xl text-xs font-bold cursor-pointer shrink-0"
                  >
                    Clear
                  </button>
                </div>

                <button
                  onClick={analyzeUpload}
                  disabled={uploadPhase === "ANALYZING"}
                  className="btn-primary mt-4 w-full py-3.5 rounded-2xl text-sm font-bold tracking-wide uppercase flex items-center justify-center gap-2.5 cursor-pointer disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {uploadPhase === "ANALYZING" ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin text-tech" />
                      Analyzing file…
                    </>
                  ) : (
                    <>
                      <Cpu className="w-5 h-5" />
                      Analyze File
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Upload error (unsupported type, oversized, failed upload, backend/AI unavailable) */}
            {uploadError && (
              <div className="flex items-start justify-between gap-3 p-3.5 rounded-2xl bg-red-50 border border-red-300 text-xs font-mono text-red-700 mt-4">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{uploadError}</span>
                </div>
                <button
                  onClick={() => setUploadError(null)}
                  className="p-1 rounded text-red-600 hover:text-red-800 hover:bg-red-100 cursor-pointer shrink-0"
                  aria-label="Clear upload error"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Real backend result for the uploaded file */}
            {uploadPhase === "COMPLETE" && uploadResult && (
              <div className="mt-6 rounded-2xl border border-blush overflow-hidden">
                <div className="flex items-center justify-between gap-3 p-4 bg-blush/40 border-b border-blush">
                  <div className="flex items-center gap-3 min-w-0">
                    <FileAudio className="w-5 h-5 text-wine shrink-0" />
                    <div className="min-w-0">
                      <div className="text-sm font-bold text-wine truncate">
                        {uploadResult.fileName}
                      </div>
                      <div className="text-[11px] font-mono text-shield-muted">
                        {typeof uploadResult.durationSeconds === "number"
                          ? `${uploadResult.durationSeconds.toFixed(2)} s`
                          : "duration --"}
                        {typeof uploadResult.sampleRate === "number"
                          ? ` • ${uploadResult.sampleRate} Hz`
                          : ""}
                        {uploadResult.inferenceTimeMs != null
                          ? ` • analyzed in ${uploadResult.inferenceTimeMs.toFixed(0)} ms`
                          : ""}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={clearUpload}
                    className="btn-secondary px-3 py-2 rounded-xl text-xs font-bold cursor-pointer shrink-0"
                  >
                    Analyze another file
                  </button>
                </div>

                <div className="p-4 sm:p-6 grid sm:grid-cols-2 gap-4 bg-white">
                  <div className="p-4 rounded-2xl bg-blush/40 border border-blush text-center">
                    <div className="text-[10px] font-mono text-shield-muted uppercase">
                      Synthetic Probability
                    </div>
                    <div
                      className={`text-3xl font-extrabold font-mono mt-1 ${
                        uploadResult.aiScore > 65
                          ? "text-red-600"
                          : uploadResult.aiScore < 40
                            ? "text-emerald-600"
                            : "text-amber-600"
                      }`}
                    >
                      {uploadResult.aiScore.toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-4 rounded-2xl bg-blush/40 border border-blush text-center">
                    <div className="text-[10px] font-mono text-shield-muted uppercase">
                      Human Probability
                    </div>
                    <div
                      className={`text-3xl font-extrabold font-mono mt-1 ${
                        uploadResult.aiScore > 65
                          ? "text-shield-muted/70"
                          : "text-emerald-600"
                      }`}
                    >
                      {(100 - uploadResult.aiScore).toFixed(1)}%
                    </div>
                  </div>

                  <div className="sm:col-span-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3.5 rounded-2xl bg-blush/40 border border-blush">
                      <div className="text-[10px] font-mono text-shield-muted uppercase">
                        Label
                      </div>
                      <div className="text-base font-extrabold font-mono mt-1 text-wine">
                        {uploadResult.label}
                      </div>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-blush/40 border border-blush">
                      <div className="text-[10px] font-mono text-shield-muted uppercase">
                        Risk Score
                      </div>
                      <div className="text-base font-extrabold font-mono mt-1 text-wine">
                        {uploadResult.riskScore} / 100
                      </div>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-blush/40 border border-blush">
                      <div className="text-[10px] font-mono text-shield-muted uppercase">
                        Risk Level
                      </div>
                      <div
                        className={`text-base font-extrabold font-mono mt-1 ${
                          HIGH_RISK_LEVELS.includes(uploadResult.riskLevel)
                            ? "text-red-600"
                            : uploadResult.riskLevel === "MEDIUM"
                              ? "text-amber-600"
                              : "text-emerald-600"
                        }`}
                      >
                        {uploadResult.riskLevel}
                      </div>
                    </div>
                  </div>

                  <div className="sm:col-span-2 text-xs text-shield-muted leading-relaxed p-3 rounded-2xl bg-blush/30 border border-blush">
                    <span className="text-wine font-bold">
                      Classification:{" "}
                    </span>
                    {uploadResult.classification}. {uploadResult.directive}
                    {uploadResult.message && (
                      <span className="block mt-1 font-mono text-[11px]">
                        <span className="text-wine font-bold">Result: </span>
                        {uploadResult.message}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Upload privacy note */}
            <div className="flex items-start space-x-3 mt-6 p-4 rounded-2xl bg-blush/30 border border-blush">
              <Lock className="w-4 h-4 text-wine shrink-0 mt-0.5" />
              <p className="text-xs text-shield-muted leading-relaxed">
                The selected file is held in browser memory only while the
                analysis request runs. It is not displayed as a playable
                recording and not persisted by this application.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
