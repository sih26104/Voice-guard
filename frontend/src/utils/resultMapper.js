/**
 * Adapts the real backend AudioAnalysisResponse into the shape consumed by
 * the dashboard components (ConfidenceMeter, warning banner, theme).
 *
 * NO fabricated values: every field traces back to the backend payload.
 */

/**
 * Human-readable verdict text derived only from real backend fields.
 */
function classificationFor(label, riskLevel) {
  if (label === 'BONAFIDE') {
    return riskLevel === 'LOW' ? 'AUTHENTIC HUMAN VOICE' : 'LIKELY HUMAN VOICE';
  }
  // SPOOF
  if (riskLevel === 'LOW') return 'LOW-CONFIDENCE SPOOF SIGNAL';
  if (riskLevel === 'MEDIUM') return 'SYNTHETIC VOICE LIKELY';
  return 'SYNTHETIC VOICE DETECTED';
}

/** Recommendation text derived from the real verdict (not invented metrics). */
function directiveFor(label, riskLevel, alert) {
  if (label === 'BONAFIDE' && riskLevel === 'LOW' && !alert) {
    return 'No strong synthetic-speech signals detected. Standard verification practices still apply for high-value transactions.';
  }
  if (riskLevel === 'CRITICAL' || alert) {
    return 'High probability of synthetic speech. Do not act on this audio; verify the speaker through a separate, trusted channel.';
  }
  if (riskLevel === 'HIGH') {
    return 'Elevated probability of synthetic speech. Continue with additional verification before trusting this audio.';
  }
  if (riskLevel === 'MEDIUM') {
    return 'Moderate synthetic-speech probability. Treat this result as inconclusive and verify through a second channel if the decision is high-stakes.';
  }
  return label === 'SPOOF'
    ? 'Some synthetic-speech indicators present, below the alert threshold. Re-test with cleaner audio if in doubt.'
    : 'No strong synthetic-speech signals detected. Standard verification practices still apply.';
}

/**
 * Map the backend payload to the dashboard's result object.
 *
 * @param {object} api normalized backend response (see utils/api.js)
 * @param {object} meta UI metadata: { fileName, source }
 * @returns {object} result for DashboardPage / ConfidenceMeter
 */
export function mapApiResponse(api, { fileName = null, source = 'mic', chunkNumber = null } = {}) {
  // spoofProbability (0-1, probability the audio is SYNTHETIC) drives the
  // ConfidenceMeter "AI score" arc directly: 0 = fully human, 100 = fully AI.
  const aiScore = Math.max(0, Math.min(100, api.spoofProbability * 100));

  const type = api.label === 'SPOOF' ? 'ai' : 'human';
  const showWarning =
    (api.label === 'SPOOF' && (api.riskLevel === 'LOW' || api.riskLevel === 'MEDIUM')) ||
    (api.label === 'BONAFIDE' && api.riskLevel !== 'LOW');

  return {
    // Identity / metadata (UI-only; not backend fields)
    fileName,
    source,
    chunkNumber,
    analyzedAt: new Date().toISOString(),

    // Core real values
    label: api.label,
    type,
    spoofProbability: api.spoofProbability,
    aiScore,
    riskLevel: api.riskLevel,
    riskScore: api.riskScore,
    alert: api.alert,
    message: api.message,
    durationSeconds: api.duration_seconds,
    sampleRate: api.sample_rate,
    inferenceTimeMs: api.inference_time_ms,

    // Derived, truthful UI strings
    classification: classificationFor(api.label, api.riskLevel),
    directive: directiveFor(api.label, api.riskLevel, api.alert),

    // ConfidenceMeter heuristics: show the amber "mixed" treatment when the
    // verdict and risk band disagree, or the band is borderline.
    warning: showWarning,
  };
}
