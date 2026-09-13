/**
 * EchoShield API client.
 *
 * Real backend: Spring Boot `POST {API_URL}/api/analyze`
 *   - multipart/form-data, field "file" = WAV audio
 *   - response: { spoofProbability, riskScore, riskLevel, alert, label,
 *                 duration_seconds, sample_rate, inference_time_ms, message }
 *
 * Base URL selection:
 *   - VITE_API_URL set  → that absolute URL is used directly. Note: the
 *     backend currently sends no CORS headers, so the target must be
 *     same-origin or CORS-enabled for browser calls to succeed.
 *   - VITE_API_URL unset → relative "/api" is used. In `npm run dev` the
 *     Vite dev server proxies it to the local development backend at
 *     http://127.0.0.1:8080 (see vite.config.js), which is the default
 *     target; in production it expects a same-origin reverse proxy.
 */

const RAW_BASE = import.meta.env.VITE_API_URL;

/** Absolute base actually used for requests ('' = relative/same-origin). */
const API_BASE = RAW_BASE ? String(RAW_BASE).replace(/\/+$/, '') : '';

const ANALYZE_ENDPOINT = `${API_BASE}/api/analyze`;

/** Expected shape of the backend AudioAnalysisResponse (all fields validated before use). */
function parseAnalyzeResponse(data) {
  if (!data || typeof data !== 'object') {
    throw new ApiError('Backend returned an unreadable response.');
  }

  const prob = Number(data.spoofProbability);
  if (Number.isNaN(prob) || prob < 0 || prob > 1) {
    throw new ApiError('Backend returned an invalid spoofProbability.');
  }

  const riskLevel = String(data.riskLevel || '');
  if (!['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].includes(riskLevel)) {
    throw new ApiError('Backend returned an invalid riskLevel.');
  }

  const label = String(data.label || '');
  if (!['BONAFIDE', 'SPOOF'].includes(label)) {
    throw new ApiError('Backend returned an invalid label.');
  }

  return {
    spoofProbability: prob,
    riskScore: Number(data.riskScore),
    riskLevel,
    alert: Boolean(data.alert),
    label,
    duration_seconds: data.duration_seconds != null ? Number(data.duration_seconds) : null,
    sample_rate: data.sample_rate != null ? Number(data.sample_rate) : null,
    inference_time_ms: data.inference_time_ms != null ? Number(data.inference_time_ms) : null,
    message: typeof data.message === 'string' ? data.message : '',
  };
}

/** Error thrown by analyzeAudio; carries a user-facing message and optional HTTP status. */
export class ApiError extends Error {
  constructor(message, { status = null, cause = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.cause = cause;
  }
}

/**
 * Send a WAV clip to the real backend analysis endpoint.
 *
 * @param {Blob|File} wavBlob WAV audio to analyze
 * @param {string} [filename] optional download/remote filename for the part
 * @returns {Promise<object>} normalized analysis result
 */
export async function analyzeAudio(wavBlob, filename = 'audio.wav') {
  if (!wavBlob) throw new ApiError('No audio provided for analysis.');

  const form = new FormData();
  // The Spring controller reads @RequestParam("file"); the part type is
  // explicit so the backend forwards a correct content type to FastAPI.
  form.append('file', wavBlob, filename);

  let response;
  try {
    response = await fetch(ANALYZE_ENDPOINT, { method: 'POST', body: form });
  } catch (err) {
    // Network failure (backend down, CORS blocked, DNS, offline).
    // User-facing message deliberately omits internal URLs/endpoints.
    throw new ApiError(
      'Could not reach the analysis backend. ' +
        'Make sure the backend is running and reachable, then try again.',
      { cause: err }
    );
  }

  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      if (body && Array.isArray(body.errors) && body.errors.length > 0) {
        detail = body.errors.join('; ');
      }
    } catch {
      // Non-JSON error body; keep generic detail.
    }
    throw new ApiError(`Analysis failed (HTTP ${response.status})${detail ? `: ${detail}` : '.'}`, {
      status: response.status,
    });
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    throw new ApiError('Backend returned a malformed response.', { cause: err });
  }

  return parseAnalyzeResponse(data);
}
