# VoiceGuard Backend

Spring Boot 3.5 / Java 17 risk engine + audio-analysis API for SIH 2026
PS 26104.

## Architecture flow

```
Browser / Frontend
      |
      | multipart audio (file=<clip.wav>)
      v
Spring Boot  POST /api/analyze          (AudioAnalysisController)
      |
      | multipart audio (RestClient, configurable timeouts)
      v
FastAPI      POST /predict              (ai-service, CPU inference)
      |
      | { spoofProbability, label, duration_seconds, ... }
      v
Spring Boot  RiskEngine (existing, reused — bands 0-40/41-60/61-80/81-100)
      |
      v
Combined JSON verdict
```

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | multipart WAV → AI detection + risk verdict (new) |
| `POST /api/risk/analyze` | JSON probability → risk verdict (existing) |

## Start FastAPI (AI service)

```powershell
cd ai-service
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Requires `ai-service\models\trial\best_head.pt` (the trained head; the
encoder comes from the local Hugging Face cache). Model load takes ~10 s.

## Start Spring Boot

```powershell
cd backend
.\mvnw.cmd spring-boot:run
```

## FastAPI URL configuration

Configured once in `backend\src\main\resources\application.properties`
(never hard-coded at call sites):

```properties
ai.base-url=http://127.0.0.1:8000
ai.connect-timeout-ms=3000
ai.read-timeout-ms=20000
ai.max-upload-bytes=26214400
```

Override per-run, e.g.:

```powershell
.\mvnw.cmd spring-boot:run "-Dspring-boot.run.arguments=--ai.base-url=http://127.0.0.1:8000"
```

## POST /api/analyze

Request (multipart/form-data, field `file`):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/api/analyze -Method Post `
    -InFile "C:\path\to\clip.wav" -ContentType "audio/wav"
```

or:

```bash
curl -X POST http://127.0.0.1:8080/api/analyze -F "file=@C:/path/to/clip.wav"
```

Response:

```json
{
  "spoofProbability": 0.5958,
  "riskScore": 60,
  "riskLevel": "MEDIUM",
  "alert": false,
  "label": "SPOOF",
  "duration_seconds": 2.0,
  "sample_rate": 16000,
  "inference_time_ms": 1238.2,
  "message": "Moderate probability of synthetic speech"
}
```

Errors: `400` missing/empty file, `413` over 25 MiB, `502` invalid AI
response, `503` AI service unavailable. Stack traces and filesystem paths
are never exposed. Uploaded audio is never persisted or logged.

## Smoke test

A developer script sends a WAV to `POST /api/analyze` end-to-end
(Spring → FastAPI → RiskEngine) and prints the verdict. PowerShell
built-ins only (PowerShell 5.1+):

```powershell
# from the repository root
cd backend; .\mvnw.cmd spring-boot:run          # plus FastAPI in another terminal

powershell -ExecutionPolicy Bypass -File scripts\smoke-test-api.ps1                       # synthetic 2 s clip
powershell -ExecutionPolicy Bypass -File scripts\smoke-test-api.ps1 C:\path\to\clip.wav   # your own WAV
powershell -ExecutionPolicy Bypass -File scripts\smoke-test-api.ps1 -SpringBaseUrl http://127.0.0.1:8080
```

Exit codes: `0` success, `1` usage error (missing/oversized WAV),
`2` backend unreachable, `3` request failed or invalid response.
Uploaded audio is only streamed into the request - never printed,
copied, or saved.

## Tests

```powershell
cd backend
.\mvnw.cmd test                                   # full suite (FastAPI not required)
.\mvnw.cmd test "-Dtest=AudioAnalysisEndToEndIT"  # real e2e (needs FastAPI running)
```

Unit tests mock the AI client; the separate e2E test self-skips unless
FastAPI is up on `127.0.0.1:8000`.

## Limitations

- **Prototype, not real-time**: CPU inference is ~1+ second for a short
  clip (encoder forward pass), so true real-time/microphone streaming is
  out of scope for this build.
- The current model is a prototype baseline (test split: accuracy 72.67%,
  F1 71.89%) with TTS systems overlapping train/test — not production
  accuracy.
- No authentication, no database, no WebSockets yet.
