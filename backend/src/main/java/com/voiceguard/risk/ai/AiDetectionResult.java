package com.voiceguard.risk.ai;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.math.BigDecimal;

/**
 * Java mapping of the FastAPI {@code POST /predict} response.
 *
 * <pre>
 * {
 *   "spoofProbability": 0.5958,
 *   "label": "SPOOF",
 *   "duration_seconds": 2.0,
 *   "sample_rate": 16000,
 *   "inference_time_ms": 1238.2
 * }
 * </pre>
 *
 * @param spoofProbability probability in [0,1] that the audio is synthetic
 * @param label            "SPOOF" or "BONAFIDE"
 * @param durationSeconds  clip duration in seconds
 * @param sampleRate       sample rate of the analyzed audio (16000)
 * @param inferenceTimeMs  server-side inference time in milliseconds
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record AiDetectionResult(
        @JsonProperty("spoofProbability") BigDecimal spoofProbability,
        @JsonProperty("label") String label,
        @JsonProperty("duration_seconds") Double durationSeconds,
        @JsonProperty("sample_rate") Integer sampleRate,
        @JsonProperty("inference_time_ms") Double inferenceTimeMs
) {
    public AiDetectionResult {
        if (spoofProbability == null) {
            throw new IllegalArgumentException("AI response is missing spoofProbability");
        }
        if (spoofProbability.compareTo(BigDecimal.ZERO) < 0
                || spoofProbability.compareTo(BigDecimal.ONE) > 0) {
            throw new IllegalArgumentException(
                    "AI response spoofProbability outside [0,1]: " + spoofProbability);
        }
    }
}
