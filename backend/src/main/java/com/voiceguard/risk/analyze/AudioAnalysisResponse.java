package com.voiceguard.risk.analyze;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.voiceguard.risk.RiskLevel;
import com.voiceguard.risk.ai.AiDetectionResult;
import com.voiceguard.risk.dto.RiskAnalysisResponse;
import java.math.BigDecimal;

/**
 * Response body for {@code POST /api/analyze}: the existing risk verdict
 * plus the AI detection metadata, in one payload.
 *
 * <pre>
 * {
 *   "spoofProbability": 0.5958,
 *   "riskScore": 60,
 *   "riskLevel": "MEDIUM",
 *   "alert": false,
 *   "label": "SPOOF",
 *   "duration_seconds": 2.0,
 *   "sample_rate": 16000,
 *   "inference_time_ms": 1238.2,
 *   "message": "Moderate probability of synthetic speech"
 * }
 * </pre>
 */
public record AudioAnalysisResponse(
        BigDecimal spoofProbability,
        int riskScore,
        RiskLevel riskLevel,
        boolean alert,
        String label,
        Double duration_seconds,
        Integer sample_rate,
        Double inference_time_ms,
        String message
) {

    static AudioAnalysisResponse combine(RiskAnalysisResponse risk, AiDetectionResult ai) {
        return new AudioAnalysisResponse(
                risk.spoofProbability(),
                risk.riskScore(),
                risk.riskLevel(),
                risk.alert(),
                ai.label(),
                ai.durationSeconds(),
                ai.sampleRate(),
                ai.inferenceTimeMs(),
                risk.message()
        );
    }
}
