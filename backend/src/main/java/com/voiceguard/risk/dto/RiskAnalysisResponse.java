package com.voiceguard.risk.dto;

import com.voiceguard.risk.RiskLevel;
import java.math.BigDecimal;

/**
 * Response body for {@code POST /api/risk/analyze}.
 *
 * @param spoofProbability the validated input probability (0.0-1.0)
 * @param riskScore        integer score 0-100 (probability x 100)
 * @param riskLevel        LOW / MEDIUM / HIGH / CRITICAL
 * @param alert            true when the level warrants an alert
 * @param message          human-readable explanation of the verdict
 */
public record RiskAnalysisResponse(
        BigDecimal spoofProbability,
        int riskScore,
        RiskLevel riskLevel,
        boolean alert,
        String message
) {
}
