package com.voiceguard.risk.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;

/**
 * Request body for {@code POST /api/risk/analyze}.
 *
 * <p>{@code spoofProbability} must be present and within [0.0, 1.0].
 * Invalid input is rejected with HTTP 400 by the controller layer —
 * the engine never sees (and never silently clamps) invalid values.
 */
public record RiskAnalysisRequest(

        @NotNull(message = "spoofProbability is required")
        @DecimalMin(value = "0.0", inclusive = true, message = "spoofProbability must be >= 0.0")
        @DecimalMax(value = "1.0", inclusive = true, message = "spoofProbability must be <= 1.0")
        BigDecimal spoofProbability
) {
}
