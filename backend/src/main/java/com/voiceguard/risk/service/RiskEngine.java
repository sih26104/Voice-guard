package com.voiceguard.risk.service;

import com.voiceguard.risk.RiskLevel;
import com.voiceguard.risk.dto.RiskAnalysisRequest;
import com.voiceguard.risk.dto.RiskAnalysisResponse;
import java.math.BigDecimal;
import java.math.RoundingMode;
import org.springframework.stereotype.Service;

/**
 * Prototype risk engine for VoiceGuard.
 *
 * <p>Pure calculation, no HTTP concerns — kept separate from the REST
 * controller so it can be unit-tested in isolation and later reused
 * (WebSocket pushes, scheduled jobs, ...) without web-layer coupling.
 *
 * <p>Mapping (prototype spec):
 * <pre>
 *   0-40   LOW
 *   41-60  MEDIUM
 *   61-80  HIGH
 *   81-100 CRITICAL
 * </pre>
 * The integer score is the probability scaled to percent, rounded
 * half-up (0.82 -&gt; 82). HIGH and CRITICAL set {@code alert=true}.
 */
@Service
public class RiskEngine {

    /** Multiplier from [0,1] probability to [0,100] percent score. */
    private static final BigDecimal HUNDRED = BigDecimal.valueOf(100);

    /**
     * Analyze a validated spoof probability and produce the risk verdict.
     *
     * @param request request whose {@code spoofProbability} is already
     *                validated to be within [0.0, 1.0] by the web layer
     * @return the risk analysis response (never {@code null})
     */
    public RiskAnalysisResponse analyze(RiskAnalysisRequest request) {
        BigDecimal probability = request.spoofProbability();
        int riskScore = toScore(probability);
        RiskLevel level = RiskLevel.forScore(riskScore);
        return new RiskAnalysisResponse(
                probability,
                riskScore,
                level,
                level.triggersAlert(),
                level.defaultMessage()
        );
    }

    /**
     * Convert a probability in [0.0, 1.0] to an integer percent score
     * in [0, 100], rounding half-up (0.825 -&gt; 83).
     */
    int toScore(BigDecimal probability) {
        return probability.multiply(HUNDRED)
                .setScale(0, RoundingMode.HALF_UP)
                .intValueExact();
    }
}
