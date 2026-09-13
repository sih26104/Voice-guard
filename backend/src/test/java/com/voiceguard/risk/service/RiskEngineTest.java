package com.voiceguard.risk.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.voiceguard.risk.RiskLevel;
import com.voiceguard.risk.dto.RiskAnalysisRequest;
import com.voiceguard.risk.dto.RiskAnalysisResponse;
import java.math.BigDecimal;
import org.junit.jupiter.api.Test;

/**
 * Pure unit tests for the risk engine (no Spring context, no HTTP).
 * Covers all four levels, every band boundary, rounding, and the
 * engine's own out-of-range guard.
 */
class RiskEngineTest {

    private final RiskEngine engine = new RiskEngine();

    private RiskAnalysisResponse analyze(String probability) {
        return engine.analyze(new RiskAnalysisRequest(new BigDecimal(probability)));
    }

    // ---- level mapping through the public API -------------------------------

    @Test
    void lowBand() {
        RiskAnalysisResponse response = analyze("0.10");
        assertThat(response.riskScore()).isEqualTo(10);
        assertThat(response.riskLevel()).isEqualTo(RiskLevel.LOW);
        assertThat(response.alert()).isFalse();
        assertThat(response.message()).isEqualTo("Low probability of synthetic speech");
    }

    @Test
    void mediumBand() {
        RiskAnalysisResponse response = analyze("0.45");
        assertThat(response.riskScore()).isEqualTo(45);
        assertThat(response.riskLevel()).isEqualTo(RiskLevel.MEDIUM);
        assertThat(response.alert()).isFalse();
        assertThat(response.message()).isEqualTo("Moderate probability of synthetic speech");
    }

    @Test
    void highBand() {
        RiskAnalysisResponse response = analyze("0.70");
        assertThat(response.riskScore()).isEqualTo(70);
        assertThat(response.riskLevel()).isEqualTo(RiskLevel.HIGH);
        assertThat(response.alert()).isTrue();
        assertThat(response.message()).isEqualTo("Elevated probability of synthetic speech");
    }

    @Test
    void criticalBand() {
        RiskAnalysisResponse response = analyze("0.95");
        assertThat(response.riskScore()).isEqualTo(95);
        assertThat(response.riskLevel()).isEqualTo(RiskLevel.CRITICAL);
        assertThat(response.alert()).isTrue();
        assertThat(response.message()).isEqualTo("High probability of synthetic speech");
    }

    // ---- band boundaries (inclusive edges, no gaps) --------------------------

    @Test
    void boundariesMapToCorrectBands() {
        assertThat(analyze("0.00").riskLevel()).isEqualTo(RiskLevel.LOW);
        assertThat(analyze("0.30").riskLevel()).isEqualTo(RiskLevel.LOW);
        assertThat(analyze("0.40").riskLevel()).isEqualTo(RiskLevel.LOW);   // 40 -> LOW
        assertThat(analyze("0.41").riskLevel()).isEqualTo(RiskLevel.MEDIUM); // 41 -> MEDIUM
        assertThat(analyze("0.60").riskLevel()).isEqualTo(RiskLevel.MEDIUM); // 60 -> MEDIUM
        assertThat(analyze("0.61").riskLevel()).isEqualTo(RiskLevel.HIGH);   // 61 -> HIGH
        assertThat(analyze("0.80").riskLevel()).isEqualTo(RiskLevel.HIGH);   // 80 -> HIGH
        assertThat(analyze("0.81").riskLevel()).isEqualTo(RiskLevel.CRITICAL); // 81 -> CRITICAL
        assertThat(analyze("1.00").riskLevel()).isEqualTo(RiskLevel.CRITICAL);
    }

    @Test
    void forScoreRejectsOutOfRangeScores() {
        assertThatThrownBy(() -> RiskLevel.forScore(-1))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> RiskLevel.forScore(101))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // ---- probability -> score conversion ------------------------------------

    @Test
    void scoreRoundsHalfUp() {
        assertThat(engine.toScore(new BigDecimal("0.825"))).isEqualTo(83);
        assertThat(engine.toScore(new BigDecimal("0.824"))).isEqualTo(82);
        assertThat(engine.toScore(new BigDecimal("0.82"))).isEqualTo(82);
    }

    @Test
    void scoreHandlesScaleVariants() {
        assertThat(engine.toScore(new BigDecimal("0.8200"))).isEqualTo(82);
        assertThat(engine.toScore(new BigDecimal("0.5"))).isEqualTo(50);
        assertThat(engine.toScore(BigDecimal.ZERO)).isZero();
        assertThat(engine.toScore(BigDecimal.ONE)).isEqualTo(100);
        assertThat(engine.toScore(new BigDecimal("0.999"))).isEqualTo(100);
        assertThat(engine.toScore(new BigDecimal("0.001"))).isZero();
    }

    @Test
    void responseEchoesInputProbability() {
        RiskAnalysisResponse response = analyze("0.82");
        assertThat(response.spoofProbability()).isEqualByComparingTo("0.82");
    }
}
