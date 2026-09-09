package com.voiceguard.risk.analyze;

import com.voiceguard.risk.ai.AiDetectionClient;
import com.voiceguard.risk.ai.AiDetectionResult;
import com.voiceguard.risk.dto.RiskAnalysisRequest;
import com.voiceguard.risk.dto.RiskAnalysisResponse;
import com.voiceguard.risk.service.RiskEngine;
import java.math.BigDecimal;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/**
 * Orchestrates the full analysis flow:
 *
 * <pre>
 * audio upload -> AiDetectionClient (FastAPI /predict) -> RiskEngine -> verdict
 * </pre>
 *
 * <p>Risk-band logic is **reused** from the existing {@link RiskEngine} —
 * nothing duplicated here. The service only adapts the AI result into the
 * request shape the engine already understands.
 */
@Service
public class AudioAnalysisService {

    private static final Logger log = LoggerFactory.getLogger(AudioAnalysisService.class);

    private final AiDetectionClient aiClient;
    private final RiskEngine riskEngine;

    public AudioAnalysisService(AiDetectionClient aiClient, RiskEngine riskEngine) {
        this.aiClient = aiClient;
        this.riskEngine = riskEngine;
    }

    /**
     * Analyze an uploaded audio clip end-to-end.
     *
     * @param audioFile multipart WAV upload (never persisted, never logged)
     * @return combined AI + risk response
     */
    public AudioAnalysisResponse analyze(MultipartFile audioFile) {
        if (audioFile == null || audioFile.isEmpty()) {
            throw new IllegalArgumentException("No audio file provided");
        }

        AiDetectionResult ai = aiClient.detect(audioFile);
        BigDecimal probability = ai.spoofProbability();

        // Reuse the existing risk engine for bands/score/alert/message.
        RiskAnalysisResponse risk = riskEngine.analyze(new RiskAnalysisRequest(probability));

        log.debug("audio analysis: probability={}, riskLevel={}, label={}",
                probability, risk.riskLevel(), ai.label());

        return AudioAnalysisResponse.combine(risk, ai);
    }
}
