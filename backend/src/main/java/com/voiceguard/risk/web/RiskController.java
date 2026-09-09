package com.voiceguard.risk.web;

import com.voiceguard.risk.dto.RiskAnalysisRequest;
import com.voiceguard.risk.dto.RiskAnalysisResponse;
import com.voiceguard.risk.service.RiskEngine;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST endpoint exposing the prototype risk engine.
 *
 * <p>Thin HTTP adapter only: validation is declarative
 * ({@code @Valid}), errors are mapped by {@link ApiExceptionHandler},
 * and all risk logic lives in {@link RiskEngine}.
 */
@RestController
@RequestMapping("/api/risk")
public class RiskController {

    private static final Logger log = LoggerFactory.getLogger(RiskController.class);

    private final RiskEngine riskEngine;

    public RiskController(RiskEngine riskEngine) {
        this.riskEngine = riskEngine;
    }

    /**
     * Analyze an AI-computed spoof probability.
     *
     * <pre>
     * POST /api/risk/analyze
     * { "spoofProbability": 0.82 }
     * </pre>
     *
     * @return 200 with the risk verdict, or 400 for missing/invalid input
     */
    @PostMapping("/analyze")
    public ResponseEntity<RiskAnalysisResponse> analyze(@Valid @RequestBody RiskAnalysisRequest request) {
        RiskAnalysisResponse response = riskEngine.analyze(request);
        log.debug("risk analysis: probability={}, score={}, level={}",
                response.spoofProbability(), response.riskScore(), response.riskLevel());
        return ResponseEntity.ok(response);
    }
}
