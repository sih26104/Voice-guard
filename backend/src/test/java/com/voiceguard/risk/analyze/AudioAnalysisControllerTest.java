package com.voiceguard.risk.analyze;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.voiceguard.risk.ai.AiDetectionResult;
import com.voiceguard.risk.ai.AiDetectionClient;
import com.voiceguard.risk.ai.AiDetectionClient.AudioUploadTooLargeException;
import com.voiceguard.risk.ai.AiInvalidResponseException;
import com.voiceguard.risk.ai.AiServiceUnavailableException;
import com.voiceguard.risk.ai.AiDetectionProperties;
import com.voiceguard.risk.service.RiskEngine;
import java.math.BigDecimal;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

/**
 * Web-slice tests for {@code POST /api/analyze} with a **mocked**
 * {@link AiDetectionClient} — FastAPI is never required. Covers the
 * success path, all four risk bands, probability pass-through into the
 * real {@link RiskEngine}, and every error mapping.
 */
@WebMvcTest(AudioAnalysisController.class)
@Import({AudioAnalysisService.class, RiskEngine.class})
@EnableConfigurationProperties(AiDetectionProperties.class)
class AudioAnalysisControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private AiDetectionClient aiClient;

    private MockMultipartFile wav(String name) {
        return new MockMultipartFile("file", name, "audio/wav",
                new byte[]{0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4}); // tiny fake payload
    }

    private void aiReturns(double probability, String label) {
        when(aiClient.detect(any())).thenReturn(new AiDetectionResult(
                BigDecimal.valueOf(probability), label, 2.0, 16000, 1238.2));
    }

    // ---- success path --------------------------------------------------------

    @Test
    void successfulAnalysisCombinesAiAndRisk() throws Exception {
        aiReturns(0.5958, "SPOOF");
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.spoofProbability").value(0.5958))
                .andExpect(jsonPath("$.riskScore").value(60))
                .andExpect(jsonPath("$.riskLevel").value("MEDIUM"))
                .andExpect(jsonPath("$.alert").value(false))
                .andExpect(jsonPath("$.label").value("SPOOF"))
                .andExpect(jsonPath("$.duration_seconds").value(2.0))
                .andExpect(jsonPath("$.sample_rate").value(16000))
                .andExpect(jsonPath("$.inference_time_ms").value(1238.2))
                .andExpect(jsonPath("$.message").value("Moderate probability of synthetic speech"));
    }

    // ---- all four risk bands through the real RiskEngine ----------------------

    @Test
    void lowRisk() throws Exception {
        aiReturns(0.10, "BONAFIDE");
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(10))
                .andExpect(jsonPath("$.riskLevel").value("LOW"))
                .andExpect(jsonPath("$.alert").value(false));
    }

    @Test
    void mediumRisk() throws Exception {
        aiReturns(0.45, "SPOOF");
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(45))
                .andExpect(jsonPath("$.riskLevel").value("MEDIUM"))
                .andExpect(jsonPath("$.alert").value(false));
    }

    @Test
    void highRisk() throws Exception {
        aiReturns(0.70, "SPOOF");
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(70))
                .andExpect(jsonPath("$.riskLevel").value("HIGH"))
                .andExpect(jsonPath("$.alert").value(true));
    }

    @Test
    void criticalRisk() throws Exception {
        aiReturns(0.82, "SPOOF");
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(82))
                .andExpect(jsonPath("$.riskLevel").value("CRITICAL"))
                .andExpect(jsonPath("$.alert").value(true))
                .andExpect(jsonPath("$.message").value("High probability of synthetic speech"));
    }

    /** The AI probability must reach the engine unchanged (no clamping). */
    @Test
    void probabilityPassedToRiskEngineUnchanged() throws Exception {
        aiReturns(0.6149, "SPOOF"); // 61.49 -> 61 (HIGH), proving no rounding to 60/MEDIUM
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.spoofProbability").value(0.6149))
                .andExpect(jsonPath("$.riskScore").value(61))
                .andExpect(jsonPath("$.riskLevel").value("HIGH"));
    }

    // ---- request validation ---------------------------------------------------

    @Test
    void missingFilePartRejected() throws Exception {
        // The service's own validation fires (file == null -> 400).
        mockMvc.perform(multipart("/api/analyze"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors[0]").value("No audio file provided"));
    }

    @Test
    void emptyFileRejected() throws Exception {
        MockMultipartFile empty = new MockMultipartFile("file", "clip.wav", "audio/wav", new byte[0]);
        mockMvc.perform(multipart("/api/analyze").file(empty))
                .andExpect(status().isBadRequest());
    }

    // ---- AI service failure mapping ---------------------------------------------

    @Test
    void aiUnavailableMapsTo503() throws Exception {
        when(aiClient.detect(any()))
                .thenThrow(new AiServiceUnavailableException("AI detection service is unavailable",
                        new java.net.ConnectException("refused")));
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.errors[0]").value("AI detection service is unavailable"));
    }

    @Test
    void invalidAiResponseMapsTo502() throws Exception {
        when(aiClient.detect(any()))
                .thenThrow(new AiInvalidResponseException("AI service response invalid: missing field"));
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.errors[0]").value(
                        "AI detection service returned an invalid response"));
    }

    @Test
    void oversizedUploadMapsTo413() throws Exception {
        when(aiClient.detect(any()))
                .thenThrow(new AudioUploadTooLargeException("audio file exceeds the 25 MiB limit"));
        mockMvc.perform(multipart("/api/analyze").file(wav("clip.wav")))
                .andExpect(status().isPayloadTooLarge());
    }
}
