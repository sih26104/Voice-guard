package com.voiceguard.risk.web;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.voiceguard.risk.service.RiskEngine;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

/**
 * Web-slice tests for {@code POST /api/risk/analyze} using the real
 * {@link RiskEngine}. Covers the four levels end-to-end through JSON,
 * plus HTTP 400 handling for below-range, above-range, missing and
 * malformed input.
 */
@WebMvcTest(RiskController.class)
@Import(RiskEngine.class)
class RiskControllerTest {

    @Autowired
    private MockMvc mockMvc;

    private String body(String probabilityJson) {
        return "{\"spoofProbability\": " + probabilityJson + "}";
    }

    // ---- valid requests ------------------------------------------------------

    @Test
    void criticalExampleFromSpec() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("0.82")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.spoofProbability").value(0.82))
                .andExpect(jsonPath("$.riskScore").value(82))
                .andExpect(jsonPath("$.riskLevel").value("CRITICAL"))
                .andExpect(jsonPath("$.alert").value(true))
                .andExpect(jsonPath("$.message").value("High probability of synthetic speech"));
    }

    @Test
    void lowLevel() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("0.10")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(10))
                .andExpect(jsonPath("$.riskLevel").value("LOW"))
                .andExpect(jsonPath("$.alert").value(false));
    }

    @Test
    void mediumLevel() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("0.45")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(45))
                .andExpect(jsonPath("$.riskLevel").value("MEDIUM"))
                .andExpect(jsonPath("$.alert").value(false));
    }

    @Test
    void highLevel() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("0.70")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(70))
                .andExpect(jsonPath("$.riskLevel").value("HIGH"))
                .andExpect(jsonPath("$.alert").value(true));
    }

    @Test
    void boundaryValuesAccepted() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("0")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(0))
                .andExpect(jsonPath("$.riskLevel").value("LOW"));

        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("1")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.riskScore").value(100))
                .andExpect(jsonPath("$.riskLevel").value("CRITICAL"));
    }

    // ---- invalid input must be rejected with 400, never clamped --------------

    @Test
    void probabilityBelowZeroRejected() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("-0.1")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors[0]").value(
                        "spoofProbability: spoofProbability must be >= 0.0"));
    }

    @Test
    void probabilityAboveOneRejected() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body("1.5")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors[0]").value(
                        "spoofProbability: spoofProbability must be <= 1.0"));
    }

    @Test
    void missingFieldRejected() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.errors[0]").value(
                        "spoofProbability: spoofProbability is required"));
    }

    @Test
    void emptyBodyRejected() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(""))
                .andExpect(status().isBadRequest());
    }

    @Test
    void malformedJsonRejected() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{spoofProbability: 0.5}"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void nonNumericValueRejected() throws Exception {
        mockMvc.perform(post("/api/risk/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"spoofProbability\": \"high\"}"))
                .andExpect(status().isBadRequest());
    }
}
