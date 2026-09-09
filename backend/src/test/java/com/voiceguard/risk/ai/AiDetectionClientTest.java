package com.voiceguard.risk.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withException;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.io.IOException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

/**
 * Unit tests for {@link AiDetectionClient} using a mocked HTTP server —
 * FastAPI is never started. Verifies URL, method, JSON mapping,
 * connection-error handling and invalid-response handling.
 */
class AiDetectionClientTest {

    private static final String BASE_URL = "http://127.0.0.1:8000";
    private static final String PREDICT_BODY = """
            {
              "spoofProbability": 0.5958,
              "label": "SPOOF",
              "duration_seconds": 2.0,
              "sample_rate": 16000,
              "inference_time_ms": 1238.2
            }
            """;

    private AiDetectionClient client;
    private MockRestServiceServer server;

    @BeforeEach
    void setUp() {
        AiDetectionProperties properties = new AiDetectionProperties(
                BASE_URL, 3_000, 20_000, 25L * 1024 * 1024);

        // Build a RestClient whose request factory is intercepted by
        // MockRestServiceServer, then inject it through the test-visible
        // constructor. No network access, no reflection field hacking.
        RestClient.Builder builder = RestClient.builder();
        server = MockRestServiceServer.bindTo(builder).build();
        RestClient restClient = builder
                .baseUrl(properties.baseUrl())
                .build();
        client = new AiDetectionClient(properties, restClient);
    }

    private MockMultipartFile wav() {
        return new MockMultipartFile("file", "clip.wav", "audio/wav",
                new byte[]{0x52, 0x49, 0x46, 0x46});
    }

    @Test
    void mapsFastApiResponseToDto() {
        server.expect(requestTo(BASE_URL + "/predict"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess(PREDICT_BODY, MediaType.APPLICATION_JSON));

        AiDetectionResult result = client.detect(wav());

        assertThat(result.spoofProbability()).isEqualByComparingTo("0.5958");
        assertThat(result.label()).isEqualTo("SPOOF");
        assertThat(result.durationSeconds()).isEqualTo(2.0);
        assertThat(result.sampleRate()).isEqualTo(16000);
        assertThat(result.inferenceTimeMs()).isEqualTo(1238.2);
        server.verify();
    }

    @Test
    void emptyFileRejected() {
        MockMultipartFile empty = new MockMultipartFile("file", "c.wav", "audio/wav", new byte[0]);
        assertThatThrownBy(() -> client.detect(empty))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void connectionRefusedMapsToUnavailable() {
        server.expect(requestTo(BASE_URL + "/predict"))
                .andRespond(withException(new IOException("connection refused")));

        assertThatThrownBy(() -> client.detect(wav()))
                .isInstanceOf(AiServiceUnavailableException.class);
    }

    @Test
    void aiServerErrorMapsToInvalidResponse502() {
        // The AI service *answered* with an error status: that is a bad
        // gateway situation (502), not an unreachable service (503).
        server.expect(requestTo(BASE_URL + "/predict"))
                .andRespond(withStatus(HttpStatus.SERVICE_UNAVAILABLE));

        assertThatThrownBy(() -> client.detect(wav()))
                .isInstanceOf(AiInvalidResponseException.class);
    }

    @Test
    void malformedJsonMapsToInvalidResponse() {
        server.expect(requestTo(BASE_URL + "/predict"))
                .andRespond(withSuccess("<html>not json</html>", MediaType.TEXT_HTML));

        assertThatThrownBy(() -> client.detect(wav()))
                .isInstanceOf(AiInvalidResponseException.class);
    }

    @Test
    void missingProbabilityMapsToInvalidResponse() {
        server.expect(requestTo(BASE_URL + "/predict"))
                .andRespond(withSuccess("{\"label\": \"SPOOF\"}", MediaType.APPLICATION_JSON));

        assertThatThrownBy(() -> client.detect(wav()))
                .isInstanceOf(AiInvalidResponseException.class);
    }

    @Test
    void outOfRangeProbabilityMapsToInvalidResponse() {
        server.expect(requestTo(BASE_URL + "/predict"))
                .andRespond(withSuccess(
                        "{\"spoofProbability\": 1.5, \"label\": \"SPOOF\"}",
                        MediaType.APPLICATION_JSON));

        assertThatThrownBy(() -> client.detect(wav()))
                .isInstanceOf(AiInvalidResponseException.class);
    }
}
