package com.voiceguard.risk.ai;

import java.io.IOException;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

/**
 * HTTP client for the FastAPI AI-detection service.
 *
 * <p>Sends the uploaded audio as {@code multipart/form-data} to
 * {@code POST /predict} and maps the JSON response to
 * {@link AiDetectionResult}. The AI base URL and timeouts come from
 * {@link AiDetectionProperties} — nothing hard-coded here.
 *
 * <p>Error contract:
 * <ul>
 *   <li>connection refused / timeouts → {@link AiServiceUnavailableException} (HTTP 503)</li>
 *   <li>unusable HTTP or JSON response → {@link AiInvalidResponseException} (HTTP 502)</li>
 * </ul>
 *
 * <p>Audio bytes live only inside the request; they are never persisted
 * or logged (only the byte size appears in debug logs).
 */
@Service
public class AiDetectionClient {

    private static final String MULTIPART_FIELD = "file";

    private final RestClient restClient;
    private final AiDetectionProperties properties;

    public AiDetectionClient(AiDetectionProperties properties, RestClient.Builder builder) {
        this.properties = properties;
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout((int) properties.connectTimeoutMs());
        factory.setReadTimeout((int) properties.readTimeoutMs());
        this.restClient = builder
                .requestFactory(factory)
                .baseUrl(properties.baseUrl())
                .build();
    }

    /** Test-visible constructor: inject a pre-built (e.g. mocked) client. */
    AiDetectionClient(AiDetectionProperties properties, RestClient restClient) {
        this.properties = properties;
        this.restClient = restClient;
    }

    /**
     * Forward the uploaded audio to FastAPI {@code /predict}.
     *
     * @throws AiServiceUnavailableException the AI service is down/unreachable
     *   or timed out (mapped to 503)
     * @throws AiInvalidResponseException the AI service answered but the
     *   response was unusable (mapped to 502)
     */
    public AiDetectionResult detect(MultipartFile audioFile) {
        if (audioFile == null || audioFile.isEmpty()) {
            throw new IllegalArgumentException("audio file is empty");
        }
        if (audioFile.getSize() > properties.maxUploadBytes()) {
            throw new AudioUploadTooLargeException(
                    "audio file exceeds the " + (properties.maxUploadBytes() / (1024 * 1024))
                            + " MiB limit");
        }

        byte[] audioBytes;
        try {
            audioBytes = audioFile.getResource().getContentAsByteArray();
        } catch (IOException ex) {
            throw new AiInvalidResponseException("could not read uploaded audio", ex);
        }

        ByteArrayResource audioPart = new ByteArrayResource(audioBytes) {
            @Override
            public String getFilename() {
                // A filename is required so Spring sends a proper
                // multipart part with the original name/extension.
                return audioFile.getOriginalFilename() != null
                        ? audioFile.getOriginalFilename()
                        : "audio.wav";
            }
        };

        MultiValueMap<String, Object> form = new LinkedMultiValueMap<>();
        form.add(MULTIPART_FIELD, audioPart);

        try {
            AiDetectionResult result = restClient.post()
                    .uri("/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(form)
                    .retrieve()
                    .body(AiDetectionResult.class);
            if (result == null) {
                throw new AiInvalidResponseException("AI service returned an empty body");
            }
            return result;
        } catch (ResourceAccessException ex) {
            // Connect/read timeout or unreachable host (cause: IOException).
            throw new AiServiceUnavailableException("AI detection service is unavailable", ex);
        } catch (RestClientResponseException ex) {
            // The AI service answered with an error status (4xx/5xx).
            throw new AiInvalidResponseException(
                    "AI service returned HTTP " + ex.getStatusCode().value(), ex);
        } catch (RestClientException ex) {
            // Deserialization failure, unexpected content type, malformed JSON.
            throw new AiInvalidResponseException("AI service response invalid", ex);
        } catch (IllegalArgumentException ex) {
            // Record validation (missing/out-of-range fields), if unwrapped.
            throw new AiInvalidResponseException(
                    "AI service response invalid: " + ex.getMessage(), ex);
        }
    }

    /** Raised when the upload exceeds {@code ai.max-upload-bytes}. */
    public static class AudioUploadTooLargeException extends RuntimeException {
        public AudioUploadTooLargeException(String message) {
            super(message);
        }
    }
}
