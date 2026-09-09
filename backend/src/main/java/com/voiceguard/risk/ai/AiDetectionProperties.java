package com.voiceguard.risk.ai;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration for the FastAPI AI-detection service connection.
 *
 * <p>Bound from {@code application.properties} (prefix {@code ai}):
 * the base URL is configured once here and injected — never hard-coded
 * at call sites.
 *
 * <pre>
 * ai.base-url=http://127.0.0.1:8000
 * ai.connect-timeout-ms=3000
 * ai.read-timeout-ms=20000
 * ai.max-upload-bytes=26214400
 * </pre>
 */
@ConfigurationProperties(prefix = "ai")
public record AiDetectionProperties(
        String baseUrl,
        long connectTimeoutMs,
        long readTimeoutMs,
        long maxUploadBytes
) {
    public AiDetectionProperties {
        if (baseUrl == null || baseUrl.isBlank()) {
            baseUrl = "http://127.0.0.1:8000";
        }
        if (connectTimeoutMs <= 0) {
            connectTimeoutMs = 3_000;
        }
        if (readTimeoutMs <= 0) {
            readTimeoutMs = 20_000;
        }
        if (maxUploadBytes <= 0) {
            maxUploadBytes = 25L * 1024 * 1024;
        }
    }

    public String predictUrl() {
        String base = baseUrl.endsWith("/")
                ? baseUrl.substring(0, baseUrl.length() - 1)
                : baseUrl;
        return base + "/predict";
    }
}
