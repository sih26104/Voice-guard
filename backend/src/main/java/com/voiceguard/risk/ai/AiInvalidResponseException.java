package com.voiceguard.risk.ai;

/**
 * The AI service responded, but the response was invalid (malformed JSON,
 * missing fields, out-of-range probability, unexpected status code).
 *
 * <p>Mapped to HTTP 502 by {@code ApiExceptionHandler}.
 */
public class AiInvalidResponseException extends RuntimeException {

    public AiInvalidResponseException(String message) {
        super(message);
    }

    public AiInvalidResponseException(String message, Throwable cause) {
        super(message, cause);
    }
}
