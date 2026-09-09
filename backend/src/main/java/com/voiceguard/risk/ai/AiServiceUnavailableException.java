package com.voiceguard.risk.ai;

/**
 * The FastAI AI service could not be reached or timed out.
 *
 * <p>Mapped to HTTP 503 by {@code ApiExceptionHandler}; the response body
 * never includes stack traces or connection details beyond the service
 * name.
 */
public class AiServiceUnavailableException extends RuntimeException {

    public AiServiceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
