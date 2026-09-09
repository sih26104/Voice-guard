package com.voiceguard.risk.web;

import com.voiceguard.risk.ai.AiInvalidResponseException;
import com.voiceguard.risk.ai.AiServiceUnavailableException;
import com.voiceguard.risk.ai.AiDetectionClient.AudioUploadTooLargeException;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;

/**
 * Maps request failures to HTTP 400 with a small JSON error body.
 *
 * <p>Invalid values are rejected, never clamped or defaulted — per the
 * prototype contract the caller must fix the input.
 */
@RestControllerAdvice
public class ApiExceptionHandler {

    /** Simple error payload: {@code {"errors": ["..."]}}. */
    public record ErrorResponse(List<String> errors) {
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(MethodArgumentNotValidException ex) {
        List<String> errors = ex.getBindingResult().getFieldErrors().stream()
                .map(error -> error.getField() + ": " + error.getDefaultMessage())
                .toList();
        return badRequest(errors);
    }

    /** Body absent, malformed JSON, or wrong type (e.g. a string where a number is expected). */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ErrorResponse> handleNotReadable(HttpMessageNotReadableException ex) {
        return badRequest(List.of("request body is missing or malformed JSON"));
    }

    /** Query/path parameter of the wrong type; kept for future endpoints. */
    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ErrorResponse> handleTypeMismatch(MethodArgumentTypeMismatchException ex) {
        return badRequest(List.of("invalid value for parameter '" + ex.getName() + "'"));
    }

    // ---- multipart /api/analyze error mapping ------------------------------

    /** Missing "file" part in a multipart request. */
    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ErrorResponse> handleMissingPart(MissingServletRequestPartException ex) {
        return badRequest(List.of("file: audio file is required"));
    }

    /** Empty upload rejected by the analysis service. */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
        return badRequest(List.of(ex.getMessage()));
    }

    /** Upload beyond the configured size cap (Spring multipart limit). */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ErrorResponse> handleMaxUpload(MaxUploadSizeExceededException ex) {
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(new ErrorResponse(List.of("audio file exceeds the upload size limit")));
    }

    /** Upload beyond the client-side cap checked in AiDetectionClient. */
    @ExceptionHandler(AudioUploadTooLargeException.class)
    public ResponseEntity<ErrorResponse> handleUploadTooLarge(AudioUploadTooLargeException ex) {
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(new ErrorResponse(List.of(ex.getMessage())));
    }

    /** FastAPI is down, unreachable, or timed out. */
    @ExceptionHandler(AiServiceUnavailableException.class)
    public ResponseEntity<ErrorResponse> handleAiUnavailable(AiServiceUnavailableException ex) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(new ErrorResponse(List.of("AI detection service is unavailable")));
    }

    /** FastAPI answered but the response was unusable. */
    @ExceptionHandler(AiInvalidResponseException.class)
    public ResponseEntity<ErrorResponse> handleAiInvalid(AiInvalidResponseException ex) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(new ErrorResponse(List.of("AI detection service returned an invalid response")));
    }

    private ResponseEntity<ErrorResponse> badRequest(List<String> errors) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new ErrorResponse(errors));
    }
}
