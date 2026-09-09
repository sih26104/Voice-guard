package com.voiceguard.risk.web;

import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

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

    private ResponseEntity<ErrorResponse> badRequest(List<String> errors) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new ErrorResponse(errors));
    }
}
