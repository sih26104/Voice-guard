package com.voiceguard.risk.analyze;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * Multipart entry point for full audio analysis.
 *
 * <pre>
 * POST /api/analyze   (multipart/form-data, field "file" = audio.wav)
 * </pre>
 *
 * <p>Thin adapter: validation and error mapping live in
 * {@code ApiExceptionHandler}; AI calls in {@link AiDetectionClient};
 * risk bands in the existing {@code RiskEngine}.
 */
@RestController
@RequestMapping("/api")
public class AudioAnalysisController {

    private static final Logger log = LoggerFactory.getLogger(AudioAnalysisController.class);

    private final AudioAnalysisService analysisService;

    public AudioAnalysisController(AudioAnalysisService analysisService) {
        this.analysisService = analysisService;
    }

    @PostMapping(value = "/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<AudioAnalysisResponse> analyze(
            @RequestParam(value = "file", required = false) MultipartFile file) {
        log.debug("analyze request: {} bytes", file == null ? 0 : file.getSize());
        AudioAnalysisResponse response = analysisService.analyze(file);
        return ResponseEntity.ok(response);
    }
}
