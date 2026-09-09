package com.voiceguard.risk.analyze;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

/**
 * Real end-to-end smoke test: audio -> Spring Boot /api/analyze ->
 * FastAPI /predict -> RiskEngine -> JSON.
 *
 * <p><b>Requires the FastAPI service on {@code http://127.0.0.1:8000}</b>
 * (with the trained checkpoint loaded). Self-skips when it is not
 * running, so normal test runs stay fast and offline. Run separately:
 *
 * <pre>
 * mvnw test -Dtest=AudioAnalysisEndToEndIT
 * </pre>
 */
class AudioAnalysisEndToEndIT {

    private static final String FASTAPI_HEALTH = "http://127.0.0.1:8000/health";
    private static final String ANALYZE_URL = "http://127.0.0.1:8080/api/analyze";

    private static boolean fastApiReachable() {
        try {
            HttpURLConnection connection =
                    (HttpURLConnection) URI.create(FASTAPI_HEALTH).toURL().openConnection();
            connection.setConnectTimeout(2000);
            connection.setReadTimeout(2000);
            int status = connection.getResponseCode();
            connection.disconnect();
            return status == 200;
        } catch (IOException unavailable) {
            return false;
        }
    }

    @BeforeAll
    static void requireFastApi() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                fastApiReachable(), "FastAPI not running on 127.0.0.1:8000 — skipping e2e test");
    }

    /** Minimal valid mono 16-bit PCM WAV: 0.5 s 440 Hz sine at 16 kHz. */
    private static byte[] synthesizeWav() {
        int sampleRate = 16_000;
        int samples = sampleRate / 2;

        ByteArrayOutputStream pcm = new ByteArrayOutputStream(samples * 2);
        for (int i = 0; i < samples; i++) {
            int value = (int) (0.4 * Math.sin(2 * Math.PI * 440.0 * i / sampleRate) * 32767);
            pcm.write(value & 0xFF);
            pcm.write((value >> 8) & 0xFF);
        }
        byte[] pcmBytes = pcm.toByteArray();

        // RIFF header: 44 bytes standard PCM WAV header.
        int dataSize = pcmBytes.length;
        ByteArrayOutputStream wav = new ByteArrayOutputStream(44 + dataSize);
        byte[] header = new byte[44];
        writeAscii(header, 0, "RIFF");
        writeIntLe(header, 4, 36 + dataSize);
        writeAscii(header, 8, "WAVE");
        writeAscii(header, 12, "fmt ");
        writeIntLe(header, 16, 16);          // PCM chunk size
        writeShortLe(header, 20, 1);         // PCM format
        writeShortLe(header, 22, 1);         // mono
        writeIntLe(header, 24, sampleRate);  // sample rate
        writeIntLe(header, 28, sampleRate * 2); // byte rate (16-bit mono)
        writeShortLe(header, 32, 2);         // block align
        writeShortLe(header, 34, 16);        // bits per sample
        writeAscii(header, 36, "data");
        writeIntLe(header, 40, dataSize);
        wav.writeBytes(header);
        wav.writeBytes(pcmBytes);
        return wav.toByteArray();
    }

    private static void writeAscii(byte[] target, int offset, String text) {
        for (int i = 0; i < text.length(); i++) {
            target[offset + i] = (byte) text.charAt(i);
        }
    }

    private static void writeIntLe(byte[] target, int offset, int value) {
        target[offset] = (byte) value;
        target[offset + 1] = (byte) (value >> 8);
        target[offset + 2] = (byte) (value >> 16);
        target[offset + 3] = (byte) (value >> 24);
    }

    private static void writeShortLe(byte[] target, int offset, int value) {
        target[offset] = (byte) value;
        target[offset + 1] = (byte) (value >> 8);
    }

    @Test
    void fullFlowAudioToRiskVerdict() {
        TestRestTemplate rest = new TestRestTemplate();

        MultiValueMap<String, Object> form = new LinkedMultiValueMap<>();
        form.add("file", new ByteArrayResource(synthesizeWav()) {
            @Override
            public String getFilename() {
                return "tone.wav";
            }
        });

        ResponseEntity<String> response = rest.postForEntity(
                URI.create(ANALYZE_URL), form, String.class);

        assertThat(response.getStatusCode().value()).isEqualTo(200);
        String body = response.getBody();
        assertThat(body).contains("\"spoofProbability\"");
        assertThat(body).contains("\"riskLevel\"");
        assertThat(body).contains("\"label\"");
        assertThat(body).contains("\"sample_rate\":16000");
        System.out.println("E2E response: " + body);
    }
}
