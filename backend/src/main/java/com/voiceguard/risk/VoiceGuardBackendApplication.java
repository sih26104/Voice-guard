package com.voiceguard.risk;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * VoiceGuard backend entrypoint.
 *
 * <p>Currently hosts the risk engine REST API ({@code /api/risk/analyze}).
 * Intentionally excludes (per prototype scope): AI-service integration,
 * WebSocket push, database persistence, and authentication.
 */
@SpringBootApplication
public class VoiceGuardBackendApplication {

    public static void main(String[] args) {
        SpringApplication.run(VoiceGuardBackendApplication.class, args);
    }
}
