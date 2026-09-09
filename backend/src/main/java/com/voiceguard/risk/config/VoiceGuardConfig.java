package com.voiceguard.risk.config;

import com.voiceguard.risk.ai.AiDetectionClient;
import com.voiceguard.risk.ai.AiDetectionProperties;
import com.voiceguard.risk.service.RiskEngine;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

/**
 * Application wiring: AI-service configuration properties and shared
 * service beans.
 */
@Configuration
@EnableConfigurationProperties(AiDetectionProperties.class)
public class VoiceGuardConfig {

    @Bean
    public AiDetectionClient aiDetectionClient(
            AiDetectionProperties properties, RestClient.Builder builder) {
        return new AiDetectionClient(properties, builder);
    }

    @Bean
    public RiskEngine riskEngine() {
        return new RiskEngine();
    }
}
