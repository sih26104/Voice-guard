package com.voiceguard.risk;

/**
 * Prototype risk bands agreed for the VoiceGuard prototype.
 *
 * <pre>
 *   0-40   LOW
 *   41-60  MEDIUM
 *   61-80  HIGH
 *   81-100 CRITICAL
 * </pre>
 *
 * <p>The score is an integer percent derived from the AI spoof
 * probability. Band edges are inclusive on both ends and adjacent bands
 * partition 0-100 without gaps.
 */
public enum RiskLevel {
    LOW(0, 40),
    MEDIUM(41, 60),
    HIGH(61, 80),
    CRITICAL(81, 100);

    private final int minScore;
    private final int maxScore;

    RiskLevel(int minScore, int maxScore) {
        this.minScore = minScore;
        this.maxScore = maxScore;
    }

    public int minScore() {
        return minScore;
    }

    public int maxScore() {
        return maxScore;
    }

    /**
     * Map an integer percent score (0-100) to its risk band.
     *
     * @throws IllegalArgumentException if the score is outside 0-100
     */
    public static RiskLevel forScore(int score) {
        for (RiskLevel level : values()) {
            if (score >= level.minScore && score <= level.maxScore) {
                return level;
            }
        }
        throw new IllegalArgumentException(
                "risk score must be within 0-100, got " + score);
    }

    /** Human-readable verdict shown in the API message field. */
    public String defaultMessage() {
        return switch (this) {
            case LOW -> "Low probability of synthetic speech";
            case MEDIUM -> "Moderate probability of synthetic speech";
            case HIGH -> "Elevated probability of synthetic speech";
            case CRITICAL -> "High probability of synthetic speech";
        };
    }

    /** Only HIGH and CRITICAL warrant raising an alert in the prototype. */
    public boolean triggersAlert() {
        return this == HIGH || this == CRITICAL;
    }
}
