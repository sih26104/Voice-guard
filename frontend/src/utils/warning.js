/**
 * Browser voice warning with cooldown.
 *
 * Speaks an audible alert when the REAL backend response has alert === true.
 * A cooldown prevents the browser from speaking on every 3-second chunk.
 * Silently skipped when SpeechSynthesis is unavailable.
 */

const COOLDOWN_MS = 30000; // at most one spoken warning per 30 s
let lastSpokenAt = 0;

export function speakHighRiskWarning() {
  try {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    const now = Date.now();
    if (now - lastSpokenAt < COOLDOWN_MS) return;
    lastSpokenAt = now;

    const utterance = new SpeechSynthesisUtterance(
      'Warning. High probability of synthetic voice detected.'
    );
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  } catch {
    // Voice warning is optional; never break monitoring over it.
  }
}
