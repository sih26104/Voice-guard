import React from 'react';
import { Link } from 'react-router-dom';
import {
  Mic,
  Scissors,
  FileAudio,
  Server,
  BrainCircuit,
  Gauge,
  BarChart3,
  AlertTriangle,
  ArrowRight,
  VolumeX,
} from 'lucide-react';

/**
 * Explains the real EchoShield pipeline. Monitoring mode records 3-second
 * chunks continuously; obvious no-speech chunks are skipped by a browser-side
 * energy gate before analysis. Every number shown is real model output, never
 * a simulated demo.
 */
const steps = [
  {
    icon: Mic,
    title: 'Microphone capture',
    text: 'You grant the browser microphone access once. While monitoring is running, the browser records temporary 3-second audio chunks continuously from a single microphone stream.',
  },
  {
    icon: VolumeX,
    title: 'Speech activity check',
    text: 'Before any analysis, a browser-side energy gate checks the chunk for speech. Silent or no-speech audio (room tone, quiet noise) is reported as NO_SPEECH and is never sent to the AI classifier.',
  },
  {
    icon: Scissors,
    title: 'Temporary 3-second window',
    text: 'Each chunk exists only in browser memory. Nothing is saved to disk, and the chunk is discarded after analysis.',
  },
  {
    icon: FileAudio,
    title: 'Browser converts audio to WAV',
    text: 'The browser converts every chunk to 16 kHz mono WAV — the exact input format the Wav2Vec2 spoof detector expects. This happens locally, before anything is sent.',
  },
  {
    icon: Server,
    title: 'Spring Boot POST /api/analyze',
    text: 'Each WAV chunk is sent as multipart form data to the Spring Boot backend at POST /api/analyze. The audio is processed in memory and never persisted.',
  },
  {
    icon: BrainCircuit,
    title: 'FastAPI → Wav2Vec2 spoof detector',
    text: 'Spring Boot forwards the audio to a FastAPI service running a fine-tuned Wav2Vec2 spoof-detection model on CPU. The model returns a real spoof probability (0–1) and a BONAFIDE / SPOOF label.',
  },
  {
    icon: Gauge,
    title: 'Risk engine',
    text: 'The existing risk engine maps the spoof probability to a 0–100 risk score with fixed bands: 0–40 LOW, 41–60 MEDIUM, 61–80 HIGH, 81–100 CRITICAL. HIGH and CRITICAL verdicts set an alert flag.',
  },
  {
    icon: BarChart3,
    title: 'Dashboard results',
    text: 'The dashboard shows the real spoof probability, the human probability (its complement), the risk score and level, and keeps a per-chunk history with a live risk graph.',
  },
  {
    icon: AlertTriangle,
    title: 'HIGH / CRITICAL warning',
    text: 'When a chunk is flagged HIGH or CRITICAL, the dashboard raises a visible warning and the browser plays a voice alert so you notice immediately.',
  },
];

export default function HowItWorksPage() {
  return (
    <div className="relative min-h-screen pt-10 pb-20 px-4 sm:px-8 max-w-5xl mx-auto">

      {/* Header */}
      <div className="text-center max-w-3xl mx-auto mb-14 space-y-4">
        <div className="inline-flex items-center space-x-2 px-4 py-1.5 rounded-full bg-blush/70 border border-wine/10 text-xs font-bold text-wine">
          <span>ECHOSHIELD PIPELINE</span>
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold text-wine tracking-tight">
          How EchoShield Works
        </h1>

        <p className="text-base sm:text-lg text-shield-muted font-light max-w-2xl mx-auto leading-relaxed">
          Every number shown in the dashboard is real model output — measured, never simulated.
        </p>
      </div>

      {/* Pipeline steps */}
      <div className="space-y-4 mb-14">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div
              key={idx}
              className="flex items-start gap-4 p-5 sm:p-6 rounded-2xl es-card es-card-hover"
            >
              <div className="shrink-0 w-11 h-11 rounded-xl bg-blush flex items-center justify-center">
                <Icon className="w-5 h-5 text-wine" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-mono text-tech font-bold">
                    STEP {idx + 1}
                  </span>
                  <h3 className="text-base font-bold text-wine">{step.title}</h3>
                </div>
                <p className="text-xs sm:text-sm text-shield-muted leading-relaxed">
                  {step.text}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Flow diagram */}
      <div className="rounded-3xl p-6 sm:p-8 bg-blush border border-blush text-center mb-12">
        <div className="text-xs font-mono uppercase tracking-widest text-wine font-bold mb-6">
          END-TO-END FLOW
        </div>

        <div className="flex flex-col md:flex-row items-center justify-center gap-3 text-xs font-mono">
          <div className="px-4 py-3 rounded-2xl bg-white border border-blush text-wine w-full md:w-auto">
            1. Mic chunk (3 s, speech-gated)
          </div>
          <ArrowRight className="w-4 h-4 text-tech hidden md:block rotate-90 md:rotate-0" aria-hidden="true" />
          <div className="px-4 py-3 rounded-2xl bg-white border border-blush text-wine w-full md:w-auto">
            2. Browser → 16 kHz WAV
          </div>
          <ArrowRight className="w-4 h-4 text-tech hidden md:block rotate-90 md:rotate-0" aria-hidden="true" />
          <div className="px-4 py-3 rounded-2xl bg-rose text-white font-bold w-full md:w-auto shadow-btn">
            3. POST /api/analyze → FastAPI → Wav2Vec2
          </div>
          <ArrowRight className="w-4 h-4 text-tech hidden md:block rotate-90 md:rotate-0" aria-hidden="true" />
          <div className="px-4 py-3 rounded-2xl bg-white border border-blush text-wine w-full md:w-auto">
            4. Spoof probability → risk engine → dashboard + alerts
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="text-center">
        <Link
          to="/dashboard"
          className="btn-primary inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-bold text-sm tracking-wide"
        >
          <span>Open the Monitoring Dashboard</span>
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>

    </div>
  );
}
