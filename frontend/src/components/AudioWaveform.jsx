import React, { useEffect, useRef } from 'react';

/**
 * Audio Waveform Visualizer.
 *
 * DATA RULE: renders the real microphone signal ONLY when a live Web Audio
 * AnalyserNode is attached and capture is active. In every other state it
 * shows a static flat baseline clearly labeled "MICROPHONE IDLE" — never a
 * synthetic animation that could be mistaken for real audio.
 *
 * Colors follow the EchoShield palette: the live signal is drawn in the
 * technical accent blue (#3B82F6); result verdicts tint it green (human),
 * red (synthetic), or amber (mixed).
 */
export default function AudioWaveform({
  analyser = null,
  isActive = false,
  colorScheme = 'blue', // colors the LIVE signal (blue | human | ai | warning)
  height = 130
}) {
  const canvasRef = useRef(null);
  const hasLiveSignal = Boolean(analyser && isActive);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId = null;

    const setup = () => {
      canvas.width = (canvas.offsetWidth || 600) * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
    };
    setup();

    const colors = {
      blue: { primary: '#3B82F6', secondary: '#1D4ED8', top: '#93C5FD' },
      human: { primary: '#22C55E', secondary: '#14532D', top: '#86EFAC' },
      ai: { primary: '#EF4444', secondary: '#7F1D1D', top: '#FCA5A5' },
      warning: { primary: '#F59E0B', secondary: '#78350F', top: '#FDE68A' }
    };
    const palette = colors[colorScheme] || colors.blue;

    const drawFrame = (dataArray) => {
      const width = canvas.width;
      const h = canvas.height;
      const dpr = window.devicePixelRatio || 1;
      ctx.clearRect(0, 0, width, h);
      const centerY = h / 2;

      // Mirrored frequency bars (live data only)
      const numBars = 48;
      const barWidth = (width / numBars) * 0.65;
      const gap = (width / numBars) * 0.35;

      for (let i = 0; i < numBars; i++) {
        const dataIdx = Math.floor((i / numBars) * dataArray.length);
        const val = dataArray[dataIdx] / 255;
        const barHeight = Math.max(val * (h * 0.8), 4 * dpr);

        const x = i * (barWidth + gap);
        const yTop = centerY - barHeight / 2;

        const grad = ctx.createLinearGradient(0, yTop, 0, yTop + barHeight);
        grad.addColorStop(0, palette.top);
        grad.addColorStop(0.5, palette.primary);
        grad.addColorStop(1, palette.secondary);

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, yTop, barWidth, barHeight, [3]);
        ctx.fill();
      }

      // Central oscilloscope line
      ctx.beginPath();
      ctx.lineWidth = 2.5 * dpr;
      ctx.strokeStyle = palette.primary;
      if (hasLiveSignal) {
        // Real time-domain waveform from the microphone
        const timeData = new Uint8Array(analyser.fftSize);
        analyser.getByteTimeDomainData(timeData);
        for (let x = 0; x < width; x += 4) {
          const idx = Math.floor((x / width) * timeData.length);
          const v = (timeData[idx] - 128) / 128; // -1..1
          const y = centerY + v * (h * 0.35);
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      } else {
        // Static flat baseline: honest "no signal" state
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
      }
      ctx.stroke();
    };

    if (hasLiveSignal) {
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const render = () => {
        analyser.getByteFrequencyData(dataArray);
        drawFrame(dataArray);
        animationFrameId = requestAnimationFrame(render);
      };
      render();
    } else {
      // Single static frame — no animation loop, nothing synthetic.
      drawFrame(new Uint8Array(64));
    }

    const handleResize = () => {
      setup();
      if (!hasLiveSignal) drawFrame(new Uint8Array(64));
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [analyser, isActive, colorScheme, height, hasLiveSignal]);

  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-blush/40 border border-blush p-3">
      {/* Truthful state header */}
      <div className="flex items-center justify-between text-[10px] font-mono text-shield-muted mb-2 px-1">
        <div className="flex items-center space-x-2">
          <span
            className={`w-2 h-2 rounded-full ${
              hasLiveSignal ? 'bg-shield-human animate-ping' : 'bg-slate-400'
            }`}
          />
          <span>{hasLiveSignal ? 'LIVE MICROPHONE SIGNAL' : 'MICROPHONE IDLE — NO LIVE AUDIO'}</span>
        </div>
        <span className="text-shield-muted/80">
          {hasLiveSignal ? 'Real mic input' : 'No signal'}
        </span>
      </div>

      <canvas
        ref={canvasRef}
        style={{ height }}
        className="w-full block"
      />
    </div>
  );
}
