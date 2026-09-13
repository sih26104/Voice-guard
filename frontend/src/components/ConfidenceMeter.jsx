import React from 'react';
import { UserCheck, Bot } from 'lucide-react';

/**
 * ConfidenceMeter: Human <-> AI Confidence Arc Gauge.
 *
 * DATA RULE: when hasData is false, the gauge shows "--" and no needle —
 * it never implies a probability that the backend did not produce.
 *
 * @param {number|null} aiScore - 0 to 100 (0 = 100% Human, 100 = 100% AI), null when no result
 * @param {boolean} hasData - true only when a real backend result exists
 * @param {string} classification - verdict pill text (real-backend derived)
 * @param {string} riskLevel - 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'IDLE'
 */
export default function ConfidenceMeter({
  aiScore = null,
  hasData = false,
  classification = 'WAITING FOR INPUT',
  riskLevel = 'IDLE'
}) {
  const score = hasData && aiScore != null ? Math.max(0, Math.min(100, aiScore)) : null;

  // State colors (only meaningful with real data)
  const isHuman = hasData && (riskLevel === 'LOW' || (score < 40 && riskLevel !== 'IDLE'));
  const isAI = hasData && (riskLevel === 'HIGH' || riskLevel === 'CRITICAL' || score > 65);
  const isMixed = hasData && !isHuman && !isAI;

  let activeColor = '#9CA3AF'; // neutral gray until real data exists
  let activeBg = 'rgba(156, 163, 175, 0.10)';

  if (isHuman) {
    activeColor = '#22C55E';
    activeBg = 'rgba(34, 197, 94, 0.14)';
  } else if (isAI) {
    activeColor = '#EF4444';
    activeBg = 'rgba(239, 68, 68, 0.16)';
  } else if (isMixed) {
    activeColor = '#F59E0B';
    activeBg = 'rgba(245, 158, 11, 0.14)';
  }

  // Arc math: 180-degree semi-circle. 0% AI = left, 100% AI = right.
  const needleAngle = score != null ? -90 + (score / 100) * 180 : 0;

  return (
    <div className="relative flex flex-col items-center justify-center p-5 sm:p-6 rounded-3xl bg-white border border-blush shadow-card overflow-hidden">

      {/* Background radial glow */}
      <div
        className="absolute inset-0 transition-colors duration-700 pointer-events-none"
        style={{
          background: `radial-gradient(circle at 50% 60%, ${activeBg} 0%, transparent 70%)`,
        }}
      />

      {/* Top Header */}
      <div className="w-full flex items-center justify-between mb-3 z-10">
        <span className="text-xs font-mono uppercase tracking-widest text-rose-hover font-bold flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: activeColor }} />
          Confidence Arc
        </span>
        <div className="text-xs font-mono px-3 py-1 rounded-full border transition-all"
          style={{
            borderColor: activeColor,
            color: activeColor,
            backgroundColor: `${activeColor}15`,
          }}
        >
          {hasData ? `RISK: ${riskLevel}` : 'NO DATA'}
        </div>
      </div>

      {/* Arc SVG & Gauge */}
      <div className="relative w-full max-w-[320px] aspect-[2/1.35] flex items-center justify-center">
        <svg viewBox="0 0 300 175" className="w-full h-full overflow-visible">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22C55E" />
              <stop offset="35%" stopColor="#22C55E" />
              <stop offset="50%" stopColor="#F59E0B" />
              <stop offset="65%" stopColor="#EF4444" />
              <stop offset="100%" stopColor="#EF4444" />
            </linearGradient>
            <filter id="needleGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={activeColor} />
            </filter>
          </defs>

          {/* Background Track Arc */}
          <path
            d="M 35 145 A 115 115 0 0 1 265 145"
            fill="none"
            stroke="#F3D6D6"
            strokeWidth="18"
            strokeLinecap="round"
          />

          {/* Colored Gradient Track Arc (dimmed without data) */}
          <path
            d="M 35 145 A 115 115 0 0 1 265 145"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="14"
            strokeLinecap="round"
            opacity={hasData ? '0.85' : '0.25'}
          />

          {/* Tick marks */}
          {[0, 25, 50, 75, 100].map((tick) => {
            const angle = (-180 + (tick / 100) * 180) * (Math.PI / 180);
            const x1 = 150 + 95 * Math.cos(angle);
            const y1 = 145 + 95 * Math.sin(angle);
            const x2 = 150 + 105 * Math.cos(angle);
            const y2 = 145 + 105 * Math.sin(angle);
            return (
              <line
                key={tick}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="rgba(99, 47, 53, 0.35)"
                strokeWidth="2"
                strokeLinecap="round"
              />
            );
          })}

          {/* Needle: rendered only when a real result exists */}
          {hasData && (
            <g
              className="transition-transform duration-700 ease-out"
              style={{
                transform: `rotate(${needleAngle}deg)`,
                transformOrigin: '150px 145px',
              }}
            >
              <line
                x1="150"
                y1="145"
                x2="150"
                y2="42"
                stroke={activeColor}
                strokeWidth="4"
                strokeLinecap="round"
                filter="url(#needleGlow)"
              />
              <polygon
                points="150,34 145,46 155,46"
                fill={activeColor}
                filter="url(#needleGlow)"
              />
            </g>
          )}

          {/* Needle Hub */}
          <circle cx="150" cy="145" r="16" fill="#FFFFFF" stroke={activeColor} strokeWidth="3" />
          <circle cx="150" cy="145" r="7" fill={activeColor} />
        </svg>

        {/* Center Readout: "--" until real data arrives */}
        <div className="absolute bottom-1 flex flex-col items-center text-center">
          <span className={`text-4xl sm:text-5xl font-extrabold tracking-tight font-mono ${hasData ? 'text-wine' : 'text-shield-muted/70'}`}>
            {score != null ? `${score.toFixed(1)}%` : '--'}
          </span>
          <span className="text-[11px] font-mono tracking-wider uppercase text-shield-muted font-semibold">
            {hasData
              ? isHuman
                ? 'Human Authenticity'
                : isAI
                ? 'Synthetic Confidence'
                : 'Hybrid Probability'
              : 'Awaiting analysis'}
          </span>
        </div>
      </div>

      {/* Flanking Probabilities */}
      <div className="w-full flex items-center justify-between mt-4 px-2 z-10">

        {/* Human left */}
        <div
          className={`flex items-center space-x-2.5 p-2.5 rounded-2xl transition-all duration-500 ${
            isHuman
              ? 'bg-emerald-50 border border-emerald-300'
              : 'opacity-40'
          }`}
        >
          <div className={`p-2 rounded-xl ${isHuman ? 'bg-shield-human text-white' : 'bg-blush text-wine'}`}>
            <UserCheck className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-wine">HUMAN</div>
            <div className={`text-[10px] font-mono ${isHuman ? 'text-emerald-600' : 'text-shield-muted'}`}>
              {hasData ? `${(100 - score).toFixed(1)}%` : '--'}
            </div>
          </div>
        </div>

        {/* Center verdict pill */}
        <div className="text-center px-3 py-1.5 rounded-xl bg-blush/40 border border-blush max-w-[40%]">
          <div className={`text-[11px] font-bold tracking-wide uppercase font-mono ${hasData ? 'text-wine' : 'text-shield-muted/70'}`}>
            {hasData ? classification : 'WAITING FOR INPUT'}
          </div>
        </div>

        {/* AI right */}
        <div
          className={`flex items-center space-x-2.5 p-2.5 rounded-2xl transition-all duration-500 ${
            isAI
              ? 'bg-red-50 border border-red-300'
              : 'opacity-40'
          }`}
        >
          <div className="text-right">
            <div className="text-xs font-mono font-bold text-wine">SYNTHETIC</div>
            <div className={`text-[10px] font-mono ${isAI ? 'text-red-600' : 'text-shield-muted'}`}>
              {hasData ? `${score.toFixed(1)}%` : '--'}
            </div>
          </div>
          <div className={`p-2 rounded-xl ${isAI ? 'bg-red-500 text-white animate-pulse' : 'bg-blush text-wine'}`}>
            <Bot className="w-5 h-5" />
          </div>
        </div>

      </div>

    </div>
  );
}
