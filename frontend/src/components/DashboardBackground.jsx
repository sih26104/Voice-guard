import React from 'react';

/**
 * DashboardBackground — subtle background depth for the dashboard.
 *
 * Extremely soft blush radial gradients, faint decorative circles, a very
 * low-opacity dotted grid, and slow ambient drift. Everything is decorative:
 * aria-hidden, pointer-events-none, and disabled under prefers-reduced-motion.
 * The page remains mostly white and fully readable.
 */
export default function DashboardBackground() {
  return (
    <div className="dash-bg-layer" aria-hidden="true">
      {/* Faint dot grid (masked so it fades out toward the edges) */}
      <div className="dash-grid" />

      {/* Soft blush/wine glow areas, slowly drifting */}
      <div className="dash-blob dash-blob-a" />
      <div className="dash-blob dash-blob-b" />
      <div className="dash-blob dash-blob-c" />

      {/* Faint decorative circles */}
      <span className="dash-circle dash-circle-a" />
      <span className="dash-circle dash-circle-b" />

      {/* Very low-opacity waveform line */}
      <svg
        className="dash-waveline"
        viewBox="0 0 1200 80"
        fill="none"
        preserveAspectRatio="none"
      >
        <path
          d="M0 40 Q 60 8 120 40 T 240 40 T 360 40 T 480 40 T 600 40 T 720 40 T 840 40 T 960 40 T 1080 40 T 1200 40"
          stroke="#3B82F6"
          strokeWidth="2"
          className="es-wave-dash"
        />
      </svg>
    </div>
  );
}
