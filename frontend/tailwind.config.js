/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // EchoShield palette
        wine: {
          DEFAULT: '#632F35', // Primary — headings, logo, footer, icons
          50: '#FBF6F6',
          100: '#F3D6D6',     // Soft Blush — section backgrounds, card accents
          200: '#E9C2C2',
          300: '#D89C9C',
          400: '#BA5A5A',     // Blushed Rose — primary buttons, CTAs
          500: '#BA5A5A',
          600: '#8F3F3F',     // Burnt Rose — hover/active states
          700: '#632F35',     // Deep Wine
          800: '#54272C',
          900: '#451F24',
          950: '#2F1518',
        },
        rose: {
          DEFAULT: '#BA5A5A', // Primary buttons / CTA
          hover: '#8F3F3F',   // Burnt Rose hover/active
          soft: '#F3D6D6',    // Soft Blush
        },
        blush: '#F3D6D6',     // Section backgrounds / card accents
        tech: {
          DEFAULT: '#3B82F6', // Electric Blue — audio/technical accents ONLY
          soft: '#DBEAFE',
        },
        shield: {
          human: '#22C55E',   // Success green
          warning: '#F59E0B', // Amber (mixed/borderline)
          ai: '#EF4444',      // Synthetic verdict
          text: '#632F35',    // Primary text
          muted: '#6B7280',   // Secondary text
          white: '#FFFFFF',
        },
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgba(99, 47, 53, 0.06), 0 8px 24px -8px rgba(99, 47, 53, 0.12)',
        'card-hover': '0 4px 8px 0 rgba(99, 47, 53, 0.08), 0 16px 40px -12px rgba(99, 47, 53, 0.20)',
        'btn': '0 4px 14px -2px rgba(186, 90, 90, 0.45)',
        'btn-hover': '0 6px 20px -2px rgba(143, 63, 63, 0.55)',
        'navbar': '0 1px 2px rgba(99, 47, 53, 0.05), 0 8px 24px -12px rgba(99, 47, 53, 0.15)',
      },
      animation: {
        'float-slow': 'float 6s ease-in-out infinite',
        'wave-bar': 'waveBar 1.6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        waveBar: {
          '0%, 100%': { transform: 'scaleY(0.3)' },
          '50%': { transform: 'scaleY(1)' },
        },
      }
    },
  },
  plugins: [],
}
