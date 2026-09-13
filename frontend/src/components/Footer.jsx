import React from 'react';
import { Link } from 'react-router-dom';
import { Mic, Heart } from 'lucide-react';

/**
 * Footer for the Dashboard and How It Works pages. (The homepage renders its
 * own footer inside .ref-page, cloned 1:1 from the reference implementation.)
 * No internal architecture details — endpoints, model names, inference
 * hardware — are displayed to end users.
 */
export default function Footer() {
  return (
    <footer className="relative mt-16 bg-wine text-white z-20">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 py-10">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-8">

          {/* Brand + tagline */}
          <div className="max-w-sm">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-xl bg-white/10 border border-white/20 flex items-center justify-center">
                <Mic className="w-5 h-5 text-blush" />
              </div>
              <span className="text-lg font-extrabold tracking-tight">EchoShield</span>
            </div>
            <p className="mt-3 text-sm text-blush/90 font-light">
              Where Authenticity is Verified, not Assumed
            </p>
          </div>

          {/* Navigation */}
          <nav aria-label="Footer navigation" className="flex flex-col space-y-2">
            <span className="text-xs font-semibold uppercase tracking-widest text-blush/70 mb-1">
              Navigate
            </span>
            {[
              { name: 'Home', path: '/' },
              { name: 'Dashboard', path: '/dashboard' },
              { name: 'How It Works', path: '/how-it-works' },
            ].map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className="text-sm text-white/85 hover:text-blush transition-colors w-fit"
              >
                {item.name}
              </Link>
            ))}
          </nav>

          {/* Honest prototype note (real constraints only, no invented claims) */}
          <div className="max-w-xs text-xs text-blush/80 leading-relaxed">
            <p>
              Student prototype — Smart India Hackathon 2026 (PS 26104). Results are model
              estimates, not guarantees; verify high-stakes audio through a second channel.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-white/15 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-blush/70">
          <span>EchoShield — voice authenticity analysis prototype</span>
          <div className="flex items-center gap-2">
            <span>A Safer, More Authentic World</span>
            <Heart className="w-3.5 h-3.5" />
          </div>
        </div>
      </div>
    </footer>
  );
}
