import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Mic, ShieldCheck } from 'lucide-react';

const navItems = [
  { name: 'Home', path: '/' },
  { name: 'Dashboard', path: '/dashboard' },
  { name: 'How It Works', path: '/how-it-works' },
];

/** Exact-match active state; the root path only matches '/' itself. */
function isActivePath(pathname, path) {
  if (path === '/') return pathname === '/';
  return pathname === path || pathname.startsWith(`${path}/`);
}

/**
 * Navbar — visual clone of the reference implementation (reference-ui.zip).
 * Router <Link>s replace the reference's scroll buttons so the SPA routes
 * still work; dimensions, spacing and states are transcribed 1:1.
 */
export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  // Close menu on route change
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  // Lock body scroll when the mobile menu is open
  useEffect(() => {
    document.body.style.overflow = isOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <header className="ref-navbar">
      <div className="ref-nav-inner">
        <Link to="/" className="ref-brand" aria-label="EchoShield home">
          <span className="ref-brand-icon">
            <Mic size={28} strokeWidth={2.4} />
          </span>
          <span>EchoShield</span>
        </Link>

        <nav className={`ref-nav-links ${isOpen ? 'open' : ''}`} aria-label="Main navigation">
          {navItems.map((item) => {
            const active = isActivePath(location.pathname, item.path);
            return (
              <Link
                key={item.path}
                to={item.path}
                aria-current={active ? 'page' : undefined}
                className={active ? 'active' : ''}
              >
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Reference trust badge (decorative) */}
        <div className="ref-trust-badge" aria-hidden="true">
          <ShieldCheck size={39} strokeWidth={1.8} />
          <div>
            <strong>Protecting</strong>
            <span>What's Real</span>
          </div>
        </div>

        <button
          className="ref-mobile-menu"
          onClick={() => setIsOpen((v) => !v)}
          aria-label="Toggle menu"
          aria-expanded={isOpen}
        >
          {isOpen ? <X /> : <Menu />}
        </button>
      </div>
    </header>
  );
}
