import React, { useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import HowItWorksPage from './pages/HowItWorksPage';

// Scroll to top upon route navigation
function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}

export default function App() {
  const { pathname } = useLocation();

  return (
    // <div className="relative min-h-screen bg-white text-wine selection:bg-blush selection:text-wine flex flex-col justify-between overflow-x-hidden">
    <div className="relative min-h-screen bg-white text-wine selection:bg-blush selection:text-wine flex flex-col justify-between overflow-x-clip">
      {/* Scroll restoration helper */}
      <ScrollToTop />

      {/* Persistent Header Navigation (reference design) */}
      <Navbar />

      {/* Routes: Home (landing) + Dashboard + How It Works.
          The home page is a single-view composition that includes its own
          reference footer (inside .ref-page), so the shared Footer is not
          rendered there. Redirects below keep old links working. */}
      <main className="relative z-10 flex-grow">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/how-it-works" element={<HowItWorksPage />} />

          {/* Redirects from removed pages */}
          <Route path="/home" element={<Navigate to="/" replace />} />
          <Route path="/technology" element={<Navigate to="/" replace />} />
          <Route
            path="/threat-intelligence"
            element={<Navigate to="/" replace />}
          />
          <Route path="/about" element={<Navigate to="/" replace />} />
          <Route path="/contact" element={<Navigate to="/" replace />} />

          {/* Catch-all fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {pathname !== "/" && <Footer />}
    </div>
  );
}
