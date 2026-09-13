import React from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  FileUp,
  Heart,
  LockKeyhole,
  Mic,
  ShieldCheck,
  UsersRound,
  Zap,
} from "lucide-react";
import robotHero from "../assets/robo.png";

const features = [
  {
    icon: ShieldCheck,
    title: "AI-Powered Detection",
    text: "Advanced models to identify synthetic and cloned voices",
  },
  {
    icon: Zap,
    title: "Real-Time Analysis",
    text: "Get instant results in seconds",
  },
  {
    icon: LockKeyhole,
    title: "Privacy Protected",
    text: "Your audio is processed temporarily. No recordings are stored.",
  },
  {
    icon: UsersRound,
    title: "A Safer Tomorrow",
    text: "Building trust in every conversation",
  },
];

/**
 * HomePage — 1:1 reproduction of the reference implementation shipped in
 * reference-ui.zip/echoshield-react.zip (src/main.jsx + src/styles.css).
 *
 * Structure, dimensions, colors, typography, spacing and the actual
 * robot-hero.png asset are transcribed from the reference. The only
 * functional changes: the two action-card buttons navigate to the existing
 * EchoShield dashboard routes (microphone mode / upload mode) instead of
 * the reference's local demo state, and the reference footer is rendered
 * INSIDE .ref-page (as in the reference, where navbar/footer are children
 * of the 100vh .page). This keeps the whole homepage within one viewport.
 */
export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="ref-page">
      <section id="home" className="ref-hero">
        <div className="ref-wave ref-wave-left" />
        <div className="ref-wave ref-wave-center" />
        <div className="ref-wave ref-wave-right" />

        <div className="ref-hero-inner">
          <div className="ref-hero-copy">
            <h1>EchoShield</h1>
            <h2>Where Authenticity is Verified, not Assumed</h2>
            <p className="ref-lead">
              Detect AI-generated and cloned voices in real-time with advanced
              AI analysis. Because every voice deserves to be trusted.
            </p>

            <div className="ref-action-grid">
              <article className="ref-action-card">
                <div className="ref-action-icon mic">
                  <Mic size={50} strokeWidth={2.2} />
                </div>
                <h3>Microphone</h3>
                <p>
                  Start real-time voice analysis
                  <br />
                  using your microphone
                </p>
                <button
                  className="ref-primary-btn"
                  onClick={() => navigate("/dashboard")}
                >
                  Start Monitoring
                  <ArrowRight size={23} />
                </button>
              </article>

              <article className="ref-action-card">
                <div className="ref-action-icon upload">
                  <FileUp size={45} strokeWidth={2.1} />
                </div>
                <h3>Upload File</h3>
                <p>
                  Analyze a pre-recorded audio file
                  <br />
                  for authenticity
                </p>
                {/* Navigates straight to the upload dashboard — file
                    selection happens there, never on the home page. */}
                <button
                  className="ref-primary-btn"
                  onClick={() => navigate("/dashboard", { state: { mode: "upload" } })}
                >
                  Upload Audio
                  <ArrowRight size={23} />
                </button>
              </article>
            </div>
          </div>

          <div className="ref-hero-art" aria-hidden="true">
            <img src={robotHero} alt="" />
          </div>
        </div>
      </section>

      <section className="ref-feature-strip" aria-label="EchoShield benefits">
        {features.map(({ icon: Icon, title, text }, index) => (
          <div
            className={`ref-feature ${index !== features.length - 1 ? "ref-with-divider" : ""}`}
            key={title}
          >
            <div className="ref-feature-icon">
              <Icon size={31} strokeWidth={2.1} />
            </div>
            <div>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          </div>
        ))}
      </section>

      {/* Reference footer — INSIDE .ref-page exactly like the reference's
          .page > footer, so the page stays a single-view composition. */}
      <footer className="ref-footer">
        <div className="ref-footer-left">
          <div className="ref-footer-mic-glyph">
            <Mic size={37} strokeWidth={2.2} />
          </div>
          <div>
            <strong>EchoShield</strong>
            <span>Where Authenticity is Verified, not Assumed</span>
          </div>
        </div>

        <nav className="ref-footer-nav" aria-label="Footer navigation">
          <button onClick={() => navigate("/")}>Home</button>
          <i />
          <button onClick={() => navigate("/dashboard")}>Dashboard</button>
          <i />
          <button onClick={() => navigate("/how-it-works")}>
            How It Works
          </button>
        </nav>

        <div className="ref-footer-right">
          <div className="ref-footer-message">
            <span>A Safer, More Authentic World</span>
            <span>For Everyone</span>
          </div>
          <Heart size={28} strokeWidth={1.8} />
        </div>
      </footer>
    </div>
  );
}
