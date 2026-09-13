/**
 * One-off CDP layout verification for the rebuilt EchoShield UI.
 * Usage: node scripts/verify-ui.mjs <port>   (default 5175)
 */
const PORT = process.argv[2] || '5175';
const CDP_PORT = 9223;

import { spawn } from 'node:child_process';
import http from 'node:http';

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http
      .get(url, (res) => {
        let data = '';
        res.on('data', (c) => (data += c));
        res.on('end', () => resolve(JSON.parse(data)));
      })
      .on('error', reject);
  });
}

function send(ws, id, method, params = {}) {
  return new Promise((resolve) => {
    const onMessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(typeof event.data === 'string' ? event.data : event);
      } catch {
        return; // ignore non-JSON frames (pings etc.)
      }
      if (msg.id === id) {
        ws.removeEventListener('message', onMessage);
        resolve(msg.result);
      }
    };
    ws.addEventListener('message', onMessage);
    ws.send(JSON.stringify({ id, method, params }));
  });
}

async function waitPort(port, tries = 40) {
  for (let i = 0; i < tries; i++) {
    try {
      const t = await fetchJson(`http://127.0.0.1:${port}/json/list`);
      if (Array.isArray(t)) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('CDP not reachable');
}

// Launch Chrome with remote debugging
const chrome = spawn(
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  [
    '--headless=new',
    '--remote-debugging-port=' + CDP_PORT,
    '--remote-allow-origins=*',
    '--user-data-dir=' + process.env.TEMP + '/es-cdp-profile',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1366,768',
    'about:blank',
  ],
  { stdio: 'ignore' }
);

try {
  await waitPort(CDP_PORT);
  const targets = await fetchJson(`http://127.0.0.1:${CDP_PORT}/json/list`);
  const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  if (!page) throw new Error('No debuggable page target');
  // Node >= 22 global WebSocket (proven to work with CDP on this machine)
  const ws = new globalThis.WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, rej) => {
    const to = setTimeout(() => rej(new Error('WS open timeout')), 10000);
    ws.addEventListener('open', () => { clearTimeout(to); r(); });
    ws.addEventListener('error', () => { clearTimeout(to); rej(new Error('WS error')); });
  });

  const evaluate = async (expression) => {
    const res = await send(ws, Math.random(), 'Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (res.exceptionDetails) throw new Error(JSON.stringify(res.exceptionDetails));
    return res.result.value;
  };

  await send(ws, Math.random(), 'Page.enable');
  await send(ws, Math.random(), 'Runtime.enable');

  const checks = [];
  const check = (name, ok, detail = '') =>
    checks.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);

  // ---------- HOME ----------
  await send(ws, Math.random(), 'Page.navigate', { url: `http://localhost:${PORT}/` });
  await new Promise((r) => setTimeout(r, 2500));

  let v = await evaluate(`(() => {
    const d = document;
    const q = (s) => d.querySelector(s);
    const img = q('.ref-hero-art img');
    return {
      title: d.title,
      hasNavbar: !!q('.ref-navbar'),
      brandText: q('.ref-brand')?.textContent.trim() || null,
      navItems: [...d.querySelectorAll('.ref-nav-links a')].map((a) => a.textContent.trim()),
      hasHeroH1: q('.ref-hero-copy h1')?.textContent || null,
      h2Text: q('.ref-hero-copy h2')?.textContent || null,
      actionCards: d.querySelectorAll('.ref-action-card').length,
      robotSrc: img ? img.src.split('/').pop() : null,
      robotNatural: img ? img.naturalWidth + 'x' + img.naturalHeight : null,
      featureStrip: d.querySelectorAll('.ref-feature').length,
      hasFooter: !!q('.ref-footer'),
      footerInPage: !!q('.ref-page > .ref-footer'),
      pageHeight: d.querySelector('.ref-page')?.scrollHeight,
      viewportH: innerHeight,
      viewportW: innerWidth,
      docScrollH: d.documentElement.scrollHeight,
      bodyOverflowX: d.documentElement.scrollWidth > d.documentElement.clientWidth,
      hasBackendText: d.body.textContent.includes('127.0.0.1') || d.body.textContent.includes('Wav2Vec2'),
      font: getComputedStyle(d.body).fontFamily.split(',')[0],
      waveEls: d.querySelectorAll('.ref-wave').length,
      h1FontSize: q('.ref-hero-copy h1') ? getComputedStyle(q('.ref-hero-copy h1')).fontSize : null,
      navbarH: q('.ref-navbar') ? q('.ref-navbar').offsetHeight : null,
    };
  })()`);

  const singleView = v.docScrollH <= v.viewportH + 2;
  check('home: title is EchoShield', v.title === 'EchoShield', v.title);
  check('home: navbar present', v.hasNavbar);
  check('home: nav = Home/Dashboard/How It Works', JSON.stringify(v.navItems) === JSON.stringify(['Home', 'Dashboard', 'How It Works']), v.navItems.join(','));
  check('home: hero h1 = EchoShield', v.hasHeroH1 === 'EchoShield', v.hasHeroH1);
  check('home: tagline correct', v.h2Text === 'Where Authenticity is Verified, not Assumed', v.h2Text);
  check('home: 2 action cards', v.actionCards === 2, String(v.actionCards));
  check('home: robot image loaded (reference asset)', v.robotNatural === '722x560', `${v.robotSrc} ${v.robotNatural}`);
  check('home: feature strip 4 features', v.featureStrip === 4, String(v.featureStrip));
  check('home: reference footer inside .ref-page', v.hasFooter && v.footerInPage);
  check(
    'home: single-view fit @1366x768',
    singleView,
    `docScrollH=${v.docScrollH} viewportH=${v.viewportH}`
  );
  check('home: no horizontal overflow', !v.bodyOverflowX);
  check('home: no backend/API/technical text', !v.hasBackendText);
  check('home: Inter font', /Inter/i.test(v.font), v.font);
  check('home: navbar height 85px', v.navbarH === 85, String(v.navbarH));

  // ---------- DASHBOARD ----------
  await send(ws, Math.random(), 'Page.navigate', { url: `http://localhost:${PORT}/dashboard` });
  await new Promise((r) => setTimeout(r, 2500));

  v = await evaluate(`(() => {
    const d = document;
    const q = (s) => d.querySelector(s);
    const bodyText = d.body.textContent;
    return {
      header: q('h1')?.textContent.replace(/\\s+/g, ' ').trim() || null,
      noTechLine: !bodyText.includes('Real-time voice monitoring and WAV file analysis'),
      noBackendLine: !bodyText.includes('127.0.0.1') && !bodyText.includes('POST /api/analyze') && !bodyText.includes('Wav2Vec2'),
      tabs: [...d.querySelectorAll('[role="tab"]')].map((b) => b.textContent.trim()),
      micVisible: !!q('button') && bodyText.includes('Microphone Monitoring'),
      noUploadUIInMicMode: !bodyText.includes('Upload a WAV File'),
      micSection: bodyText.includes('Microphone Monitoring'),
      voiceAnalysisBelow: (() => {
        const el = [...d.querySelectorAll('h2')].find((h) => h.textContent.includes('Current Voice Analysis'));
        const controls = [...d.querySelectorAll('span')].find((s) => s.textContent.trim() === 'Microphone Monitoring');
        if (!el || !controls) return null;
        return el.getBoundingClientRect().top > controls.getBoundingClientRect().top;
      }),
      hasStartStopNew: ['Start Monitoring', 'Stop Monitoring', 'New Session'].every((t) => bodyText.includes(t)),
      hasProbCards: bodyText.includes('Synthetic Probability') && bodyText.includes('Human Probability'),
      hasRisk: bodyText.includes('Risk Score') && bodyText.includes('Risk Level'),
      hasHistory: bodyText.includes('Live Risk History'),
      overflowX: d.documentElement.scrollWidth > d.documentElement.clientWidth,
    };
  })()`);

  check('dash: header present', v.header && v.header.includes('EchoShield Dashboard'), v.header);
  check('dash: technical description line removed', v.noTechLine);
  check('dash: no backend/API/model technical text', v.noBackendLine);
  check('dash: mode tabs = Microphone Monitoring / Upload File', JSON.stringify(v.tabs) === JSON.stringify(['Microphone Monitoring', 'Upload File']), v.tabs.join(' | '));
  check('dash: mic mode shows no upload UI', v.noUploadUIInMicMode);
  check('dash: Voice Analysis below mic controls', v.voiceAnalysisBelow === true, String(v.voiceAnalysisBelow));
  check('dash: Start/Stop/New Session present', v.hasStartStopNew);
  check('dash: probability + risk fields present', v.hasProbCards && v.hasRisk);
  check('dash: risk history present', v.hasHistory);
  check('dash: no horizontal overflow', !v.overflowX);

  // Upload mode isolation
  await evaluate(`[...document.querySelectorAll('[role="tab"]')].find((b) => b.textContent.includes('Upload')).click()`);
  await new Promise((r) => setTimeout(r, 600));
  v = await evaluate(`(() => {
    const bodyText = document.body.textContent;
    return {
      uploadUI: bodyText.includes('Upload a WAV File'),
      micControlsGone: !bodyText.includes('New Session') && !bodyText.includes('Stop Monitoring'),
      noMicStatus: !bodyText.includes('Requesting Mic Access'),
      dragDrop: bodyText.includes('drag & drop'),
    };
  })()`);
  check('dash: upload mode shows upload UI', v.uploadUI);
  check('dash: upload mode hides mic controls', v.micControlsGone && v.noMicStatus);

  // Full URL-surface sweep on both pages
  const leakSweep = await evaluate(`(() => {
    const t = document.body.textContent;
    return !(t.includes('127.0.0.1') || t.includes('localhost') || t.includes('POST /api') || t.includes('Wav2Vec2') || t.includes('CPU inference') || t.includes('multipart'));
  })()`);
  check('dash: no URL/endpoint/model leakage anywhere', leakSweep);

  ws.close();
  chrome.kill();
  console.log(checks.join('\n'));
  const failed = checks.filter((c) => c.startsWith('FAIL')).length;
  console.log(`\n${failed === 0 ? 'ALL UI CHECKS PASSED' : failed + ' CHECK(S) FAILED'}`);
  process.exitCode = failed === 0 ? 0 : 1;
} finally {
  chrome.kill();
}
