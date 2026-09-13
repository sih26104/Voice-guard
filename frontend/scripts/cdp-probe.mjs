/**
 * Diagnostic CDP probe — logs every stage so a hang is locatable.
 * Usage: node scripts/cdp-probe.mjs <vitePort>
 */
import { spawn } from 'node:child_process';
import http from 'node:http';

const VITE_PORT = process.argv[2] || '5175';
const WIN_SIZE = process.argv[3] || '1366,768';
const CDP_PORT = 9225 + parseInt(process.argv[4] || '0', 10);
const log = (...a) => console.log('[probe]', ...a);

const get = (url) =>
  new Promise((res, rej) =>
    http.get(url, (r) => {
      let d = '';
      r.on('data', (c) => (d += c));
      r.on('end', () => res(JSON.parse(d)));
    }).on('error', rej)
  );

// Hard watchdog so this can never hang a CI/agent session
const killer = setTimeout(() => {
  console.error('[probe] GLOBAL WATCHDOG: aborting after 75s');
  process.exit(2);
}, 75000);

log('spawning chrome...');
const chrome = spawn(
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  [
    '--headless=new',
    '--remote-debugging-port=' + CDP_PORT,
    '--remote-allow-origins=*',
    '--user-data-dir=' + process.env.TEMP + '/es-probe-profile-' + Date.now(),
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--window-size=' + WIN_SIZE,
    'about:blank',
  ],
  { stdio: 'ignore' }
);
chrome.on('error', (e) => console.error('[probe] chrome spawn error:', e.message));
chrome.on('exit', (c) => log('chrome exited with', c));

log('waiting for CDP HTTP endpoint...');
let version;
for (let i = 0; ; i++) {
  try {
    version = await get(`http://127.0.0.1:${CDP_PORT}/json/version`);
    break;
  } catch (e) {
    if (i > 40) {
      console.error('[probe] CDP endpoint never came up');
      process.exit(3);
    }
    await new Promise((r) => setTimeout(r, 250));
  }
}
log('CDP up:', version.Browser);

const targets = await get(`http://127.0.0.1:${CDP_PORT}/json/list`);
log('targets:', targets.map((t) => `${t.type}:${t.url.slice(0, 40)}`).join(' | '));
const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
if (!page) {
  console.error('[probe] no page target');
  process.exit(4);
}

log('opening websocket...');
const ws = new globalThis.WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  const to = setTimeout(() => rej(new Error('WS open timeout')), 10000);
  ws.addEventListener('open', () => { clearTimeout(to); res(); }, { once: true });
  ws.addEventListener('error', (e) => { clearTimeout(to); rej(new Error('WS error: ' + (e.message || 'unknown'))); }, { once: true });
});
log('websocket open');

let id = 0;
const pending = new Map();
ws.addEventListener('message', (ev) => {
  try {
    const msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString());
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  } catch { /* ignore */ }
});
const call = (method, params = {}) =>
  new Promise((res) => {
    const msgId = ++id;
    pending.set(msgId, res);
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });

const evaluate = async (expression) => {
  const r = await call('Runtime.evaluate', { expression, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result?.result?.value;
};

await call('Runtime.enable');
log('runtime enabled');

// Sanity: evaluate on about:blank
log('title(about:blank) =', await evaluate('document.title'));

log('navigating to home...');
await call('Page.enable');
await call('Page.navigate', { url: `http://localhost:${VITE_PORT}/` });
await new Promise((r) => setTimeout(r, 3000));
log('navigated; title =', await evaluate('document.title'));

log('viewport ' + WIN_SIZE);
const home = await evaluate(`(() => {
  const q = (s) => document.querySelector(s);
  return {
    hasNavbar: !!q('.ref-navbar'),
    brand: q('.ref-brand')?.textContent.trim() || null,
    navItems: [...document.querySelectorAll('.ref-nav-links a')].map((a) => a.textContent.trim()),
    h1: q('.ref-hero-copy h1')?.textContent || null,
    h2: q('.ref-hero-copy h2')?.textContent || null,
    cards: document.querySelectorAll('.ref-action-card').length,
    robot: (() => { const i = q('.ref-hero-art img'); return i ? i.naturalWidth + 'x' + i.naturalHeight : null; })(),
    features: document.querySelectorAll('.ref-feature').length,
    footerInPage: !!q('.ref-page > .ref-footer'),
    docScrollH: document.documentElement.scrollHeight,
    viewportH: innerHeight,
    overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    backendText: document.body.textContent.includes('127.0.0.1') || document.body.textContent.includes('Wav2Vec2'),
    navbarH: q('.ref-navbar')?.offsetHeight ?? null,
    h1Font: q('.ref-hero-copy h1') ? getComputedStyle(q('.ref-hero-copy h1')).fontSize : null,
    navbarHRect: q('.ref-navbar')?.getBoundingClientRect().height ?? null,
    heroH: q('.ref-hero')?.getBoundingClientRect().height ?? null,
    stripH: q('.ref-feature-strip')?.getBoundingClientRect().height ?? null,
    stripMB: getComputedStyle(q('.ref-feature-strip')).marginBottom,
    footerH: q('.ref-footer')?.getBoundingClientRect().height ?? null,
    copyH: q('.ref-hero-copy')?.getBoundingClientRect().height ?? null,
    gridH: q('.ref-action-grid')?.getBoundingClientRect().height ?? null,
    geometry: ['ref-navbar', 'ref-hero', 'ref-feature-strip', 'ref-footer'].map((s) => {
      const e = q('.' + s);
      const r = e && e.getBoundingClientRect();
      return r ? s + ': top=' + r.top.toFixed(0) + ' bottom=' + r.bottom.toFixed(0) + ' h=' + r.height.toFixed(0) : s + ': missing';
    }),
    pageH: q('.ref-page')?.getBoundingClientRect().height ?? null,
    robotLeft: q('.ref-hero-art img') ? Math.round(q('.ref-hero-art img').getBoundingClientRect().left) : null,
    gridRight: q('.ref-action-grid') ? Math.round(q('.ref-action-grid').getBoundingClientRect().right) : null,
    robotRight: q('.ref-hero-art img') ? Math.round(q('.ref-hero-art img').getBoundingClientRect().right) : null,
    footerBottom: q('.ref-footer') ? Math.round(q('.ref-footer').getBoundingClientRect().bottom) : null,
    bodyScrollH: document.body.scrollHeight,
    deepest: (() => {
      let best = null, maxB = 0;
      document.querySelectorAll('body *').forEach((el) => {
        const b = el.getBoundingClientRect().bottom;
        if (b > maxB && el.offsetParent !== null) { maxB = b; best = el; }
      });
      if (!best) return null;
      const chain = [];
      let cur = best;
      while (cur && cur !== document.body) {
        chain.push(cur.tagName + (cur.className ? '.' + String(cur.className).split(' ').slice(0, 3).join('.') : '') + '#' + (cur.id || '') + ' b=' + cur.getBoundingClientRect().bottom.toFixed(0));
        cur = cur.parentElement;
      }
      return chain.join(' <- ');
    })(),
  };
})()`);
log('HOME:', JSON.stringify(home, null, 1));

log('navigating to dashboard...');
await call('Page.navigate', { url: `http://localhost:${VITE_PORT}/dashboard` });
await new Promise((r) => setTimeout(r, 3000));

const dash = await evaluate(`(() => {
  const bodyText = document.body.textContent;
  const byText = (sel, txt) => [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(txt));
  const controls = byText('span', 'Microphone Monitoring');
  const voice = byText('h2', 'Current Voice Analysis');
  const dashLink = [...document.querySelectorAll('.ref-nav-links a')].find((a) => a.textContent.trim() === 'Dashboard');
  return {
    navbarOnDash: !!document.querySelector('.ref-navbar'),
    navBrand: document.querySelector('.ref-brand')?.textContent.trim() || null,
    navItems: [...document.querySelectorAll('.ref-nav-links a')].map((a) => a.textContent.trim()),
    dashActive: !!dashLink?.classList.contains('active'),
    dashAriaCurrent: dashLink?.getAttribute('aria-current') || null,
    trustBadge: !!document.querySelector('.ref-trust-badge'),
    header: document.querySelector('h1')?.textContent.replace(/\\s+/g, ' ').trim() || null,
    tabs: [...document.querySelectorAll('[role="tab"]')].map((b) => b.textContent.trim()),
    noTechLine: !bodyText.includes('Real-time voice monitoring and WAV file analysis'),
    noBackend: !bodyText.includes('127.0.0.1') && !bodyText.includes('POST /api/analyze') && !bodyText.includes('Wav2Vec2') && !bodyText.includes('CPU inference') && !bodyText.includes('localhost'),
    startStopNew: ['Start Monitoring', 'Stop Monitoring', 'New Session'].every((t) => bodyText.includes(t)),
    noUploadInMic: !bodyText.includes('Upload a WAV File'),
    voiceBelowControls: controls && voice ? voice.getBoundingClientRect().top > controls.getBoundingClientRect().top : null,
    historyPresent: bodyText.includes('Live Risk History'),
    overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
  };
})()`);
log('DASH:', JSON.stringify(dash, null, 1));

log('checking nav route Home / How It Works from dashboard...');
await evaluate(`[...document.querySelectorAll('.ref-nav-links a')].find((a) => a.textContent.trim() === 'How It Works').click()`);
await new Promise((r) => setTimeout(r, 1200));
const hiw = await evaluate(`({ path: location.pathname, h1: document.querySelector('h1')?.textContent.trim().slice(0, 60) })`);
await evaluate(`[...document.querySelectorAll('.ref-nav-links a')].find((a) => a.textContent.trim() === 'Home').click()`);
await new Promise((r) => setTimeout(r, 1200));
const home2 = await evaluate(`({ path: location.pathname, hero: !!document.querySelector('.ref-hero') })`);
await call('Page.navigate', { url: `http://localhost:${VITE_PORT}/dashboard` });
await new Promise((r) => setTimeout(r, 2500));
log('NAV:', JSON.stringify({ howItWorks: hiw, home: home2 }));

log('switching to upload tab...');
await evaluate(`[...document.querySelectorAll('[role="tab"]')].find((b) => b.textContent.includes('Upload')).click()`);
await new Promise((r) => setTimeout(r, 800));
const up = await evaluate(`(() => {
  const t = document.body.textContent;
  return {
    uploadUI: t.includes('Upload a WAV File'),
    micGone: !t.includes('Stop Monitoring') && !t.includes('New Session'),
    dragDrop: t.includes('drag & drop'),
  };
})()`);
log('UPLOAD:', JSON.stringify(up));

ws.close();
chrome.kill();
clearTimeout(killer);
log('DONE');
process.exit(0);
