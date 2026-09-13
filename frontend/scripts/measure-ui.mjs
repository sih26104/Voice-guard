/**
 * Minimal CDP probe: measures single-view fit and section positions.
 * Usage: node scripts/measure-ui.mjs <vitePort>
 */
import { spawn } from 'node:child_process';
import http from 'node:http';

const VITE_PORT = process.argv[2] || '5175';
const CDP_PORT = 9227;

const get = (url) =>
  new Promise((res, rej) =>
    http.get(url, (r) => {
      let d = '';
      r.on('data', (c) => (d += c));
      r.on('end', () => res(JSON.parse(d)));
    }).on('error', rej)
  );

const chrome = spawn(
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  [
    '--headless=new',
    '--remote-debugging-port=' + CDP_PORT,
    '--remote-allow-origins=*',
    '--user-data-dir=' + process.env.TEMP + '/es-measure-profile',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1366,768',
    'about:blank',
  ],
  { stdio: 'ignore' }
);

try {
  // Wait for the DevTools HTTP endpoint
  for (let i = 0; ; i++) {
    try {
      await get(`http://127.0.0.1:${CDP_PORT}/json/version`);
      break;
    } catch {
      if (i > 40) throw new Error('DevTools endpoint never came up');
      await new Promise((r) => setTimeout(r, 250));
    }
  }

  const targets = await get(`http://127.0.0.1:${CDP_PORT}/json/list`);
  const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  if (!page) throw new Error('No debuggable page target');

  const ws = new WebSocket(page.webSocketDebuggerUrl); // Node >= 22 global
  await new Promise((res, rej) => {
    ws.addEventListener('open', res, { once: true });
    ws.addEventListener('error', () => rej(new Error('WS error')), { once: true });
  });

  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        pending.get(msg.id)(msg);
        pending.delete(msg.id);
      }
    } catch {
      /* ignore non-JSON frames */
    }
  });
  const call = (method, params = {}) =>
    new Promise((res) => {
      const msgId = ++id;
      pending.set(msgId, res);
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });

  const evaluate = async (expression) => {
    const r = await call('Runtime.evaluate', { expression, returnByValue: true });
    if (r.result?.exceptionDetails) {
      throw new Error(JSON.stringify(r.result.exceptionDetails));
    }
    return r.result?.result?.value;
  };

  await call('Runtime.enable');

  // ---------------- HOME @1366x768 ----------------
  await evaluate(`location.href = 'http://localhost:${VITE_PORT}/'`);
  await new Promise((r) => setTimeout(r, 2500));

  const home = await evaluate(`(() => {
    const q = (s) => document.querySelector(s);
    const rect = (s) => { const e = q(s); return e ? e.getBoundingClientRect().toJSON() : null; };
    return {
      docScrollH: document.documentElement.scrollHeight,
      viewportH: innerHeight,
      viewportW: innerWidth,
      overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      navbar: rect('.ref-navbar'),
      hero: rect('.ref-hero'),
      h1: rect('.ref-hero-copy h1'),
      cards: rect('.ref-action-grid'),
      robotImg: q('.ref-hero-art img') ? { w: q('.ref-hero-art img').naturalWidth, h: q('.ref-hero-art img').naturalHeight, cssW: q('.ref-hero-art img').getBoundingClientRect().width } : null,
      strip: rect('.ref-feature-strip'),
      footer: rect('.ref-footer'),
      footerBottom: q('.ref-footer') ? q('.ref-footer').getBoundingClientRect().bottom : null,
    };
  })()`);

  console.log('=== HOME @1366x768 ===');
  console.log(JSON.stringify(home, null, 1));
  console.log('single view (footerBottom <= 768):', home.footerBottom <= 768);

  // ---------------- HOME @1920x1080 ----------------
  await evaluate(`window.resizeTo(1920, 1080); location.href='http://localhost:${VITE_PORT}/?w=1920'`);
  await new Promise((r) => setTimeout(r, 2000));
  const home1920 = await evaluate(`(() => ({
    vw: innerWidth, vh: innerHeight,
    docScrollH: document.documentElement.scrollHeight,
    footerBottom: document.querySelector('.ref-footer')?.getBoundingClientRect().bottom ?? null,
    robotW: document.querySelector('.ref-hero-art img')?.getBoundingClientRect().width ?? null,
    h1Size: getComputedStyle(document.querySelector('.ref-hero-copy h1')).fontSize,
  }))()`);
  console.log('=== HOME @1920x1080 (if resize applied) ===');
  console.log(JSON.stringify(home1920));

  // ---------------- DASHBOARD ----------------
  await evaluate(`location.href = 'http://localhost:${VITE_PORT}/dashboard'`);
  await new Promise((r) => setTimeout(r, 2500));

  const dash = await evaluate(`(() => {
    const rect = (el) => el ? el.getBoundingClientRect().toJSON() : null;
    const byText = (sel, txt) => [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(txt));
    const controls = byText('span', 'Microphone Monitoring');
    const voice = byText('h2', 'Current Voice Analysis');
    const history = byText('h2', 'Live Risk History');
    return {
      docScrollH: document.documentElement.scrollHeight,
      overflowX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      controlsTop: controls ? controls.getBoundingClientRect().top : null,
      voiceTop: voice ? voice.getBoundingClientRect().top : null,
      historyTop: history ? history.getBoundingClientRect().top : null,
      voiceDirectlyBelowControls: controls && voice ? voice.getBoundingClientRect().top > controls.getBoundingClientRect().top : null,
      historyBelowVoice: voice && history ? history.getBoundingClientRect().top > voice.getBoundingClientRect().top : null,
    };
  })()`);

  console.log('=== DASHBOARD @1366x768 ===');
  console.log(JSON.stringify(dash, null, 1));

  ws.close();
  console.log('MEASURE DONE');
} finally {
  chrome.kill();
  setTimeout(() => process.exit(0), 300);
}
