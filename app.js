// app.js — the party game: landing, loading, board, fly avatar, brain canvas, commentary,
// result card, leaderboard, party mode, sounds. All chess rules come from chess.js; every fly
// move comes from the network running in engine/worker.js.

import { Chess } from './vendor/chess.js';
import { Board } from './board.js';
import { repetitionCount } from './engine/encoding.js';
import { BrainCanvas, ClassStrip, sampleGroups } from './brainviz.js';
import { EyePanel, squareDrive, gazeShift, gazeLine, typeCounts, eyeFiles } from './eye.js';

const $ = (id) => document.getElementById(id);
const DIFF_LABEL = { larva: 'Larva', fly: 'Fly', superfly: 'Superfly' };
const LB_KEY = 'flychess.leaderboard.v1';
const fmtInt = (n) => Number(n).toLocaleString('en-US');
const fmtMs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const fmtClock = (ms) => { const s = Math.round(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

// ============================================================================ worker client
// A request that gets no answer (result, error or 'thinking' progress) for this long is treated as a
// dead worker: terminate()/OOM kills fire no onerror, so without it a move would hang forever.
const REQUEST_TIMEOUT_MS = 60_000;

class Brain {
  constructor() {
    this.pending = new Map();
    this.nextId = 1;
    this.ready = null;
    this.info = null;
    this.dead = false;
    this.baseUrl = null;
    this.onProgress = () => {};
    this.onThinking = () => {};
    this.onBackend = () => {};    // {backend:'webgpu'|'js', reason?} — the worker switched engines (WebGPU lost → JS)
    this._spawn();
  }

  _spawn() {
    this.worker = new Worker('./engine/worker.js', { type: 'module' });
    this.worker.onmessage = (ev) => this._onMessage(ev.data);
    this.worker.onerror = (ev) => { this.dead = true; this._fail(new Error(ev.message || 'worker crashed')); };
  }

  load(baseUrl = 'model/') {
    this.baseUrl = baseUrl;
    this.dead = false;
    this.ready = new Promise((resolve, reject) => { this._resolveReady = resolve; this._rejectReady = reject; });
    this.worker.postMessage({ type: 'load', baseUrl });
    return this.ready;
  }

  /** Fresh worker + reload. Requests are stateless (the worker replays msg.moves from the start
   *  position), so nothing else needs restoring. Any in-flight request is rejected. */
  restart() {
    try { this.worker.terminate(); } catch { /* already gone */ }
    this._fail(new Error('restarting'));
    this._spawn();
    return this.load(this.baseUrl || 'model/');
  }

  _fail(err) {
    this._rejectReady?.(err);
    for (const [, p] of this.pending) { clearTimeout(p.timer); p.reject(err); }
    this.pending.clear();
  }

  _settle(id, fn) {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id); clearTimeout(p.timer); fn(p);
  }

  _onMessage(msg) {
    if (msg.type === 'progress') { this.onProgress(msg); return; }
    if (msg.type === 'ready') { this.info = msg; this._resolveReady?.(msg); return; }
    if (msg.type === 'backend') { if (this.info) this.info.backend = msg.backend; this.onBackend(msg); return; }
    if (msg.type === 'thinking') {
      const p = this.pending.get(msg.id);
      if (!p || p.cancelled) return;      // progress of an abandoned (or unknown) search must not leak into the UI
      p.touch(); this.onThinking(msg); return;
    }
    if (msg.type === 'error') {
      if (msg.id && this.pending.has(msg.id)) this._settle(msg.id, (p) => p.reject(new Error(msg.message)));
      else this._fail(new Error(msg.message));
      return;
    }
    this._settle(msg.id, (p) => p.resolve(msg));
  }

  _request(payload) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const p = { resolve, reject, timer: 0, type: payload.type, cancelled: false };
      p.touch = () => {
        clearTimeout(p.timer);
        p.timer = setTimeout(() => {
          this.dead = true;   // the next restart() replaces the silent worker
          this._settle(id, () => reject(new Error(`no answer from the fly brain after ${REQUEST_TIMEOUT_MS / 1000} s`)));
        }, REQUEST_TIMEOUT_MS);
      };
      this.pending.set(id, p);
      p.touch();
      this.worker.postMessage({ ...payload, id });
    });
  }

  move(fen, moves, difficulty) { return this._request({ type: 'move', fen, moves, difficulty, trace: true }); }
  eval(fen, moves) { return this._request({ type: 'eval', fen, moves }); }

  /** Abandon every in-flight move request: the worker stops a running superfly search within a few
   *  simulations (and skips queued ones) instead of finishing it before the next game's first move.
   *  The promise still settles (with `cancelled: true`) once the worker acknowledges. */
  cancelMoves() {
    for (const [id, p] of this.pending) {
      if (p.type !== 'move' || p.cancelled) continue;
      p.cancelled = true;
      this.worker.postMessage({ type: 'cancel', id });
    }
  }
}

// ============================================================================ sounds (WebAudio, synthesised)
class Sounds {
  constructor() { this.enabled = false; this.ctx = null; this.buzz = null; }
  toggle() { this.enabled = !this.enabled; if (!this.enabled) this.stopBuzz(); return this.enabled; }
  _ctx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }
  click(kind = 'move') {
    if (!this.enabled) return;
    const ctx = this._ctx(), t = ctx.currentTime;
    const len = kind === 'capture' ? 0.09 : 0.045;
    const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * len), ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3);
    const src = ctx.createBufferSource(); src.buffer = buf;
    const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = kind === 'capture' ? 500 : 1800; f.Q.value = 1.2;
    const g = ctx.createGain(); g.gain.value = kind === 'capture' ? 0.5 : 0.35;
    src.connect(f).connect(g).connect(ctx.destination);
    src.start(t);
    if (kind === 'capture') { // a little wooden thud underneath
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.setValueAtTime(160, t); o.frequency.exponentialRampToValueAtTime(70, t + 0.09);
      const og = ctx.createGain(); og.gain.setValueAtTime(0.25, t); og.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
      o.connect(og).connect(ctx.destination); o.start(t); o.stop(t + 0.13);
    }
  }
  chord(win) {
    if (!this.enabled) return;
    const ctx = this._ctx(), t = ctx.currentTime;
    const notes = win ? [440, 554, 659] : [330, 311, 262];
    notes.forEach((fq, i) => {
      const o = ctx.createOscillator(); o.type = 'triangle'; o.frequency.value = fq;
      const g = ctx.createGain(); g.gain.setValueAtTime(0.0001, t + i * 0.12); g.gain.exponentialRampToValueAtTime(0.12, t + i * 0.12 + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.12 + 0.6);
      o.connect(g).connect(ctx.destination); o.start(t + i * 0.12); o.stop(t + i * 0.12 + 0.65);
    });
  }
  startBuzz() {
    if (!this.enabled || this.buzz) return;
    const ctx = this._ctx(), t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = 175;
    const o2 = ctx.createOscillator(); o2.type = 'square'; o2.frequency.value = 176.5;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 24;
    const lg = ctx.createGain(); lg.gain.value = 0.012;
    const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 700;
    const g = ctx.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.03, t + 0.25);
    lfo.connect(lg).connect(g.gain);
    o.connect(f); o2.connect(f); f.connect(g).connect(ctx.destination);
    o.start(); o2.start(); lfo.start();
    this.buzz = { o, o2, lfo, g };
  }
  stopBuzz() {
    const b = this.buzz; if (!b) return; this.buzz = null;
    const t = this.ctx.currentTime;
    b.g.gain.cancelScheduledValues(t); b.g.gain.setValueAtTime(b.g.gain.value, t); b.g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
    setTimeout(() => { b.o.stop(); b.o2.stop(); b.lfo.stop(); }, 260);
  }
}

// ============================================================================ commentary & mood
const MOUTHS = {
  thinking: 'M95 73 H105', confident: 'M94 72 Q100 78 106 72', smug: 'M93 73 Q100 77 108 69',
  nervous: 'M94 75 Q100 70 106 75', panicking: 'M95 71 Q100 82 105 71 Z', curious: 'M94 72 Q100 75 106 72',
};
const MOOD_WORDS = {
  thinking: 'thinking', confident: 'confident', smug: 'smug', nervous: 'nervous', panicking: 'panicking', curious: 'curious',
};
function moodFor(value) {
  if (value > 0.55) return 'smug';
  if (value > 0.18) return 'confident';
  if (value < -0.55) return 'panicking';
  if (value < -0.18) return 'nervous';
  return 'curious';
}
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
function commentaryFor({ policyTop, value, san, difficulty, sims }) {
  const top = policyTop?.[0];
  const second = policyTop?.[1];
  const pct = top ? Math.round(top.p * 100) : 0;
  const lines = [];
  if (top && san && top.san === san) {
    if (difficulty === 'superfly' && sims) lines.push(pick([`${sims} simulations later, the fly still wants ${san} (${pct}% on instinct).`, `Instinct and search agree: ${san}.`, `The tree search confirmed the fly's gut: ${san}.`]));
    else if (pct >= 70) lines.push(pick([`The fly is ${pct}% sure about ${san}.`, `${san}, obviously. ${pct}% of the fly agrees.`, `${pct}% of the fly's descending neurons wanted ${san}.`]));
    else if (pct >= 40) lines.push(pick([`${san} — the fly's favourite at ${pct}%.`, `The fly leans ${san} (${pct}%).`]));
    else lines.push(pick([`The fly wavered between ${top.san} and ${second?.san || '…'}, then played ${san}.`, `Only ${pct}% sure, the fly plays ${san} anyway.`]));
  } else if (san && top) {
    if (difficulty === 'superfly' && sims) lines.push(pick([`Instinct said ${top.san}; after ${sims} simulations the fly prefers ${san}.`, `The tree search talked the fly out of ${top.san}. ${san} it is.`]));
    else if (difficulty === 'larva') lines.push(pick([`Larva mode: the fly rolled the dice and got ${san}.`, `${san}! The fly wasn't sure either (${pct}% wanted ${top.san}).`]));
    else lines.push(pick([`The fly liked ${top.san} but its value head vetoed it. ${san} instead.`, `One ply of doubt turned ${top.san} into ${san}.`]));
  } else if (top) {
    lines.push(pct >= 60 ? `The fly already likes ${top.san} (${pct}%).` : `The fly is torn between ${top.san} and ${second?.san || '…'}.`);
  }
  if (value > 0.55) lines.push(pick(['It smells victory.', 'Rubbing its front legs together.', 'It thinks it is winning. It might be.']));
  else if (value > 0.18) lines.push(pick(['It feels good about this.', 'Wings relaxed.', 'Quietly confident.']));
  else if (value < -0.55) lines.push(pick(['The fly senses danger.', 'Panic in the mushroom bodies.', 'Every descending neuron is screaming.']));
  else if (value < -0.18) lines.push(pick(['The fly is nervous.', 'Antennae twitching.', 'It does not love this position.']));
  return lines.join(' ');
}

// ============================================================================ leaderboard
function loadBoard() { try { return JSON.parse(localStorage.getItem(LB_KEY) || '[]'); } catch { return []; } }
function saveBoard(rows) { try { localStorage.setItem(LB_KEY, JSON.stringify(rows.slice(0, 200))); } catch { /* ignore */ } }
function rankRows(rows) {
  const order = { win: 0, draw: 1, loss: 2 };
  return [...rows].sort((a, b) => (order[a.result] - order[b.result]) || (a.moves - b.moves) || (a.timeMs - b.timeMs));
}

// ============================================================================ app
class App {
  constructor() {
    this.brain = new Brain();
    this.sounds = new Sounds();
    this.chess = new Chess();
    this.board = null;
    this.viz = new BrainCanvas($('brain-canvas'));
    this.strip = new ClassStrip($('strip-canvas'));
    this.eye = new EyePanel($('eye-panel'));
    this.groups = [];          // strip-chart rows: sampled neurons per super class (+ retina)
    this.flySvg = '';
    this.state = null;
    this.party = null;
    this.pendingMoveId = 0;
    this._bindUI();
    this._loadFly();
    this._startLoading();
  }

  // ---------------------------------------------------------------- boot
  async _loadFly() {
    try {
      this.flySvg = await (await fetch('assets/fly.svg')).text();
      for (const id of ['hero-fly', 'avatar']) $(id).innerHTML = this.flySvg;
      $('brand').querySelector('.brand-fly').innerHTML = this.flySvg;
    } catch { /* decorative */ }
  }

  /** "engine: WebGPU" / "engine: JS" in the specimen line (the worker reports which backend answered). */
  _showBackend(backend) {
    const el = $('spec-engine');
    if (!el || !backend) return;
    el.textContent = backend === 'webgpu' ? 'WebGPU' : 'JS';
    el.title = backend === 'webgpu' ? 'forward pass on the GPU (WebGPU compute shaders)' : 'forward pass in plain JavaScript';
  }

  _startLoading({ retry = false } = {}) {
    const fill = $('bar-fill'), bytes = $('load-bytes'), neurons = $('load-neurons').querySelector('b');
    let n = 0, shown = 0, tick = 0;
    if (retry) {   // back to the initial "growing" state before the bar moves again
      $('load-error').hidden = true; $('loading').hidden = false;
      this._loadFrac = 0; fill.style.width = '0%'; bytes.textContent = 'connecting…'; neurons.textContent = '0';
      $('btn-start').querySelector('.btn-start-label').textContent = 'Growing the fly brain…';
    }
    const animateCount = () => {
      const goal = n * Math.min(1, this._loadFrac || 0);
      if (Math.abs(goal - shown) > 1) { shown += (goal - shown) * 0.2; neurons.textContent = fmtInt(Math.round(shown)); tick = requestAnimationFrame(animateCount); } else tick = 0;
    };
    this.brain.onProgress = (m) => {
      if (m.n) n = m.n;
      if (m.phase === 'header') { $('spec-name').textContent = m.runName || 'brain'; bytes.textContent = 'fetching synapses…'; return; }
      const frac = m.total ? m.loaded / m.total : (m.phase === 'decode' ? 1 : 0);
      this._loadFrac = frac;
      fill.style.width = `${Math.round(frac * 100)}%`;
      bytes.textContent = m.phase === 'decode' ? 'wiring synapses…' : m.total ? `${(m.loaded / 1e6).toFixed(1)} MB / ${(m.total / 1e6).toFixed(1)} MB` : `${(m.loaded / 1e6).toFixed(1)} MB`;
      if (!tick) tick = requestAnimationFrame(animateCount);
    };
    // where the brain lives: <meta name="fly-model-base"> (a relative path such as 'model/' or an absolute
    // URL, e.g. a Hugging Face 'resolve' folder); the deploy script sets it for the published site
    const base = document.querySelector('meta[name="fly-model-base"]')?.content?.trim() || 'model/';
    const url = new URL(base.endsWith('/') ? base : base + '/', location.href).href;
    // a retry always gets a fresh worker: the old one may have died, or be stuck mid-decode
    (retry ? this.brain.restart() : this.brain.load(url)).then((info) => {
      const h = info.header;
      this._loadFrac = 1; fill.style.width = '100%';
      neurons.textContent = fmtInt(h.n);
      bytes.textContent = info.fromCache ? 'from cache' : `${(info.bytes / 1e6).toFixed(1)} MB decoded`;
      $('spec-name').textContent = h.run_name || 'brain';
      $('spec-neurons').textContent = fmtInt(h.n); $('spec-synapses').textContent = fmtInt(h.nnz);
      this._showBackend(info.backend);
      this.brain.onBackend = (m) => this._showBackend(m.backend);
      const syn = h.total_synapses ? `${(h.total_synapses / 1e6).toFixed(1)} million synapses in` : '';
      $('lede-connections-note').textContent = syn;
      $('lede-neurons').textContent = fmtInt(h.n); $('lede-synapses').textContent = `${(h.nnz / 1e6).toFixed(2)} million`;
      this.viz.setData(info.sample, info.silhouette, info.legend);
      this._renderLegend(info);
      this.groups = sampleGroups(info.sample.idx, info.sample.cls, info.legend, info.retina?.idx);
      this.eye.setRetina(info.retina);
      this._describeEyes(info);
      $('brain-caption').textContent = `${fmtInt(info.sample.idx.length)} of ${fmtInt(h.n)} neurons · real connectome positions`;
      $('btn-start').disabled = false;
      $('btn-start').querySelector('.btn-start-label').textContent = 'Play the fly';
      $('loading').hidden = true;
    }).catch((err) => {
      if (tick) { cancelAnimationFrame(tick); tick = 0; }
      $('loading').hidden = true;
      const el = $('load-error');
      el.hidden = false;
      const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
      const hint = local
        ? ` The model files live in <code>web/model/</code>; export them with <code>fly export-web --run &lt;name&gt;</code>, then serve the site (<code>scripts/serve-web.sh</code>).`
        : '';
      el.innerHTML = `Could not download the fly brain — check your connection and try again.${hint} <button class="btn btn-small" id="btn-retry-load" type="button">Retry</button><br><small class="mono dim">${escapeHtml(err.message)}</small>`;
      $('btn-retry-load').addEventListener('click', () => this._startLoading({ retry: true }));
      $('btn-start').querySelector('.btn-start-label').textContent = 'No fly brain found';
    });
  }

  /** Photoreceptor counts and eye → files coverage from the header (specimen line, landing page, about). */
  _describeEyes(info) {
    const R = info.retina;
    $('spec-eyes').hidden = !R;
    $('lede-eyes').hidden = !R; $('fact-eyes').hidden = !R; $('about-eyes').hidden = !R;
    if (!R) return;
    $('spec-eyes-n').textContent = fmtInt(R.n);
    const files = eyeFiles(R.square, R.eye);
    const types = typeCounts(R.type, R.legend).filter(([, c]) => c > 0).map(([t, c]) => `${fmtInt(c)} ${t}`).join(', ');
    $('lede-eyes').innerHTML = `It also has eyes: <b>${fmtInt(R.n)}</b> photoreceptors of the fly's two compound eyes (${escapeHtml(types)}) each watch one square of the board — the left eye sees files ${files.left}, the right eye files ${files.right} — and the picture travels through the real lamina, medulla and lobula before the central brain hears about it.`;
    $('about-eyes').innerHTML = `<b>The eyes.</b> This specimen also sees. ${fmtInt(R.n)} photoreceptors (${escapeHtml(types)}) sit on the ommatidial columns of the two compound eyes mapped by FlyWire; every photoreceptor looks at exactly one board square — the left eye covers files ${files.left} (${fmtInt(files.leftCount)} photoreceptors), the right eye files ${files.right} (${fmtInt(files.rightCount)}) — and receives the twenty board planes of that square through a tiny learned tuning, nothing else. The signal then has to cross the real lamina → medulla → lobula wiring to reach the central brain. The <i>fly's eye</i> panel beside the board shows what each photoreceptor watches and, in <i>feels</i> mode, the actual input current it receives.`;
  }

  _renderLegend(info) {
    const counts = {};
    for (const c of info.silhouette.cls) counts[c] = (counts[c] || 0) + 1;
    const el = $('legend');
    el.innerHTML = '';
    Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([cls]) => {
      const name = info.legend[cls] || `class ${cls}`;
      const [r, g, b] = this.viz.colorOf(+cls);
      const span = document.createElement('span');
      span.style.setProperty('--c', `rgb(${r},${g},${b})`);
      span.textContent = name.replace(/_/g, ' ');
      el.appendChild(span);
    });
  }

  // ---------------------------------------------------------------- ui wiring
  _bindUI() {
    $('btn-start').addEventListener('click', () => {
      this.party = null;   // the landing form always starts a solo game
      this.newGame({
        difficulty: document.querySelector('input[name=difficulty]:checked').value,
        color: document.querySelector('input[name=color]:checked').value,
        name: $('player-name').value.trim() || 'Human',
      });
    });
    $('brand').addEventListener('click', (e) => { e.preventDefault(); this.showLanding(); });
    $('btn-new').addEventListener('click', () => { if (this.party) this.newGame({ ...this.state.opts, name: this.party.names[this.party.current] }); else this.showLanding(); });
    $('btn-undo').addEventListener('click', () => this.undo());
    $('btn-flip').addEventListener('click', () => this.flip());
    $('btn-retry').addEventListener('click', () => this.retryFlyMove());
    $('btn-resign').addEventListener('click', () => this.resign());
    $('btn-pgn').addEventListener('click', () => this.copyText(this.pgn(), 'PGN copied'));
    $('btn-sound').addEventListener('click', () => {
      const on = this.sounds.toggle();
      $('btn-sound').setAttribute('aria-pressed', String(on));
      $('btn-sound').querySelector('.snd-on').hidden = !on; $('btn-sound').querySelector('.snd-off').hidden = on;
      if (on) this.sounds.click('move');
    });
    $('btn-about').addEventListener('click', () => $('about-modal').showModal());
    $('btn-leaderboard').addEventListener('click', () => { this.renderLeaderboard(); $('leaderboard-modal').showModal(); });
    $('btn-clear-board').addEventListener('click', () => { saveBoard([]); this.renderLeaderboard(); });
    $('btn-party').addEventListener('click', () => $('party-modal').showModal());
    $('btn-party-start').addEventListener('click', () => this.startParty());
    $('btn-rematch').addEventListener('click', () => { $('result-modal').close(); this.newGame({ ...this.state.opts }); });
    $('btn-next-player').addEventListener('click', () => { $('result-modal').close(); this.nextPartyPlayer(); });
    $('btn-copy-image').addEventListener('click', () => this.copyImage());
    $('btn-download-image').addEventListener('click', () => this.downloadImage());
    $('btn-copy-text').addEventListener('click', () => this.copyText(this.shareText(), 'Copied'));
    $('player-name').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !$('btn-start').disabled) { e.preventDefault(); $('btn-start').click(); }
    });
    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea') || !this.state) return;
      if (e.key === 'Escape') this.board?.cancelPromotion();
      if (document.querySelector('dialog[open]')) return;   // no board shortcuts behind an open modal
      if (e.key === 'f' && !e.ctrlKey && !e.metaKey && !e.altKey) this.flip();   // plain f only: Ctrl+F is find
      if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.undo(); }
    });
    this.brain.onThinking = (m) => { $('clk-sims').textContent = `${m.done}/${m.total}`; };
    // thought replay: scrubber + play button follow the brain canvas
    const scrub = $('replay-scrub'), stepEl = $('replay-step'), playBtn = $('replay-play');
    this.viz.onReplay = (pos, steps, playing) => {
      scrub.max = String(steps - 1);
      if (document.activeElement !== scrub || !playing) scrub.value = String(pos);
      stepEl.textContent = `step ${Math.min(steps, Math.floor(pos + 1e-6) + 1)} / ${steps}`;
      playBtn.textContent = playing ? '❚❚' : '▶';
      playBtn.setAttribute('aria-label', playing ? 'pause the replay' : "replay the fly's thought");
      this.strip.setPos(pos);
    };
    scrub.addEventListener('input', () => this.viz.seek(+scrub.value));
    playBtn.addEventListener('click', () => this.viz.toggle());
    $('replay-look').addEventListener('click', () => this.lookAgain());
  }

  /** A trace (steps × sample) from the worker: replay it on the brain canvas and the class strip. */
  showTrace(trace, steps, { autoplay = true } = {}) {
    if (!trace || !steps || !this.viz.sample || trace.length !== steps * this.viz.sample.idx.length) { this.hideTrace(); return; }
    $('replay').hidden = false; $('strip-canvas').hidden = false;
    $('strip-canvas').parentElement.classList.add('has-trace');
    this.strip.setTrace(trace, steps, this.groups);
    this.viz.setTrace(trace, steps, { autoplay });
    $('brain-caption').textContent = `${fmtInt(this.viz.sample.idx.length)} sampled neurons · ${steps} timesteps · mean |activity| per class`;
  }

  hideTrace() { $('replay').hidden = true; $('strip-canvas').hidden = true; $('strip-canvas').parentElement.classList.remove('has-trace'); }

  /** "look again": ask the network to evaluate the current position and replay that thought (no move is made). */
  async lookAgain() {
    const s = this.state;
    if (!s || s.thinking) return;
    const btn = $('replay-look'); btn.disabled = true;
    try {
      const r = await this.brain.eval(this.chess.fen(), s.moves.slice());
      if (this.state !== s || s.thinking) return;
      this.showTrace(r.trace, r.traceSteps);
      if (r.backend) this._showBackend(r.backend);
    } catch (err) { this.toast(`The fly would not look (${err.message})`); }
    finally { btn.disabled = false; }
  }

  showLanding() {
    this.party = null; this.renderPartyBar();   // leaving the board ends the party; later games are solo
    $('screen-game').hidden = true; $('screen-landing').hidden = false;
    this.viz.setThinking(false); this.sounds.stopBuzz();
    window.scrollTo({ top: 0 });
  }

  flip() {
    if (!this.board) return;
    this.board.flip();
    if (this.state) this.updateTurn();   // the who-top / who-bottom labels follow the orientation
  }

  toast(msg) {
    const t = $('toast'); t.textContent = msg; t.hidden = false;
    clearTimeout(this._toastT); this._toastT = setTimeout(() => { t.hidden = true; }, 2200);
  }

  // ---------------------------------------------------------------- game flow
  newGame(opts) {
    const color = opts.color === 'random' ? (Math.random() < 0.5 ? 'white' : 'black') : opts.color;
    this.chess = new Chess();
    this.state = {
      opts, human: color === 'white' ? 'w' : 'b', difficulty: opts.difficulty, name: opts.name,
      moves: [], thinkTotal: 0, lastThink: 0, sims: 0, result: null, started: performance.now(), thinking: false, lastValue: 0,
    };
    this.pendingMoveId++;
    this.brain.cancelMoves();
    if (!this.board) {
      this.board = new Board($('board'), { onMove: (f, t, p) => this.humanMove(f, t, p), orientation: color });
    } else this.board.setOrientation(color);
    this.board.setPosition(this.chess.fen(), { animate: false });
    this.board.highlight({ lastMove: null, check: null });
    this.eye.setGame(this.state.human === 'w' ? 'b' : 'w');
    this.eye.setBoard(this.chess.fen(), null);
    this.glance = null;                       // per-(eye, square) drive of the fly's previous look, for the gaze commentary
    this.hideTrace();
    $('screen-landing').hidden = true; $('screen-game').hidden = false;
    $('diff-pill').textContent = DIFF_LABEL[opts.difficulty];
    $('clk-last').textContent = '—'; $('clk-total').textContent = '0.0 s'; $('clk-sims').textContent = opts.difficulty === 'superfly' ? '0 sims' : '—';
    $('btn-resign').disabled = false;
    this.renderMoves();
    this.renderPartyBar();
    this.setMood('curious', 'The fly is watching the board.');
    this.setValue(0);
    this.updateTurn();
    if (this.chess.turn() !== this.state.human) this.flyMove();
    else this.brain.eval(this.chess.fen(), []).then((r) => {
      // the fly's opinion of the start position: value is the mover's (yours), so flip it for the fly
      if (this.state && this.state.moves.length === 0 && !this.state.thinking) {
        if (r.trace) this.showTrace(r.trace, r.traceSteps); else this.viz.setActivity(r.activitySample);
        this.setValue(-r.value);
        this.setMood(moodFor(-r.value), commentaryFor({ policyTop: r.policyTop, value: -r.value }));
      }
    }).catch(() => {});
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  updateTurn() {
    const s = this.state;
    const humanTurn = this.chess.turn() === s.human && !s.result;
    this.board.setMovable(humanTurn ? s.human : null);
    this.board.setLegal(humanTurn ? this.chess.moves({ verbose: true }) : []);
    const flyName = `the fly (${DIFF_LABEL[s.difficulty]})`;
    const top = this.board.orientation === 'white' ? 'b' : 'w';
    const who = (c) => (c === s.human ? `<b>${escapeHtml(s.name)}</b>` : `<b>${flyName}</b>`) + (this.chess.turn() === c && !s.result ? ' <span class="turn">· to move</span>' : '');
    $('who-top').innerHTML = who(top); $('who-bottom').innerHTML = who(top === 'w' ? 'b' : 'w');
    const st = $('status');
    st.classList.toggle('over', !!s.result);
    st.classList.remove('error'); $('btn-retry').hidden = true;
    if (s.result) st.textContent = s.result.text;
    else if (this.chess.isCheck()) st.textContent = humanTurn ? 'Check! Your move.' : 'The fly is in check…';
    else st.textContent = humanTurn ? 'Your move.' : 'The fly is thinking…';
    $('btn-undo').disabled = !humanTurn || s.moves.length < 2;
    const inCheck = this.chess.isCheck();
    this.board.highlight({ check: inCheck ? this.kingSquare(this.chess.turn()) : null });
  }

  kingSquare(color) {
    const b = this.chess.board();
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) { const p = b[r][f]; if (p && p.type === 'k' && p.color === color) return 'abcdefgh'[f] + (8 - r); }
    return null;
  }

  applyMove(mv) {
    const s = this.state;
    s.moves.push(mv.from + mv.to + (mv.promotion || ''));
    this.board.setPosition(this.chess.fen());
    this.board.highlight({ lastMove: [mv.from, mv.to] });
    this.eye.setBoard(this.chess.fen(), mv.from + mv.to);
    this.sounds.click(mv.captured ? 'capture' : 'move');
    this.renderMoves();
    const over = this.checkGameOver();
    this.updateTurn();
    return over;
  }

  humanMove(from, to, promotion) {
    const s = this.state;
    if (!s || s.thinking || s.result || this.chess.turn() !== s.human) return;
    let mv;
    try { mv = this.chess.move({ from, to, promotion }); } catch { return; }
    if (!mv) return;
    if (!this.applyMove(mv)) this.flyMove();
  }

  async flyMove() {
    const s = this.state;
    if (!s || s.result) return;
    s.thinking = true;
    const id = ++this.pendingMoveId;
    this.setMood('thinking', null);
    this.viz.setThinking(true);
    this.sounds.startBuzz();
    $('commentary').classList.add('fade');
    const fen = this.chess.fen();
    let r;
    try {
      r = await this.brain.move(fen, s.moves.slice(), s.difficulty);
    } catch (err) {
      if (id !== this.pendingMoveId || this.state !== s) return;   // game was reset meanwhile: leave the new game's UI alone
      s.thinking = false; this.viz.setThinking(false); this.sounds.stopBuzz();
      $('commentary').classList.remove('fade');
      this.setMood('nervous', `The fly brain crashed (${err.message}).`);
      this.showRetry('The fly brain crashed.');
      return;
    }
    if (id !== this.pendingMoveId || this.state !== s) return;   // game was reset meanwhile
    s.thinking = false;
    this.viz.setThinking(false);
    this.sounds.stopBuzz();
    $('commentary').classList.remove('fade');
    if (!r.move) { this.checkGameOver(); this.updateTurn(); return; }
    // a small pause so the fly visibly "decides" even when the network is fast
    const wait = Math.max(0, 450 - r.thinkMs);
    if (wait) await new Promise((res) => setTimeout(res, wait));
    if (id !== this.pendingMoveId) return;
    const mv = this.chess.move({ from: r.move.slice(0, 2), to: r.move.slice(2, 4), promotion: r.move[4] || undefined });
    s.lastThink = r.thinkMs; s.thinkTotal += r.thinkMs; s.sims = r.sims; s.lastValue = r.value;
    $('clk-last').textContent = fmtMs(r.thinkMs); $('clk-total').textContent = fmtMs(s.thinkTotal);
    $('clk-sims').textContent = s.difficulty === 'superfly' ? `${r.sims} sims` : (r.sims ? `1-ply × ${r.sims}` : 'policy only');
    if (r.backend) this._showBackend(r.backend);
    if (r.trace) this.showTrace(r.trace, r.traceSteps); else this.viz.setActivity(r.activitySample);
    this.setValue(r.value);
    let text = commentaryFor({ policyTop: r.policyTop, value: r.value, san: mv.san, difficulty: s.difficulty, sims: r.sims });
    const gaze = this.seeGaze(r.retinaDrive, s.moves.length);
    if (gaze && Math.random() < 0.45) text += ` ${gaze}`;
    this.setMood(moodFor(r.value), text);
    this.applyMove(mv);
  }

  /**
   * The retina drive of the position the fly just looked at (it was the side to move, so the
   * photoreceptors' squares are in its own perspective): feed the eye panel and, comparing with the
   * previous glance, find the (eye, square) whose input current changed most — a commentary line
   * driven purely by the network's inputs.
   */
  seeGaze(drive, ply) {
    if (!drive || !this.eye.hasRetina) return '';
    const R = this.eye.retina;
    const moveNo = Math.floor(ply / 2) + 1;
    this.eye.setDrive(drive, `at its last glance (move ${moveNo})`);
    const cur = squareDrive(drive, R.square, R.eye);
    const shift = gazeShift(this.glance, cur);
    this.glance = cur;
    return gazeLine(shift, this.eye.board, this.eye.flyColor);
  }

  /** Persistent way out of a failed fly move (a toast alone would leave the board locked forever). */
  showRetry(text) {
    const st = $('status');
    st.textContent = text; st.classList.add('error');
    $('btn-retry').hidden = false;
  }

  async retryFlyMove() {
    const s = this.state;
    if (!s || s.result || s.thinking) return;
    $('btn-retry').hidden = true; $('status').classList.remove('error');
    $('status').textContent = 'The fly is thinking…';
    if (this.brain.dead) {
      try { await this.brain.restart(); } catch (err) {
        if (this.state !== s || s.result) return;
        this.toast(`Reload failed: ${err.message}`);
        this.showRetry('The fly brain could not be reloaded.');
        return;
      }
    }
    if (this.state === s && !s.result && !s.thinking) this.flyMove();
  }

  undo() {
    const s = this.state;
    if (!s || s.thinking || s.result || s.moves.length < 2 || this.chess.turn() !== s.human) return;
    this.chess.undo(); this.chess.undo();
    s.moves.length -= 2;
    this.board.setPosition(this.chess.fen());
    const last = this.chess.history({ verbose: true }).at(-1);
    this.board.highlight({ lastMove: last ? [last.from, last.to] : null });
    this.eye.setBoard(this.chess.fen(), last ? last.from + last.to : null);
    this.eye.setDrive(null); this.glance = null;
    this.renderMoves();
    this.updateTurn();
    this.setMood('curious', 'The fly pretends that never happened.');
  }

  resign() {
    const s = this.state;
    if (!s || s.result) return;
    this.pendingMoveId++; this.brain.cancelMoves(); s.thinking = false; this.viz.setThinking(false); this.sounds.stopBuzz();
    this.finish({ outcome: 'loss', reason: 'resignation', text: 'You resigned. The fly wins.' });
  }

  checkGameOver() {
    const c = this.chess;
    // chess.js' hash-based threefold check misses repetitions whose first occurrence followed a
    // double push with a pinned (illegal) en-passant capture; the FEN-based count does not
    const threefold = repetitionCount(c) >= 3;
    if (!c.isGameOver() && !threefold) return false;
    const s = this.state;
    if (c.isCheckmate()) {
      const winner = c.turn() === 'w' ? 'b' : 'w';
      const humanWon = winner === s.human;
      this.finish({ outcome: humanWon ? 'win' : 'loss', reason: 'checkmate', text: humanWon ? 'Checkmate! You beat the fly.' : 'Checkmate. The fly wins.' });
    } else {
      const reason = c.isStalemate() ? 'stalemate' : threefold ? 'threefold repetition' : c.isInsufficientMaterial() ? 'insufficient material' : 'fifty-move rule';
      this.finish({ outcome: 'draw', reason, text: `Draw by ${reason}.` });
    }
    return true;
  }

  finish(result) {
    const s = this.state;
    s.result = result;
    s.timeMs = performance.now() - s.started;
    s.plies = s.moves.length;
    s.fullMoves = Math.ceil(s.moves.length / 2);
    this.board.setMovable(null);
    this.updateTurn();
    $('btn-resign').disabled = true;
    this.sounds.chord(result.outcome === 'win');
    const mood = result.outcome === 'win' ? 'panicking' : result.outcome === 'loss' ? 'smug' : 'curious';
    this.setMood(mood, result.outcome === 'win' ? 'The fly has been swatted.' : result.outcome === 'loss' ? 'The fly grooms its wings, victorious.' : 'The fly accepts the draw. Probably.');
    const row = { name: s.name, result: result.outcome, reason: result.reason, difficulty: s.difficulty, color: s.human, moves: s.fullMoves, plies: s.plies, timeMs: Math.round(s.timeMs), thinkMs: Math.round(s.thinkTotal), date: new Date().toISOString() };
    saveBoard([row, ...loadBoard()]);
    if (this.party) { this.party.results[this.party.current] = row; this.renderPartyBar(); }
    this.showResult(row);
  }

  // ---------------------------------------------------------------- fly panel
  setMood(mood, text) {
    $('avatar-wrap').dataset.mood = mood;
    $('mood-label').textContent = MOOD_WORDS[mood] || mood;
    const mouth = $('avatar').querySelector('.mouth');
    if (mouth) mouth.setAttribute('d', MOUTHS[mood] || MOUTHS.curious);
    if (text !== null && text !== undefined) $('commentary').textContent = text;
  }

  setValue(v) {
    $('value-fill').style.left = `${50 + 50 * Math.max(-1, Math.min(1, v))}%`;
    $('value-fill').title = `value head: ${v.toFixed(2)}`;
  }

  // ---------------------------------------------------------------- move list / pgn
  renderMoves() {
    const ol = $('moves');
    const hist = this.chess.history();
    ol.innerHTML = '';
    for (let i = 0; i < hist.length; i += 2) {
      const li = document.createElement('li');
      const white = this.state.human === 'w' ? '' : ' fly', black = this.state.human === 'b' ? '' : ' fly';
      li.innerHTML = `<span class="num">${i / 2 + 1}.</span><span class="san${white}${i === hist.length - 1 ? ' cur' : ''}">${hist[i]}</span><span class="san${black}${i + 1 === hist.length - 1 ? ' cur' : ''}">${hist[i + 1] || ''}</span>`;
      ol.appendChild(li);
    }
    ol.scrollTop = ol.scrollHeight;
  }

  pgn() {
    const s = this.state;
    const flyName = `Fruit fly brain (${DIFF_LABEL[s.difficulty]})`;
    this.chess.setHeader('Event', 'Human vs fruit fly brain');
    this.chess.setHeader('Site', location.host || 'local');
    this.chess.setHeader('Date', new Date().toISOString().slice(0, 10).replace(/-/g, '.'));
    this.chess.setHeader('White', s.human === 'w' ? s.name : flyName);
    this.chess.setHeader('Black', s.human === 'b' ? s.name : flyName);
    const res = !s.result ? '*' : s.result.outcome === 'draw' ? '1/2-1/2' : (s.result.outcome === 'win') === (s.human === 'w') ? '1-0' : '0-1';
    this.chess.setHeader('Result', res);
    const pgn = this.chess.pgn();
    return pgn.trimEnd().endsWith(res) ? pgn : `${pgn} ${res}`;
  }

  // ---------------------------------------------------------------- result / share
  shareText() {
    const s = this.state, r = s.result;
    const head = r.outcome === 'win' ? 'I beat a fruit fly brain at chess' : r.outcome === 'loss' ? 'A fruit fly brain beat me at chess' : 'I drew with a fruit fly brain at chess';
    return `${head} — ${DIFF_LABEL[s.difficulty]} difficulty, ${plural(s.fullMoves, 'move')}, ${fmtClock(s.timeMs)}. ${location.href.split('#')[0]}`;
  }

  async showResult(row) {
    const s = this.state, r = s.result;
    $('result-eyebrow').textContent = `assay complete · ${DIFF_LABEL[s.difficulty]} · ${r.reason}`;
    $('result-title').textContent = r.outcome === 'win' ? `${s.name} beat the fly` : r.outcome === 'loss' ? `The fly beat ${s.name}` : `${s.name} drew with the fly`;
    $('result-sub').textContent = `${plural(s.fullMoves, 'move')} · ${fmtClock(s.timeMs)} · the fly thought for ${fmtMs(s.thinkTotal)} in total`;
    $('btn-next-player').hidden = !this.party || this.party.current >= this.party.names.length - 1;
    $('btn-rematch').textContent = this.party ? 'Replay this round' : 'Play again';
    $('share-hint').textContent = '';
    await this.drawShareCard(row);
    $('result-modal').showModal();
  }

  async drawShareCard() {
    const canvas = $('share-canvas'), ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const s = this.state, r = s.result;
    ctx.fillStyle = '#0e100b'; ctx.fillRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * 0.85, -50, 20, W * 0.85, -50, 700);
    g.addColorStop(0, 'rgba(139,224,90,.14)'); g.addColorStop(1, 'rgba(139,224,90,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // brain silhouette top right, final position below it
    if (this.viz.silhouette) {
      const sil = this.viz.silhouette, b = this.viz.bounds;
      const bw = 380, bh = 300, ox = 770, oy = 30;
      const sc = Math.min(bw / (b.maxX - b.minX), bh / (b.maxY - b.minY));
      const cx = ox + (bw - sc * (b.maxX - b.minX)) / 2, cy = oy + (bh - sc * (b.maxY - b.minY)) / 2;
      for (let i = 0; i < sil.cls.length; i++) {
        const [cr, cg, cb] = this.viz.colorOf(sil.cls[i]);
        ctx.fillStyle = `rgba(${cr},${cg},${cb},.4)`;
        ctx.fillRect(cx + (sil.xy[2 * i] - b.minX) * sc, cy + (sil.xy[2 * i + 1] - b.minY) * sc, 2.2, 2.2);
      }
    }
    try { const img = await this.boardImage(); ctx.drawImage(img, 880, 350, 240, 240); } catch { /* ignore */ }
    if (this.flySvg) {
      try { const img = await svgToImage(this.flySvg); ctx.drawImage(img, 56, 40, 130, 130); } catch { /* ignore */ }
    }
    const MAXW = 640;
    // draw text shrinking the size (down to 70%) before truncating with an ellipsis
    const line = (text, x, y, font, color) => {
      const m = /^(.*?)(\d+(?:\.\d+)?)px(.*)$/.exec(font);
      let size = +m[2];
      const setFont = () => { ctx.font = `${m[1]}${size}px${m[3]}`; };
      setFont(); ctx.fillStyle = color;
      while (ctx.measureText(text).width > MAXW && size > +m[2] * 0.7) { size -= 1; setFont(); }
      let t = text;
      while (t.length > 3 && ctx.measureText(t).width > MAXW) t = t.slice(0, -2).trimEnd() + '…';
      ctx.fillText(t, x, y);
    };
    const mono = 'ui-monospace, Menlo, Consolas, monospace', serif = '"Iowan Old Style", Palatino, Georgia, serif', sans = 'system-ui, sans-serif';
    line('DROSOPHILA MELANOGASTER · CHESS ASSAY', 210, 88, `600 22px ${mono}`, '#8be05a');
    const title = r.outcome === 'win' ? 'I beat a fruit fly brain' : r.outcome === 'loss' ? 'A fruit fly brain beat me' : 'I drew with a fruit fly brain';
    line(title, 210, 160, `italic 64px ${serif}`, '#ece5cf');
    line('at chess.', 210, 232, `italic 64px ${serif}`, '#ece5cf');
    line(`${cleanText(s.name)} · ${DIFF_LABEL[s.difficulty]} difficulty · ${s.human === 'w' ? 'white' : 'black'}`, 210, 310, `30px ${sans}`, '#ffcb6b');
    line(`${plural(s.fullMoves, 'move')} · ${fmtClock(s.timeMs)} · by ${r.reason}`, 210, 356, `26px ${mono}`, '#949a80');
    const h = this.brain.info?.header || {};
    line(`${fmtInt(h.n || 0)} neurons · ${fmtInt(h.nnz || 0)} real connections`, 210, 396, `26px ${mono}`, '#949a80');
    line('FlyWire connectome · wired like the real fly', 210, 432, `26px ${mono}`, '#949a80');
    const sans_ = this.chess.history();
    const movesLine = sans_.map((m, i) => (i % 2 === 0 ? `${i / 2 + 1}.` : '') + m).join(' ');
    line(movesLine, 210, 500, `22px ${mono}`, '#6b7160');
    line(location.host ? `${location.host}${location.pathname.replace(/index\.html$/, '')}` : 'flychess', 210, 570, `600 24px ${sans}`, '#e9a63a');
  }

  boardImage() {
    const svg = this.board.svg.cloneNode(true);
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.setAttribute('width', '480'); svg.setAttribute('height', '480');
    const css = getComputedStyle(document.documentElement);
    const v = (n) => css.getPropertyValue(n).trim();
    svg.querySelectorAll('.cb-dots, .cb-promo, .cb-coords, .cb-cursor').forEach((e) => e.remove());
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = `.cb-light{fill:${v('--sq-light')}}.cb-dark{fill:${v('--sq-dark')}}.cb-last{fill:${v('--hl-last')}}.cb-check{fill:${v('--hl-check')}}.cb-selected{fill:none}
      .cb-piece{stroke-width:3;stroke-linejoin:round}.cb-w{fill:${v('--piece-w')};stroke:${v('--piece-w-ink')}}.cb-w .ink{fill:${v('--piece-w-ink')};stroke:${v('--piece-w-ink')}}
      .cb-b{fill:${v('--piece-b')};stroke:${v('--piece-b-ink')}}.cb-b .ink{fill:${v('--piece-b-ink')};stroke:${v('--piece-b-ink')}}.cb-vanish{display:none}`;
    svg.prepend(style);
    // inline transforms (CSS transforms on <g> are not serialised)
    svg.querySelectorAll('.cb-piece').forEach((el) => {
      const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(el.style.transform || '');
      if (m) el.setAttribute('transform', `translate(${m[1]} ${m[2]})`);
      el.removeAttribute('style');
    });
    return svgToImage(new XMLSerializer().serializeToString(svg));
  }

  async copyImage() {
    try {
      const blob = await new Promise((res) => $('share-canvas').toBlob(res, 'image/png'));
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      $('share-hint').textContent = 'image copied';
    } catch (err) {
      $('share-hint').textContent = `copy failed (${err.name}); try Download`;
    }
  }

  async downloadImage() {
    try {
      const blob = await new Promise((res) => $('share-canvas').toBlob(res, 'image/png'));
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = `flychess-${this.state.result.outcome}-${Date.now()}.png`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      $('share-hint').textContent = 'downloading… (if nothing happens, use Copy image)';
    } catch (err) { $('share-hint').textContent = `download blocked (${err.name}); use Copy image`; }
  }

  async copyText(text, msg) {
    try { await navigator.clipboard.writeText(text); this.toast(msg); }
    catch { this.toast('Clipboard blocked — select and copy manually'); window.prompt('Copy:', text); }
  }

  // ---------------------------------------------------------------- leaderboard
  renderLeaderboard() {
    const rows = rankRows(loadBoard());
    const t = $('leaderboard-table');
    if (!rows.length) { t.innerHTML = '<tr><td class="empty">No games yet. Beat the fly and come back.</td></tr>'; return; }
    t.innerHTML = '<tr><th>#</th><th>player</th><th>result</th><th>vs</th><th>moves</th><th>time</th><th>date</th></tr>' + rows.slice(0, 50).map((r, i) => `<tr><td>${i + 1}</td><td>${escapeHtml(r.name)}</td><td class="${r.result}">${r.result}</td><td>${DIFF_LABEL[r.difficulty] || r.difficulty}</td><td>${r.moves}</td><td>${fmtClock(r.timeMs)}</td><td>${r.date.slice(0, 10)}</td></tr>`).join('');
  }

  // ---------------------------------------------------------------- party mode
  startParty() {
    const names = $('party-names').value.split('\n').map((s) => s.trim()).filter(Boolean).slice(0, 32);
    if (names.length < 2) { this.toast('Enter at least two names'); return; }
    $('party-modal').close();
    this.party = { names, current: 0, results: [], difficulty: $('party-difficulty').value, color: $('party-color').value };
    this.newGame({ difficulty: this.party.difficulty, color: this.party.color, name: names[0] });
  }

  nextPartyPlayer() {
    const p = this.party;
    if (!p) return;
    p.current = Math.min(p.current + 1, p.names.length - 1);
    this.newGame({ difficulty: p.difficulty, color: p.color, name: p.names[p.current] });
  }

  renderPartyBar() {
    const el = $('party-bar'), p = this.party;
    if (!p) { el.hidden = true; return; }
    el.hidden = false;
    const ranked = rankRows(p.results.filter(Boolean));
    const leader = ranked[0];
    const done = p.results.filter(Boolean).length;
    const rows = p.names.map((n, i) => {
      const r = p.results[i];
      const cls = i === p.current && !r ? 'now' : r ? 'done' : '';
      const lead = leader && r === leader && r.result === 'win' ? ' lead' : '';
      const res = r ? `${r.result === 'win' ? 'beat the fly' : r.result === 'loss' ? 'lost' : 'drew'} · ${r.moves} mv · ${fmtClock(r.timeMs)}` : i === p.current ? 'playing…' : 'waiting';
      return `<li class="${cls}${lead}"><span>${i + 1}. ${escapeHtml(n)}</span><span class="res">${res}</span></li>`;
    });
    const champ = done === p.names.length ? (leader && leader.result === 'win' ? `${escapeHtml(leader.name)} beat the fly fastest.` : 'Nobody beat the fly. The fly wins the party.') : `${done}/${p.names.length} played`;
    el.innerHTML = `<h3>Party · ${champ}</h3><ol>${rows.join('')}</ol>`;
    if (done === p.names.length) el.innerHTML += `<div class="row"><button class="btn btn-ghost" id="btn-party-end" type="button">End party</button></div>`;
    $('btn-party-end')?.addEventListener('click', () => { this.party = null; this.renderPartyBar(); });
  }
}

// ============================================================================ helpers
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function cleanText(s) { return String(s).replace(/[\u0000-\u001f]/g, ''); }
function svgToImage(svgText) {
  return new Promise((resolve, reject) => {
    const blob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}

window.flychess = new App();
