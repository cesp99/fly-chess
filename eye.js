// eye.js — "what the fly sees": the two compound eyes drawn from the blob's retina map (SPEC §8,
// docs/RETINA.md). Every photoreceptor of the network looks at exactly one board square; the panel
// colours each one by the CURRENT board as the fly sees it from its own side ("sees"), or by the
// input current it actually receives — w_ret · planes + b_ret, the retina drive the engines report
// ("feels"). Nothing here influences the network; it only reads its inputs.
//
// Pure helpers (exported, tested in test/eye.test.mjs) are kept free of DOM so node can import them;
// EyePanel is the canvas widget the page instantiates.

const FILES = 'abcdefgh';
const PIECE_NAME = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
const PIECE_GLYPH = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' };
// piece type → brightness of its dot (kings brightest so the fly's "targets" stand out)
const PIECE_LEVEL = { p: 0.55, n: 0.72, b: 0.72, r: 0.82, q: 0.95, k: 1 };

/** Real board square (a1 = 0 … h8 = 63) behind mover-perspective square `s` when the fly plays `flyColor` ('w' | 'b'). */
export function realSquare(s, flyColor) { return flyColor === 'b' ? s ^ 56 : s; }

/** 28 → 'e4'. */
export function squareName(sq) { return FILES[sq & 7] + (1 + (sq >> 3)); }

/** FEN piece placement → array of 64 entries (a1 = 0): null or {type, color}. */
export function boardFromFen(fen) {
  const out = new Array(64).fill(null);
  const rows = String(fen).split(' ')[0].split('/');
  for (let r = 0; r < Math.min(8, rows.length); r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') { file += +ch; continue; }
      if (file > 7) break;
      out[(7 - r) * 8 + file] = { type: ch.toLowerCase(), color: ch === ch.toUpperCase() ? 'w' : 'b' };
      file++;
    }
  }
  return out;
}

/**
 * Lay the photoreceptors out on a W × H canvas: the left eye (u < 0.5) in the left half, the right
 * eye in the right half, each scaled to its own bounding box so both look the same size. Every
 * ommatidial column (identical uv) holds up to eight photoreceptors: R7 / R8 sit at the centre, the
 * six outer R1-6 on a small ring around it, like the real ommatidium seen from above.
 * Returns pixel centres, the dot radius, and the per-eye boxes.
 */
export function layoutEyes(uv, eye, type, legend, W, H, pad = 8) {
  const n = eye.length;
  const x = new Float32Array(n), y = new Float32Array(n);
  const boxes = [];
  const R16 = legend.indexOf('R1-6');
  const half = W / 2;
  let spacing = 6;
  for (let e = 0; e < 2; e++) {
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity, count = 0;
    const cols = new Map();      // column key → members
    for (let k = 0; k < n; k++) {
      if (eye[k] !== e) continue;
      const u = uv[2 * k] * 2 - e, v = uv[2 * k + 1];   // each eye's u spans half of [0, 1]: stretch it so the eye is round
      minU = Math.min(minU, u); maxU = Math.max(maxU, u); minV = Math.min(minV, v); maxV = Math.max(maxV, v); count++;
      const key = `${u.toFixed(4)},${v.toFixed(4)}`;
      let m = cols.get(key); if (!m) cols.set(key, m = []); m.push(k);
    }
    if (!count) continue;
    const bw = half - 2 * pad, bh = H - 2 * pad;
    const du = Math.max(maxU - minU, 1e-3), dv = Math.max(maxV - minV, 1e-3);
    const s = Math.min(bw / du, bh / dv);
    const ox = e * half + pad + (bw - s * du) / 2, oy = pad + (bh - s * dv) / 2;
    // estimated column pitch: columns fill the box roughly uniformly
    const pitch = Math.sqrt((s * du) * (s * dv) / Math.max(cols.size, 1));
    spacing = Math.min(spacing, pitch);
    const ring = Math.min(0.32 * pitch, 6), centreOff = Math.min(0.1 * pitch, 2);
    for (const members of cols.values()) {
      const k0 = members[0];
      const cx = ox + (uv[2 * k0] * 2 - e - minU) * s;
      const cy = oy + (maxV - uv[2 * k0 + 1]) * s;    // v = 1 is dorsal = up
      let outer = 0, centre = 0;
      for (const k of members) {
        if (type[k] === R16 && members.length > 1) {
          const a = (outer++ % 6) * Math.PI / 3 - Math.PI / 2;
          x[k] = cx + ring * Math.cos(a); y[k] = cy + ring * Math.sin(a);
        } else {
          const off = members.length > 1 ? (centre++ % 2 ? centreOff : -centreOff) : 0;
          x[k] = cx + off; y[k] = cy;
        }
      }
    }
    boxes[e] = { x: e * half + pad, y: pad, w: bw, h: bh, count, columns: cols.size, cx: ox + s * du / 2, cy: oy + s * dv / 2, rx: s * du / 2, ry: s * dv / 2 };
  }
  const r = Math.max(1.1, Math.min(4, spacing * 0.28));
  return { x, y, r, boxes, spacing };
}

/**
 * Colour of photoreceptor `k` in the "sees" view: the square it watches on the real board, from
 * the fly's side. Empty squares keep the board's light / dark tones (dimmed), the fly's own pieces
 * are amber, the opponent's cool blue, both brighter for bigger pieces. Returns [r, g, b, a].
 */
export function seesColor(sq, board, flyColor) {
  const real = realSquare(sq, flyColor);
  const piece = board[real];
  if (!piece) {
    const light = ((real >> 3) + (real & 7)) % 2 === 1;
    return light ? [205, 195, 154, 0.55] : [111, 125, 79, 0.6];
  }
  const level = PIECE_LEVEL[piece.type] || 0.7;
  return piece.color === flyColor ? [233, 166, 58, 0.45 + 0.55 * level] : [127, 184, 230, 0.45 + 0.55 * level];
}

/** Robust scale for the "feels" view: the 98th percentile of |drive| (≥ 1e-6). */
export function driveScale(drive) {
  if (!drive || !drive.length) return 1;
  const a = Float32Array.from(drive, (v) => Math.abs(v)).sort();
  return Math.max(a[Math.min(a.length - 1, Math.floor(a.length * 0.98))], 1e-6);
}

/** Colour of a photoreceptor by its input current: excitatory (positive) → green, inhibitory → red; brightness = |d| / scale. */
export function feelsColor(d, scale) {
  const m = Math.min(1, Math.abs(d) / scale);
  const a = 0.12 + 0.88 * m;
  return d >= 0 ? [60 + 140 * m, 120 + 135 * m, 50 + 60 * m, a] : [120 + 135 * m, 50 + 30 * m, 40 + 30 * m, a];
}

/**
 * Mean drive per (eye, mover-perspective square): Float32Array(128), index eye * 64 + square,
 * NaN where that eye has no photoreceptor on the square.
 */
export function squareDrive(drive, square, eye) {
  const sum = new Float32Array(128), cnt = new Int32Array(128);
  for (let k = 0; k < square.length; k++) { const i = eye[k] * 64 + square[k]; sum[i] += drive[k]; cnt[i]++; }
  for (let i = 0; i < 128; i++) sum[i] = cnt[i] ? sum[i] / cnt[i] : NaN;
  return sum;
}

/**
 * Where the fly's gaze moved: the (eye, square) whose mean input current changed most between two
 * glances. Returns {eye: 0|1, square (mover perspective), delta} or null when nothing changed.
 */
export function gazeShift(prev, cur, minDelta = 1e-3) {
  if (!prev || !cur) return null;
  let best = -1, bestD = minDelta;
  for (let i = 0; i < 128; i++) {
    const d = Math.abs(cur[i] - prev[i]);
    if (Number.isFinite(d) && d > bestD) { bestD = d; best = i; }
  }
  return best < 0 ? null : { eye: best >> 6, square: best & 63, delta: bestD };
}

/** "the fly's right eye is fixed on your knight on f6" — from a gaze shift, the position it looked at and its colour. */
export function gazeLine(shift, board, flyColor, rnd = Math.random) {
  if (!shift) return '';
  const real = realSquare(shift.square, flyColor);
  const piece = board[real];
  const name = squareName(real);
  const what = !piece ? `the empty square ${name}` : `${piece.color === flyColor ? 'its own' : 'your'} ${PIECE_NAME[piece.type]} on ${name}`;
  const eyeName = shift.eye ? 'right' : 'left';
  const lines = [
    `The fly's ${eyeName} eye is fixed on ${what}.`,
    `Its ${eyeName} eye lit up over ${what}.`,
    `Photoreceptors in the ${eyeName} eye twitch at ${what}.`,
  ];
  return lines[Math.floor(rnd() * lines.length) % lines.length];
}

/** Which files each eye covers, e.g. {left: 'a–d', right: 'e–h'} (mover perspective; files do not mirror). */
export function eyeFiles(square, eye) {
  const seen = [new Set(), new Set()], counts = [0, 0];
  for (let k = 0; k < square.length; k++) { seen[eye[k]].add(square[k] & 7); counts[eye[k]]++; }
  const fmt = (set) => {
    const f = [...set].sort((a, b) => a - b);
    if (!f.length) return '—';
    const contiguous = f.every((v, i) => i === 0 || v === f[i - 1] + 1);
    return contiguous && f.length > 1 ? `${FILES[f[0]]}–${FILES[f[f.length - 1]]}` : f.map((v) => FILES[v]).join(', ');
  };
  return { left: fmt(seen[0]), right: fmt(seen[1]), leftCount: counts[0], rightCount: counts[1] };
}

/** Photoreceptor counts per type in legend order, e.g. [['R1-6', 4141], ['R7', 654], ['R8', 748]]. */
export function typeCounts(type, legend) {
  const c = new Array(legend.length).fill(0);
  for (let k = 0; k < type.length; k++) if (type[k] < c.length) c[type[k]]++;
  return legend.map((name, i) => [name, c[i]]);
}

/** Squares touched by a UCI move ('e2e4' → [12, 28]) in real coordinates. */
export function moveSquares(uci) {
  if (!uci || uci.length < 4) return [];
  const idx = (s) => (s.charCodeAt(1) - 49) * 8 + (s.charCodeAt(0) - 97);
  return [idx(uci.slice(0, 2)), idx(uci.slice(2, 4))];
}

// ============================================================================ the widget
const PULSE_MS = 1400;

/**
 * The eye panel: a canvas with both compound eyes, a sees / feels toggle, a hover tooltip.
 *   setRetina(retina | null)          from the worker's 'ready' (null → "no retina" note)
 *   setGame(flyColor)                 which side the fly plays ('w' | 'b')
 *   setBoard(fen, lastMoveUci)        the current board; last-move squares pulse briefly
 *   setDrive(drive, label)            the retina drive of the fly's last glance (Float32Array | null)
 */
export class EyePanel {
  constructor(root) {
    this.root = root;
    this.canvas = root.querySelector('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.tip = root.querySelector('.eye-tip');
    this.caption = root.querySelector('.eye-caption');
    this.note = root.querySelector('.eye-note');
    this.retina = null; this.layout = null; this.files = '';
    this.flyColor = 'w'; this.board = boardFromFen('8/8/8/8/8/8/8/8');
    this.drive = null; this.driveLabel = ''; this.scale = 1;
    this.mode = 'sees';
    this.pulse = { squares: [], t0: 0 };
    this.hover = -1;
    this.raf = 0; this.w = 0; this.h = 0;
    this.onModeChange = () => {};
    root.querySelectorAll('[data-eye-mode]').forEach((b) => b.addEventListener('click', () => this.setMode(b.dataset.eyeMode)));
    this.canvas.addEventListener('pointermove', (e) => this._hover(e));
    this.canvas.addEventListener('pointerleave', () => { this.hover = -1; this.tip.hidden = true; this._draw(); });
    new ResizeObserver(() => this._resize()).observe(this.canvas);
    this._resize();
  }

  get hasRetina() { return !!(this.retina && this.retina.n); }

  setRetina(retina) {
    this.retina = retina && retina.n ? retina : null;
    this.root.classList.toggle('eye-none', !this.hasRetina);
    this.drive = null; this.layout = null;
    if (this.hasRetina) {
      const files = eyeFiles(retina.square, retina.eye);
      this.files = `left eye files ${files.left} · right eye files ${files.right}`;
      this.note.textContent = '';
      this._relayout();
    } else {
      this.note.textContent = 'This specimen has no retina wiring — it smells the board through 2,048 sensory neurons.';
    }
    this._caption();
    this._draw();
  }

  setGame(flyColor) { this.flyColor = flyColor; this.drive = null; this.pulse.squares = []; this._caption(); this._draw(); }

  setBoard(fen, lastMove) {
    this.board = boardFromFen(fen);
    if (lastMove) { this.pulse = { squares: moveSquares(lastMove), t0: performance.now() }; this._loop(); }
    this._draw();
  }

  setDrive(drive, label = '') {
    this.drive = drive || null;
    this.driveLabel = label;
    this.scale = driveScale(drive);
    this._caption();
    this._draw();
  }

  setMode(mode) {
    if (mode !== 'sees' && mode !== 'feels') return;
    this.mode = mode;
    this.root.querySelectorAll('[data-eye-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.eyeMode === mode)));
    this._caption(); this._draw();
    this.onModeChange(mode);
  }

  _caption() {
    if (!this.caption) return;
    if (!this.hasRetina) { this.caption.textContent = ''; return; }
    if (this.mode === 'sees') this.caption.textContent = `${this.retina.n.toLocaleString('en-US')} photoreceptors, the board from the fly's side · amber = its pieces · blue = yours · ${this.files}`;
    else this.caption.textContent = this.drive ? `input current per photoreceptor (w_ret · board + b_ret) ${this.driveLabel} · green = excitatory · red = inhibitory · scaled to this view` : 'input current per photoreceptor — the fly has not looked yet';
  }

  _resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w; this.h = h;
    this._relayout();
    this._draw();
  }

  _relayout() {
    if (!this.hasRetina || !this.w) return;
    const r = this.retina;
    this.layout = layoutEyes(r.uv, r.eye, r.type, r.legend, this.w, this.h);
  }

  _hover(e) {
    if (!this.layout) return;
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const { x, y, r } = this.layout;
    let best = -1, bestD = Math.max(r * 2.5, 6) ** 2;
    for (let k = 0; k < x.length; k++) { const d = (x[k] - mx) ** 2 + (y[k] - my) ** 2; if (d < bestD) { bestD = d; best = k; } }
    if (best === this.hover) return;
    this.hover = best;
    if (best < 0) { this.tip.hidden = true; this._draw(); return; }
    const R = this.retina;
    const real = realSquare(R.square[best], this.flyColor);
    const piece = this.board[real];
    const what = piece ? ` · ${piece.color === this.flyColor ? 'its' : 'your'} ${PIECE_NAME[piece.type]} ${PIECE_GLYPH[piece.type]}` : ' · empty';
    const drive = this.drive ? ` · drive ${this.drive[best] >= 0 ? '+' : ''}${this.drive[best].toFixed(2)}` : '';
    this.tip.textContent = `${R.legend[R.type[best]] || 'R?'} · ${R.eye[best] ? 'right' : 'left'} eye · sees ${squareName(real)}${what}${drive}`;
    this.tip.hidden = false;
    // keep the tooltip inside the panel: to the right of the pointer, else to its left, else clamped
    const tw = this.tip.offsetWidth || 0;
    let left = mx + 12;
    if (left + tw > this.w - 4) left = mx - 12 - tw;
    if (left < 4) left = 4;
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${my + 14}px`;
    this._draw();
  }

  _loop() {
    if (this.raf) return;
    const step = () => {
      this.raf = 0;
      this._draw();
      if (performance.now() - this.pulse.t0 < PULSE_MS) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  _draw() {
    const ctx = this.ctx;
    if (!this.w || !ctx) return;
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.layout) return;
    const R = this.retina, L = this.layout;
    // the two eyes' outlines
    for (const b of L.boxes) {
      if (!b) continue;
      ctx.beginPath(); ctx.ellipse(b.cx, b.cy, b.rx + L.spacing * 0.8, b.ry + L.spacing * 0.8, 0, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(148, 154, 128, .22)'; ctx.lineWidth = 1; ctx.stroke();
    }
    const now = performance.now();
    const pulseK = this.pulse.squares.length ? Math.max(0, 1 - (now - this.pulse.t0) / PULSE_MS) : 0;
    const pulsing = pulseK > 0 ? new Set(this.pulse.squares) : null;
    const hoverSq = this.hover >= 0 ? R.square[this.hover] : -1;
    const feels = this.mode === 'feels';
    const r = L.r;
    if (pulsing) {   // a soft amber glow behind the photoreceptors watching the last move's squares
      ctx.fillStyle = `rgba(255, 203, 107, ${0.22 * pulseK * (0.7 + 0.3 * Math.sin(now / 90))})`;
      for (let k = 0; k < R.n; k++) {
        if (!pulsing.has(realSquare(R.square[k], this.flyColor))) continue;
        ctx.beginPath(); ctx.arc(L.x[k], L.y[k], r + 1.5 + 2 * pulseK, 0, Math.PI * 2); ctx.fill();
      }
    }
    for (let k = 0; k < R.n; k++) {
      let c;
      if (feels) c = this.drive ? feelsColor(this.drive[k], this.scale) : [120, 120, 110, 0.18];
      else c = seesColor(R.square[k], this.board, this.flyColor);
      const px = L.x[k], py = L.y[k];
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0}, ${c[3]})`; ctx.fill();
      if (hoverSq >= 0 && R.square[k] === hoverSq) {
        ctx.beginPath(); ctx.arc(px, py, r + 1.5, 0, Math.PI * 2);
        ctx.strokeStyle = k === this.hover ? '#ffcb6b' : 'rgba(255, 203, 107, .55)'; ctx.lineWidth = k === this.hover ? 1.5 : 1; ctx.stroke();
      }
    }
    // eye labels
    ctx.font = '600 10px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillStyle = 'rgba(148, 154, 128, .8)'; ctx.textAlign = 'left';
    if (L.boxes[0]) ctx.fillText('LEFT EYE', 6, this.h - 5);
    ctx.textAlign = 'right';
    if (L.boxes[1]) ctx.fillText('RIGHT EYE', this.w - 6, this.h - 5);
  }
}
