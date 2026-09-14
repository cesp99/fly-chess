// brainviz.js — the live "fly brain" canvas (sampled neurons glowing at their connectome positions)
// and the thought replay: the worker's activity trace (the sampled neurons after every recurrent
// timestep) scrubbed / played back on the same canvas, with a strip chart of the mean activity per
// super class so the wave retina → optic lobe → central brain → descending neurons is visible.
//
// Pure helpers (exported, tested in test/eye.test.mjs) never touch the DOM.

export const CLASS_COLORS = {
  retina: [255, 240, 150],
  optic: [86, 196, 150], central: [233, 166, 58], sensory: [139, 224, 90], visual_projection: [96, 210, 220],
  visual_centrifugal: [120, 170, 240], ascending: [255, 203, 107], descending: [255, 120, 70], motor: [255, 80, 70],
  sensory_ascending: [190, 240, 120], endocrine: [240, 130, 200], unknown: [150, 150, 150],
};
/** Rows of the strip chart, in the order the signal travels. */
export const FLOW_ORDER = ['retina', 'sensory', 'sensory_ascending', 'optic', 'visual_projection', 'visual_centrifugal', 'ascending', 'central', 'endocrine', 'descending', 'motor', 'unknown'];

/**
 * Group the sampled neurons for the strip chart: one row per super class present in the sample
 * (FLOW_ORDER), plus a 'retina' row for sampled photoreceptors (retinaIdx: blob neuron indices,
 * may be null). Photoreceptors count in the retina row only. Returns [{name, idx: Int32Array}].
 */
export function sampleGroups(sampleIdx, cls, legend, retinaIdx) {
  const ret = retinaIdx && retinaIdx.length ? new Set(retinaIdx) : null;
  const buckets = new Map();
  for (let j = 0; j < cls.length; j++) {
    const name = ret && ret.has(sampleIdx[j]) ? 'retina' : (legend[cls[j]] || 'unknown');
    let b = buckets.get(name); if (!b) buckets.set(name, b = []); b.push(j);
  }
  const order = [...FLOW_ORDER, ...[...buckets.keys()].filter((k) => !FLOW_ORDER.includes(k))];
  return order.filter((k) => buckets.has(k)).map((name) => ({ name, idx: Int32Array.from(buckets.get(name)) }));
}

/**
 * Mean |activity| of every group after every timestep: Float32Array(groups × steps), row-major.
 * trace[t * m + j] is sample neuron j after step t (m = trace.length / steps).
 */
export function traceGroupMeans(trace, steps, groups) {
  const m = Math.floor(trace.length / steps);
  const out = new Float32Array(groups.length * steps);
  for (let g = 0; g < groups.length; g++) {
    const idx = groups[g].idx;
    if (!idx.length) continue;
    for (let t = 0; t < steps; t++) {
      let s = 0;
      const base = t * m;
      for (let i = 0; i < idx.length; i++) s += Math.abs(trace[base + idx[i]]);
      out[g * steps + t] = s / idx.length;
    }
  }
  return out;
}

/** Each row divided by its own maximum (rows that never fire stay 0). */
export function normalizeRows(means, steps) {
  const out = new Float32Array(means.length);
  for (let g = 0; g * steps < means.length; g++) {
    let max = 0;
    for (let t = 0; t < steps; t++) max = Math.max(max, means[g * steps + t]);
    if (max > 0) for (let t = 0; t < steps; t++) out[g * steps + t] = means[g * steps + t] / max;
  }
  return out;
}

/**
 * Display values for the canvas: log-compressed |activity| scaled by a robust maximum — the 97th
 * percentile over the whole trace, so early steps are dim and the brain lights up as it thinks.
 * Returns Float32Array(trace.length) in [0, 1]. A one-step "trace" is a plain activity sample.
 */
export function normalizeTrace(trace) {
  const v = new Float32Array(trace.length);
  for (let i = 0; i < v.length; i++) v[i] = Math.log1p(Math.abs(trace[i]));
  const sorted = Float32Array.from(v).sort();
  const p97 = sorted[Math.floor(sorted.length * 0.97)] || sorted[sorted.length - 1] || 1;
  for (let i = 0; i < v.length; i++) v[i] = Math.min(1, v[i] / p97);
  return v;
}

/** Fractional replay position p ∈ [0, steps-1] → the interpolated row (into `out`). */
export function traceRow(norm, m, p, out) {
  const t0 = Math.floor(p), t1 = Math.min(t0 + 1, norm.length / m - 1), k = p - t0;
  const a = t0 * m, b = t1 * m;
  for (let j = 0; j < m; j++) out[j] = norm[a + j] + (norm[b + j] - norm[a + j]) * k;
  return out;
}

// ============================================================================ brain canvas
const STEP_MS = 150;       // playback: one recurrent timestep every 150 ms (16 steps ≈ 2.4 s)

export class BrainCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.sample = null; this.silhouette = null; this.legend = [];
    this.values = null; this.target = null; this.from = null; this.tStart = 0;
    this.pulse = 0; this.thinking = false;
    this.sprites = {};
    this.raf = 0;
    this.bounds = null;
    // replay state
    this.trace = null; this.steps = 1; this.m = 0; this.pos = 0; this.playing = false; this.playT0 = 0; this.playFrom = 0;
    this.onReplay = () => {};     // (pos, steps, playing) — the page updates its scrubber / counter
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
  }

  setData(sample, silhouette, legend) {
    this.sample = sample; this.silhouette = silhouette; this.legend = legend || [];
    const xs = silhouette.xy;
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (let i = 0; i < xs.length; i += 2) { minX = Math.min(minX, xs[i]); maxX = Math.max(maxX, xs[i]); minY = Math.min(minY, xs[i + 1]); maxY = Math.max(maxY, xs[i + 1]); }
    this.bounds = { minX, maxX: Math.max(maxX, minX + 1e-3), minY, maxY: Math.max(maxY, minY + 1e-3) };
    this.values = new Float32Array(sample.idx.length);
    this.target = new Float32Array(sample.idx.length);
    this.m = sample.idx.length;
    this.trace = null;
    this._draw();
  }

  colorOf(cls) {
    const name = this.legend[cls] || 'unknown';
    return CLASS_COLORS[name] || CLASS_COLORS.unknown;
  }

  /** New final-step activity sample from the network (no trace): animate towards it. */
  setActivity(values) {
    if (!this.sample || !values) return;
    this.trace = null; this.playing = false;
    this.from = Float32Array.from(this.values);
    this.target = normalizeTrace(values); this.tStart = performance.now();
    this._loop();
  }

  /**
   * A full activity trace (steps × m, the sampled neurons after every timestep). The canvas shows
   * the final step until play() / seek(); with autoplay it replays the thought once.
   */
  setTrace(trace, steps, { autoplay = true } = {}) {
    if (!this.sample || !trace || !steps || trace.length !== steps * this.m) { this.setActivity(trace && steps ? trace.subarray(trace.length - this.m) : null); return; }
    this.trace = normalizeTrace(trace); this.steps = steps;
    this.from = null; this.target = null;
    if (autoplay) this.play(0); else this.seek(steps - 1);
  }

  get hasTrace() { return !!this.trace && this.steps > 1; }

  /** Jump to replay position p (fractional timesteps; 0 = after the first step). */
  seek(p) {
    if (!this.trace) return;
    this.playing = false;
    this.pos = Math.max(0, Math.min(this.steps - 1, p));
    traceRow(this.trace, this.m, this.pos, this.values);
    this.onReplay(this.pos, this.steps, false);
    this._draw();
  }

  play(from = null) {
    if (!this.trace) return;
    const start = from !== null ? from : (this.pos >= this.steps - 1 - 1e-6 ? 0 : this.pos);
    this.pos = start; this.playFrom = start; this.playT0 = performance.now(); this.playing = true;
    this.onReplay(this.pos, this.steps, true);
    this._loop();
  }

  pause() { if (this.playing) { this.playing = false; this.onReplay(this.pos, this.steps, false); } }
  toggle() { if (this.playing) this.pause(); else this.play(); }

  setThinking(on) { this.thinking = on; if (on) { this.playing = false; this._loop(); } }

  _resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w; this.h = h;
    this._draw();
  }

  _sprite(cls) {
    if (this.sprites[cls]) return this.sprites[cls];
    const [r, g, b] = this.colorOf(cls);
    const c = document.createElement('canvas'); c.width = c.height = 32;
    const x = c.getContext('2d');
    const grad = x.createRadialGradient(16, 16, 0, 16, 16, 16);
    grad.addColorStop(0, `rgba(${r},${g},${b},1)`); grad.addColorStop(0.25, `rgba(${r},${g},${b},.7)`); grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
    x.fillStyle = grad; x.fillRect(0, 0, 32, 32);
    this.sprites[cls] = c;
    return c;
  }

  _project(x, y) {
    const b = this.bounds, pad = 10;
    const sx = (this.w - 2 * pad) / (b.maxX - b.minX), sy = (this.h - 2 * pad) / (b.maxY - b.minY);
    const s = Math.min(sx, sy);
    const ox = (this.w - s * (b.maxX - b.minX)) / 2, oy = (this.h - s * (b.maxY - b.minY)) / 2;
    return [ox + (x - b.minX) * s, oy + (y - b.minY) * s];
  }

  _loop() {
    if (this.raf) return;
    const step = () => {
      this.raf = 0;
      const t = performance.now();
      let busy = false;
      if (this.playing && this.trace) {
        this.pos = Math.min(this.steps - 1, this.playFrom + (t - this.playT0) / STEP_MS);
        traceRow(this.trace, this.m, this.pos, this.values);
        if (this.pos >= this.steps - 1) this.playing = false; else busy = true;
        this.onReplay(this.pos, this.steps, this.playing);
      } else if (this.from && this.target) {
        const k = Math.min(1, (t - this.tStart) / 500);
        for (let i = 0; i < this.values.length; i++) this.values[i] = this.from[i] + (this.target[i] - this.from[i]) * k;
        if (k < 1) busy = true;
      }
      this.pulse = this.thinking ? 0.5 + 0.5 * Math.sin(t / 130) : Math.max(0, this.pulse - 0.05);
      this._draw(t);
      if (this.thinking || busy || this.pulse > 0) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  _draw(t = performance.now()) {
    const ctx = this.ctx;
    if (!this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.silhouette) return;
    // silhouette: the shape of the brain, dim
    const sil = this.silhouette;
    for (let i = 0; i < sil.cls.length; i++) {
      const [r, g, b] = this.colorOf(sil.cls[i]);
      const [px, py] = this._project(sil.xy[2 * i], sil.xy[2 * i + 1]);
      ctx.fillStyle = `rgba(${r},${g},${b},.13)`;
      ctx.fillRect(px, py, 1.2, 1.2);
    }
    // sampled neurons glowing with activity
    const s = this.sample;
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < s.cls.length; i++) {
      let a = this.values[i];
      if (this.thinking) a = Math.min(1, a * 0.7 + 0.35 * this.pulse * (0.5 + 0.5 * Math.sin(t / 220 + i * 0.37)));
      if (a < 0.03) continue;
      const [px, py] = this._project(s.xy[2 * i], s.xy[2 * i + 1]);
      const size = 3 + 9 * a;
      ctx.globalAlpha = 0.25 + 0.75 * a;
      ctx.drawImage(this._sprite(s.cls[i]), px - size / 2, py - size / 2, size, size);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }
}

// ============================================================================ strip chart
const SHORT_NAMES = { visual_projection: 'vis. projection', visual_centrifugal: 'vis. centrifugal', sensory_ascending: 'sens. ascending' };
/**
 * Mean activity per group across the timesteps, one thin band per group in signal order, each
 * scaled to its own maximum, with a playhead. setTrace(trace, steps, groups); setPos(p).
 */
export class ClassStrip {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.groups = []; this.norm = null; this.means = null; this.steps = 1; this.pos = 0;
    this.w = 0; this.h = 0;
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
  }

  setTrace(trace, steps, groups) {
    this.groups = groups; this.steps = steps;
    const h = Math.max(96, groups.length * 10);          // one legible band per class
    if (this.canvas.style.height !== `${h}px`) this.canvas.style.height = `${h}px`;
    this.means = traceGroupMeans(trace, steps, groups);
    this.norm = normalizeRows(this.means, steps);
    this.pos = steps - 1;
    this._draw();
  }

  clear() { this.norm = null; this._draw(); }
  setPos(p) { this.pos = p; this._draw(); }

  _resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w; this.h = h;
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    if (!this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.norm || !this.groups.length) return;
    const labelW = 104, left = labelW, right = this.w - 6;
    const rows = this.groups.length, rowH = this.h / rows;
    const colW = (right - left) / this.steps;
    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    ctx.textBaseline = 'middle';
    for (let g = 0; g < rows; g++) {
      const [r, gg, b] = CLASS_COLORS[this.groups[g].name] || CLASS_COLORS.unknown;
      const y = g * rowH;
      ctx.textAlign = 'right'; ctx.fillStyle = `rgba(${r},${gg},${b},.9)`;
      ctx.fillText(SHORT_NAMES[this.groups[g].name] || this.groups[g].name.replace(/_/g, ' '), labelW - 6, y + rowH / 2);
      for (let t = 0; t < this.steps; t++) {
        const v = this.norm[g * this.steps + t];
        const hh = Math.max(1, (rowH - 2) * v);
        ctx.fillStyle = `rgba(${r},${gg},${b},${0.15 + 0.85 * v})`;
        ctx.fillRect(left + t * colW + 0.5, y + rowH - 1 - hh, Math.max(1, colW - 1), hh);
      }
    }
    // playhead
    const px = left + (this.pos + 1) * colW;
    ctx.fillStyle = 'rgba(255, 203, 107, .85)';
    ctx.fillRect(px - 0.75, 0, 1.5, this.h);
  }
}
