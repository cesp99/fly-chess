// worker.js — Web Worker hosting the fly brain. Every move sent back was chosen by the network
// (SPEC §9): larva = temperature-1.2 sample of the policy, fly = argmax policy with a 1-ply
// value-head check over the top-3 policy moves, superfly = PUCT MCTS with a backend-dependent budget
// (400 simulations on WebGPU ≈ 1.5–3 s per move on the full brain, 40 in plain JS ≈ 3.5 s; Python uses 200).
//
// Messages in:  {type:'load', baseUrl, gpu?}   gpu:false skips WebGPU (plain-JS engine only)
//               {type:'move', id, fen, moves:[uci...], difficulty:'larva'|'fly'|'superfly', trace?:true}
//               {type:'eval', id, fen, moves:[uci...]}
//               {type:'cancel', id}     abandon the move request `id` (a running superfly search stops
//                                       within a few simulations; a queued one is skipped)
//               {type:'debug', id, op:'lose-gpu'}   tests only: destroy the WebGPU device (→ JS fallback)
// Messages out: {type:'progress', loaded, total, phase, n, nnz, runName}
//               {type:'ready', header, features, sample:{idx, xy, cls}, silhouette:{xy, cls}, legend, retina, fromCache, bytes, backend, gpu}
//                   backend: 'webgpu' | 'js'; gpu: {error} says why WebGPU was not used;
//                   features: {vision, sensoryInput, readoutSteps, neuromod, centralDim, nRet, nCentral, steps} (SPEC §8);
//                   retina (vision only, else null): {n, uv: Float32Array(2n) eye-map coordinates, eye: Uint8Array(n)
//                   0 left / 1 right, type: Uint8Array(n) index into `legend`, legend: ['R1-6', 'R7', 'R8'],
//                   square: Uint8Array(n) board square rank*8+file (mover's perspective), idx: Int32Array(n) blob neuron}
//               {type:'thinking', id, done, total}            (superfly only, every 10 simulations)
//               {type:'move', id, move, san, policyTop, value, activitySample, retinaDrive, trace, thinkMs, sims, stepMs, backend}
//                   retinaDrive: Float32Array(n_ret) — the per-photoreceptor input drive of the position the fly
//                   looked at (what the fly sees; null without vision); trace: only when the request asked for it
//               {type:'move', id, move:null, cancelled:true}  (reply to a cancelled request)
//               {type:'eval', id, value, policyTop, activitySample, retinaDrive, trace, traceSteps, backend}
//                   trace: Float32Array(steps × 2048) — the sampled neurons' activity after every timestep
//                   ([t][j], same neurons as activitySample, whose positions / classes came with 'ready')
//               {type:'error', id?, message}
//
// The activity trace is a mode of the engines' forward pass (one gather per timestep on the GPU,
// read back with everything else): eval always carries it (the page's brain view), a move only on
// request, and the superfly search never asks for it.
//
// Requests are handled strictly one at a time (they share one chess.js instance); `cancel` is the
// only message acted on immediately.
//
// Backends: the worker always builds the plain-JS FlyBrain and, unless asked not to, a FlyBrainGPU
// on the same arrays. Every forward goes through `net` (below): WebGPU when it is alive, else JS;
// a GPU failure (device lost, out of memory, shader error) is reported once and the JS engine
// answers that and every later request — the reply's `backend` field says which one did.

import { Chess } from '../vendor/chess.js';
import * as enc from './encoding.js';
import { loadBrain } from './loader.js';
import { FlyBrain, policyForLegal, sampleIndex, topK } from './flybrain.js';
import { FlyBrainGPU } from './flybrain-gpu.js';
import { runMCTSAsync, uciOf } from './mcts.js';

const SAMPLE_N = 2048;
const SILHOUETTE_N = 6000;
const GPU_SETUP_TIMEOUT_MS = 20000;   // upload of ~55 MB of parameters + shader compilation takes < 1 s on a desktop GPU
const DIFFICULTY = {
  larva: { kind: 'sample', temperature: 1.2 },
  fly: { kind: 'lookahead', topN: 3 },
  // superfly's search budget depends on the engine: ~4 ms per forward on WebGPU vs ~85 ms in plain JS
  superfly: { kind: 'mcts', sims: { webgpu: 400, js: 40 } },
};

function simsFor(cfg) {
  return typeof cfg.sims === 'number' ? cfg.sims : (gpu && !gpu.lost ? cfg.sims.webgpu : cfg.sims.js);
}

let brain = null;             // FlyBrain (plain JS) — always present once loaded
let gpu = null;               // FlyBrainGPU, or null when unavailable / lost
let gpuError = null;          // why the GPU engine is not in use (string)
let sampleIdx = null;         // Int32Array(SAMPLE_N) fixed at load
const chess = new Chess();

function backend() { return gpu ? 'webgpu' : 'js'; }

function dropGpu(err) {
  if (!gpu) return;
  gpuError = String(err && err.message || err);
  try { gpu.destroy(); } catch { /* ignore */ }
  gpu = null;
  self.postMessage({ type: 'backend', backend: 'js', reason: gpuError });
}

/**
 * The network as seen by the move logic and MCTS: forward(x) returns a result synchronously from
 * the JS engine, or a promise from WebGPU. A rejected GPU forward falls back to the JS engine for
 * that call and all later ones.
 */
const net = {
  forward(x, opts) {
    if (!gpu) return brain.forward(x, opts);
    return gpu.forward(x, opts).catch((err) => { dropGpu(err); return brain.forward(x, opts); });
  },
  get lastStepMs() { return gpu ? gpu.lastStepMs : brain.lastStepMs; },
};

/** await-if-needed: keeps the JS path free of extra microtasks */
const settle = (r) => (r && typeof r.then === 'function' ? r : Promise.resolve(r));

let queue = Promise.resolve();          // requests run one after another
let activeSearch = null;                // {id, cancelled} while a superfly search is in flight
const cancelledIds = new Set();         // cancelled requests not yet dispatched

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === 'cancel') {
    if (activeSearch && activeSearch.id === msg.id) activeSearch.cancelled = true;
    else { cancelledIds.add(msg.id); if (cancelledIds.size > 256) cancelledIds.clear(); }
    return;
  }
  queue = queue.then(() => dispatch(msg));
};

async function dispatch(msg) {
  try {
    if (msg.type === 'load') await handleLoad(msg);
    else if (msg.type === 'move') await handleMove(msg);
    else if (msg.type === 'eval') await handleEval(msg);
    else if (msg.type === 'bench') await handleBench(msg);
    else if (msg.type === 'debug' && msg.op === 'lose-gpu') { gpu?.device.destroy(); self.postMessage({ type: 'debug', id: msg.id, backend: backend() }); }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err && err.message || err) });
  }
}

async function handleLoad(msg) {
  // relative URLs are resolved against the page by app.js; resolve against this script otherwise
  const baseUrl = new URL(msg.baseUrl || '../model/', self.location?.href).href;
  const model = await loadBrain(baseUrl, (loaded, total, phase, header) => {
    self.postMessage({ type: 'progress', loaded, total, phase, n: header?.n, nnz: header?.nnz, runName: header?.run_name });
  });
  brain = new FlyBrain(model);
  if (gpu) { try { gpu.destroy(); } catch { /* ignore */ } gpu = null; }
  gpuError = null;
  if (msg.gpu === false) gpuError = 'disabled by request';
  else {
    // a stalled adapter/device request must not hold up 'ready': give WebGPU a bounded time
    let timer = 0;
    const creation = FlyBrainGPU.create(model);
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`WebGPU setup took longer than ${GPU_SETUP_TIMEOUT_MS} ms`)), GPU_SETUP_TIMEOUT_MS); });
    try { gpu = await Promise.race([creation, timeout]); }
    catch (err) {
      gpu = null; gpuError = String(err && err.message || err);
      creation.then((late) => late.destroy(), () => {});   // a late success is released, not used
    } finally { clearTimeout(timer); }
  }
  const n = brain.n;
  sampleIdx = stratifiedSample(n, Math.min(SAMPLE_N, n), brain.superClass, model.arrays.retina_idx, 0x9e3779b9);
  const silIdx = spreadSample(n, Math.min(SILHOUETTE_N, n), 0x85ebca6b);
  const legend = readLegend(model.header);
  const F = brain.features;
  self.postMessage({
    type: 'ready',
    header: model.header,
    fromCache: model.fromCache,
    bytes: model.bytes,
    legend,
    backend: backend(),
    gpu: gpu ? { timestamps: gpu.timestamps } : { error: gpuError },
    features: { vision: F.vision, sensoryInput: F.sensoryInput, readoutSteps: F.readoutSteps, neuromod: F.neuromod, centralDim: F.centralDim, nRet: F.nRet, nCentral: F.nCentral, steps: brain.steps },
    retina: retinaInfo(model),
    sample: { idx: sampleIdx, xy: projectXY(brain, sampleIdx), cls: classesOf(brain, sampleIdx) },
    silhouette: { xy: projectXY(brain, silIdx), cls: classesOf(brain, silIdx) },
  });
}

/** The retina metadata the page draws (eye map + which square every photoreceptor watches); null without vision. */
function retinaInfo({ header, arrays }) {
  if (!brain.vision) return null;
  const n = brain.nRet;
  const u8 = (a) => (a && a.length === n ? Uint8Array.from(a) : new Uint8Array(n));
  return {
    n,
    uv: arrays.retina_uv && arrays.retina_uv.length === 2 * n ? Float32Array.from(arrays.retina_uv) : new Float32Array(2 * n),
    eye: u8(arrays.retina_eye),
    type: u8(arrays.retina_type),
    legend: Array.isArray(header.retina_type_legend) ? header.retina_type_legend : [],
    square: Uint8Array.from(arrays.retina_square),
    idx: Int32Array.from(arrays.retina_idx),
  };
}

const CLASS_QUOTA = 160;   // sampled neurons guaranteed per super class (and for the photoreceptors) when the brain has that many

/**
 * The page's activity sample: k of n neurons, deterministic. A plain random sample of the full brain
 * is 58 % optic lobe and holds ~20 descending neurons, so the thought replay could not show the
 * signal reaching the motor side; every super class (and the retina's photoreceptors) is therefore
 * guaranteed up to CLASS_QUOTA members, the rest of the budget is filled by the spread sample.
 * Without class labels (or when k >= n) this is spreadSample.
 */
function stratifiedSample(n, k, superClass, retinaIdx, seed) {
  if (!superClass || k >= n) return spreadSample(n, k, seed);
  const groups = new Map();
  const ret = retinaIdx && retinaIdx.length ? new Set(retinaIdx) : null;
  for (let i = 0; i < n; i++) {
    const key = ret && ret.has(i) ? -1 : superClass[i];
    let g = groups.get(key); if (!g) groups.set(key, g = []); g.push(i);
  }
  const chosen = new Set();
  let x = seed >>> 0;
  const next = () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x; };
  const keys = [...groups.keys()].sort((a, b) => a - b);
  const quota = Math.min(CLASS_QUOTA, Math.floor(k / keys.length));
  for (const key of keys) {
    const g = groups.get(key);
    const want = Math.min(quota, g.length);
    let got = 0;
    for (let tries = 0; got < want && chosen.size < k && tries < 20 * want; tries++) {
      const i = g[next() % g.length];
      if (!chosen.has(i)) { chosen.add(i); got++; }
    }
  }
  for (const i of spreadSample(n, k, seed ^ 0x5bd1e995)) { if (chosen.size >= k) break; chosen.add(i); }
  return Int32Array.from(chosen).sort();
}

/** Deterministic, well-spread sample of k indices out of n (golden-ratio stride + hash jitter). */
function spreadSample(n, k, seed) {
  const idx = new Int32Array(k);
  const seen = new Set();
  let x = seed >>> 0;
  for (let i = 0; i < k; i++) {
    let v;
    do {
      x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;   // xorshift32
      v = x % n;
    } while (seen.has(v));
    seen.add(v);
    idx[i] = v;
  }
  idx.sort();
  return idx;
}

function projectXY(b, idx) {
  const xy = new Float32Array(idx.length * 2);
  if (!b.positions) { for (let i = 0; i < idx.length; i++) { xy[2 * i] = Math.random(); xy[2 * i + 1] = Math.random(); } return xy; }
  const P = b.positions;
  for (let i = 0; i < idx.length; i++) { xy[2 * i] = P[idx[i] * 3]; xy[2 * i + 1] = P[idx[i] * 3 + 1]; }
  return xy;
}

function classesOf(b, idx) {
  const c = new Uint8Array(idx.length);
  if (b.superClass) for (let i = 0; i < idx.length; i++) c[i] = b.superClass[idx[i]];
  return c;
}

function readLegend(header) {
  const raw = header.super_class_legend || header.super_classes || header.legend || null;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  const out = [];
  for (const [k, v] of Object.entries(raw)) {
    if (Number.isFinite(+k)) out[+k] = v; else if (Number.isFinite(+v)) out[+v] = k;
  }
  return out;
}

/** Put the worker's chess instance into the requested game state (history matters for repetition). */
function setPosition(fen, moves) {
  chess.reset();
  let ok = false;
  if (Array.isArray(moves) && moves.length) {
    ok = true;
    for (const u of moves) {
      try { chess.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] || undefined }); }
      catch { ok = false; break; }
    }
    if (ok && fen && chess.fen() !== fen) ok = false;
  }
  if (!ok) chess.load(fen);
}

function requireBrain() {
  if (!brain) throw new Error('brain not loaded');
}

function legalMoves() {
  const moves = chess.moves({ verbose: true });
  const idx = moves.map((m) => enc.moveToIndex(m, chess));
  return { moves, idx };
}

function policyTopFrom(moves, probs, k = 5) {
  const order = topK(probs, k);
  return order.map((i) => ({ uci: uciOf(moves[i]), san: moves[i].san, p: probs[i] }));
}

function activitySample(activity) {
  const out = new Float32Array(sampleIdx.length);
  for (let i = 0; i < sampleIdx.length; i++) out[i] = activity[sampleIdx[i]];
  return out;
}

async function handleMove(msg) {
  if (cancelledIds.delete(msg.id)) {
    self.postMessage({ type: 'move', id: msg.id, move: null, san: null, cancelled: true, policyTop: [], value: 0, activitySample: null, thinkMs: 0, sims: 0 });
    return;
  }
  requireBrain();
  const t0 = performance.now();
  setPosition(msg.fen, msg.moves);
  const cfg = DIFFICULTY[msg.difficulty] || DIFFICULTY.fly;
  const { moves, idx } = legalMoves();
  if (moves.length === 0) {
    self.postMessage({ type: 'move', id: msg.id, move: null, san: null, policyTop: [], value: 0, activitySample: null, thinkMs: 0, sims: 0 });
    return;
  }
  const x = enc.encodeBoard(chess);
  const res = await settle(net.forward(x, msg.trace ? { trace: sampleIdx } : undefined));
  const probs = policyForLegal(res.policy, idx, 1);
  const act = activitySample(res.activity);
  const retinaDrive = res.retinaDrive ? Float32Array.from(res.retinaDrive) : null;   // the engines reuse the view
  const trace = msg.trace ? res.trace : null;
  let chosen, value = res.value, sims = 0, note = '';

  if (cfg.kind === 'sample') {
    const pT = policyForLegal(res.policy, idx, cfg.temperature);
    chosen = moves[sampleIndex(pT)];
  } else if (cfg.kind === 'lookahead') {
    // top-N policy candidates; play each, ask the brain how the opponent likes it, keep the worst for them
    const cands = topK(probs, Math.min(cfg.topN, moves.length));
    let best = cands[0], bestScore = -Infinity;
    for (const i of cands) {
      chess.move(moves[i]);
      let oppValue;
      if (chess.isCheckmate()) oppValue = -1;
      else if (chess.isDraw() || enc.repetitionCount(chess) >= 3) oppValue = 0;
      else oppValue = (await settle(net.forward(enc.encodeBoard(chess), { activity: false }))).value;
      chess.undo();
      const score = -oppValue;       // our value = negated opponent value
      if (score > bestScore + 1e-9) { bestScore = score; best = i; }
    }
    chosen = moves[best];
    value = bestScore;
    sims = cands.length;
  } else {
    activeSearch = { id: msg.id, cancelled: false };
    let out;
    try {
      out = await runMCTSAsync(net, chess, enc, {
        sims: simsFor(cfg), cPuct: 1.5, dirichletAlpha: 0, temperature: 0, yieldEvery: 10,
        shouldStop: () => activeSearch.cancelled,
        onProgress: (done, total) => { if (done % 10 === 0) self.postMessage({ type: 'thinking', id: msg.id, done, total }); },
      });
    } finally {
      activeSearch = null;
    }
    if (out.stopped) {
      self.postMessage({ type: 'move', id: msg.id, move: null, san: null, cancelled: true, policyTop: [], value: 0, activitySample: null, thinkMs: performance.now() - t0, sims: out.sims });
      return;
    }
    chosen = out.moveObj;
    value = out.rootValue;
    sims = out.sims;
    note = out.visits.slice(0, 5).map((v) => `${v.uci}:${v.n}`).join(' ');
  }

  const thinkMs = performance.now() - t0;
  self.postMessage({
    type: 'move', id: msg.id,
    move: uciOf(chosen), san: chosen.san,
    policyTop: policyTopFrom(moves, probs, 5),
    value, activitySample: act, retinaDrive, trace, traceSteps: brain.steps, thinkMs, sims, stepMs: net.lastStepMs, note, backend: backend(),
  });
}

async function handleEval(msg) {
  requireBrain();
  setPosition(msg.fen, msg.moves);
  const { moves, idx } = legalMoves();
  if (moves.length === 0) {
    self.postMessage({ type: 'eval', id: msg.id, value: 0, policyTop: [], activitySample: null, retinaDrive: null, trace: null, traceSteps: brain.steps, backend: backend() });
    return;
  }
  const res = await settle(net.forward(enc.encodeBoard(chess), { trace: sampleIdx }));
  const probs = policyForLegal(res.policy, idx, 1);
  self.postMessage({
    type: 'eval', id: msg.id, value: res.value,
    policyTop: policyTopFrom(moves, probs, 5), activitySample: activitySample(res.activity),
    retinaDrive: res.retinaDrive ? Float32Array.from(res.retinaDrive) : null,
    trace: res.trace, traceSteps: brain.steps, backend: backend(),
  });
}

/** {type:'bench', reps?, backend?: 'js'|'webgpu', activity?: boolean, trace?: boolean} → per-forward latency of the requested engine (default: the active one). */
async function handleBench(msg) {
  requireBrain();
  chess.reset();
  const x = enc.encodeBoard(chess);
  const reps = msg.reps || 5;
  const useGpu = msg.backend ? msg.backend === 'webgpu' && !!gpu : !!gpu;
  if (msg.backend === 'webgpu' && !gpu) throw new Error(`WebGPU engine not available: ${gpuError}`);
  const opts = { activity: msg.activity !== false, trace: msg.trace ? sampleIdx : null };
  const engine = useGpu ? gpu : brain;
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) await settle(engine.forward(x, opts));
  const ms = (performance.now() - t0) / reps;
  self.postMessage({ type: 'bench', id: msg.id, forwardMs: ms, stepMs: engine.lastStepMs, backend: useGpu ? 'webgpu' : 'js', reps, trace: !!msg.trace });
}
