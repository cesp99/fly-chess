// mcts.js — PUCT Monte-Carlo tree search driven by the fly brain (SPEC §6 stage 2, §9 "superfly").
//
// Mirrors flychess/train/mcts.py: priors = legal-masked softmax of the policy head, leaf value =
// value head from the leaf mover's perspective, negated on every backup step, c_puct = 1.5,
// optional Dirichlet noise at the root, terminal detection via chess.js plus the FEN-based
// repetition count of encoding.js (checkmate = -1 for the mover, any draw = 0). One chess.js instance is reused: moves are played along the selected
// path and undone on the way back.
//
// No chess knowledge lives here except the rules; every evaluation is the network's. Every leaf
// evaluation asks for `{activity: false}` (no hidden-state / retina-drive readback and no activity
// trace): the search only needs the policy and the value, whatever features the blob carries.
//
// The brain may be synchronous (FlyBrain: forward() returns the result) or asynchronous
// (FlyBrainGPU: forward() returns a promise). `step()` returns a boolean for the former and a
// Promise<boolean> for the latter; `runMCTS` needs the synchronous kind, `runMCTSAsync` takes both
// (it only awaits when it is handed a promise, so the plain-JS engine keeps its speed).

import { policyForLegal } from './flybrain.js';

const isThenable = (v) => !!v && typeof v.then === 'function';

class Node {
  constructor(prior) {
    this.prior = prior;
    this.visits = 0;
    this.valueSum = 0;      // from the perspective of the player who moved INTO this node
    this.children = null;   // Array<{move: string(san-ish verbose move obj), idx, node}>
    this.terminal = 0;      // 0 = unknown/not terminal, 1 = terminal (value cached in terminalValue)
    this.terminalValue = 0;
  }
  get q() { return this.visits ? this.valueSum / this.visits : 0; }
}

function dirichlet(k, alpha, rnd) {
  // Gamma(alpha) samples via Marsaglia–Tsang (alpha < 1 boost trick)
  const out = new Float64Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const g = gammaSample(alpha, rnd);
    out[i] = g; sum += g;
  }
  for (let i = 0; i < k; i++) out[i] /= sum || 1;
  return out;
}

function gammaSample(a, rnd) {
  if (a < 1) return gammaSample(a + 1, rnd) * Math.pow(rnd(), 1 / a);
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = normalSample(rnd); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = rnd();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function normalSample(rnd) {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * @typedef {object} MctsOptions
 * @property {number} [sims=200]
 * @property {number} [cPuct=1.5]
 * @property {number} [dirichletAlpha=0]     0 disables root noise
 * @property {number} [dirichletEps=0.25]
 * @property {number} [temperature=0]        0 = pick the most visited move; >0 = sample ∝ visits^(1/T)
 * @property {()=>number} [rnd=Math.random]
 * @property {(done:number, total:number)=>void} [onProgress]
 */

/**
 * @typedef {object} MctsResult
 * @property {string|null} move
 * @property {object|null} moveObj
 * @property {Array<{uci:string, n:number, q:number, p:number}>} visits
 * @property {number} rootValue
 * @property {number} rootPrior
 * @property {number} sims        simulations actually run (less than requested when stopped early)
 */

/**
 * Create a resumable search from the current position of `chess`: `step()` runs one simulation
 * (returns false once the budget is spent or the root is terminal), `result()` picks the move from
 * the visits so far. `runMCTS` drives it synchronously; `runMCTSAsync` yields to the event loop
 * between batches so a Web Worker can receive a cancel message mid-search.
 * Requires a synchronous brain (FlyBrain); use `createSearchAsync` for a brain whose forward()
 * returns a promise.
 * @param {import('./flybrain.js').FlyBrain} brain
 * @param {object} chess  chess.js instance positioned at the root (its history is used for repetition planes)
 * @param {{encodeBoard:Function, legalMoveIndices:Function, moveToIndex:Function, repetitionCount?:Function}} enc  encoding module
 * @param {MctsOptions} [opts]
 * @returns {{sims:number, done:number, step:()=>boolean, result:()=>MctsResult}}
 */
export function createSearch(brain, chess, enc, opts = {}) {
  const search = buildSearch(brain, chess, enc, opts);
  if (isThenable(search.init())) throw new Error('createSearch needs a synchronous brain; use createSearchAsync');
  return search;
}

/**
 * Promise-returning twin of `createSearch` for either kind of brain: the root has been expanded when
 * the promise resolves; `step()` then returns a boolean (sync brain) or a Promise<boolean> (async brain).
 * @returns {Promise<{sims:number, done:number, step:()=>boolean|Promise<boolean>, result:()=>MctsResult}>}
 */
export async function createSearchAsync(brain, chess, enc, opts = {}) {
  const search = buildSearch(brain, chess, enc, opts);
  await search.init();
  return search;
}

function buildSearch(brain, chess, enc, opts) {
  const sims = opts.sims ?? 200;
  const cPuct = opts.cPuct ?? 1.5;
  const rnd = opts.rnd ?? Math.random;
  const root = new Node(1);
  let rootValue = 0;
  let hasChildren = false;
  const path = [];
  const afterRoot = (value) => {
    rootValue = value;
    root.visits = 1; root.valueSum = -rootValue;   // root's stored value is from the parent's (opponent's) view; only Q of children matters
    hasChildren = !!(root.children && root.children.length);
    if (hasChildren && opts.dirichletAlpha > 0 && root.children.length > 1) {
      const noise = dirichlet(root.children.length, opts.dirichletAlpha, rnd);
      const eps = opts.dirichletEps ?? 0.25;
      root.children.forEach((c, i) => { c.node.prior = (1 - eps) * c.node.prior + eps * noise[i]; });
    }
    search.sims = hasChildren ? sims : 0;
  };
  // backup + undo of one simulation once the leaf value is known (sync or after the promise settles)
  const finish = (value, depth) => {
    // --- backup: value is from the leaf mover's perspective; the node was entered by the opponent ---
    let v = -value;
    for (let i = path.length - 1; i >= 0; i--) {
      const p = path[i];
      p.visits++; p.valueSum += v;
      v = -v;
    }
    root.visits++;
    for (let i = 0; i < depth; i++) chess.undo();
    search.done++;
    opts.onProgress?.(search.done, sims);
    return search.done < sims;
  };
  const search = {
    sims: 0,
    done: 0,
    /** expand the root; returns undefined (sync brain) or a promise (async brain) */
    init() {
      const r = expand(root, brain, chess, enc);
      if (isThenable(r)) return r.then(afterRoot);
      afterRoot(r);
      return undefined;
    },
    step() {
      if (!hasChildren || search.done >= sims) return false;
      let node = root;
      path.length = 0;
      let depth = 0;
      // --- select ---
      while (node.children && node.children.length && !node.terminal) {
        const child = selectChild(node, cPuct);
        chess.move(child.moveObj);
        depth++;
        node = child.node;
        path.push(node);
      }
      // --- expand / evaluate ---
      let value;
      if (node.terminal) value = node.terminalValue;
      else value = expand(node, brain, chess, enc);
      if (isThenable(value)) return value.then((v) => finish(v, depth));
      return finish(value, depth);
    },
    result() {
      if (!hasChildren) return { move: null, moveObj: null, visits: [], rootValue, rootPrior: rootValue, sims: 0 };
      const visits = root.children.map((c) => ({ uci: c.uci, n: c.node.visits, q: c.node.q, p: c.node.prior }));
      visits.sort((a, b) => b.n - a.n || b.p - a.p);
      let chosen;
      const T = opts.temperature ?? 0;
      if (T > 0) {
        const ws = visits.map((v) => Math.pow(v.n, 1 / T));
        const total = ws.reduce((a, b) => a + b, 0);
        let r = rnd() * total;
        chosen = visits[visits.length - 1];
        for (let i = 0; i < ws.length; i++) { r -= ws[i]; if (r <= 0) { chosen = visits[i]; break; } }
      } else {
        chosen = visits[0];
      }
      const child = root.children.find((c) => c.uci === chosen.uci);
      // root value estimate = visit-weighted Q of children (from the root mover's perspective)
      let qSum = 0, nSum = 0;
      for (const c of root.children) { qSum += c.node.valueSum; nSum += c.node.visits; }
      const rootQ = nSum ? qSum / nSum : rootValue;
      return { move: chosen.uci, moveObj: child.moveObj, visits, rootValue: rootQ, rootPrior: rootValue, sims: search.done };
    },
  };
  return search;
}

/**
 * Run MCTS to completion from the current position of `chess`.
 * @param {import('./flybrain.js').FlyBrain} brain
 * @param {object} chess
 * @param {{encodeBoard:Function, legalMoveIndices:Function, moveToIndex:Function, repetitionCount?:Function}} enc
 * @param {MctsOptions} [opts]
 * @returns {MctsResult}
 */
export function runMCTS(brain, chess, enc, opts = {}) {
  const search = createSearch(brain, chess, enc, opts);
  while (search.step()) { /* simulate */ }
  return search.result();
}

/**
 * Same as `runMCTS`, but yields to the event loop every `opts.yieldEvery` (default 10) simulations
 * and stops early when `opts.shouldStop()` returns true — the result then reflects the simulations
 * run so far (`result.sims < opts.sims`). Works with a brain whose forward() returns a promise
 * (FlyBrainGPU): each simulation then awaits its leaf evaluation, so `chess` may be at a leaf
 * position while a forward is in flight. Nothing else may touch `chess` while the search is
 * in flight (the board is left at the root between batches).
 * @param {import('./flybrain.js').FlyBrain} brain
 * @param {object} chess
 * @param {{encodeBoard:Function, legalMoveIndices:Function, moveToIndex:Function, repetitionCount?:Function}} enc
 * @param {MctsOptions & {yieldEvery?: number, shouldStop?: () => boolean}} [opts]
 * @returns {Promise<MctsResult & {stopped: boolean}>}
 */
export async function runMCTSAsync(brain, chess, enc, opts = {}) {
  const every = Math.max(1, opts.yieldEvery ?? 10);
  const search = await createSearchAsync(brain, chess, enc, opts);
  let stopped = false;
  for (;;) {
    let more = true;
    for (let i = 0; i < every && more; i++) {
      more = search.step();
      if (isThenable(more)) more = await more;     // async brain: one simulation at a time
    }
    if (!more) break;
    if (opts.shouldStop?.()) { stopped = true; break; }
    await new Promise((r) => setTimeout(r, 0));
    if (opts.shouldStop?.()) { stopped = true; break; }
  }
  return { ...search.result(), stopped };
}

function selectChild(node, cPuct) {
  const sqrtN = Math.sqrt(node.visits);
  let best = null, bestScore = -Infinity;
  for (const c of node.children) {
    const ch = c.node;
    const u = ch.q + cPuct * ch.prior * sqrtN / (1 + ch.visits);
    if (u > bestScore) { bestScore = u; best = c; }
  }
  return best;
}

/**
 * Evaluate `chess` with the brain, create the children; returns the value for the side to move —
 * or a promise of it when the brain's forward() is asynchronous.
 */
function expand(node, brain, chess, enc) {
  if (chess.isCheckmate()) { node.terminal = 1; node.terminalValue = -1; node.children = []; return -1; }
  // repetition via the FEN-based count (python-chess semantics); chess.js' hash counter misses
  // repetitions whose first occurrence followed a double push with an illegal (pinned) ep capture
  const reps = enc.repetitionCount ? enc.repetitionCount(chess) : 0;
  if (chess.isDraw() || chess.isStalemate() || reps >= 3) {
    node.terminal = 1; node.terminalValue = 0; node.children = []; return 0;
  }
  const moves = chess.moves({ verbose: true });
  if (moves.length === 0) { node.terminal = 1; node.terminalValue = 0; node.children = []; return 0; }
  const x = enc.encodeBoard(chess, enc.repetitionCount ? { repeated: reps >= 2 } : {});
  const idx = moves.map((m) => enc.moveToIndex(m, chess));
  const res = brain.forward(x, { activity: false });     // the search never needs the hidden state (FlyBrain ignores the hint)
  const attach = ({ policy, value }) => {
    const probs = policyForLegal(policy, idx, 1);
    node.children = moves.map((m, i) => ({ uci: uciOf(m), moveObj: m, idx: idx[i], node: new Node(probs[i]) }));
    return value;
  };
  return isThenable(res) ? res.then(attach) : attach(res);
}

/** UCI string of a chess.js verbose move. */
export function uciOf(m) {
  return m.from + m.to + (m.promotion || '');
}
