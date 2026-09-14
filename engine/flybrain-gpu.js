// flybrain-gpu.js — the fly brain forward pass on WebGPU (same math as flybrain.js / SPEC §4, §8).
//
//   inj    = w_in·x + b_in                                                          (inject: matvec shader, `sensory_input`)
//   drive  = w_ret[k]·planes[:, square_k] + b_ret[k]                                (retina shader, `vision`)
//   base   = bias + inj[inv_in] + drive[inv_ret]                                    (base shader, once per forward)
//   h_0 = 0 ; for t < steps:  h_{t+1}[i] = (1-α[i])·h_t[i] + α[i]·act(pre[i])       (step shaders, one per row bucket)
//       pre[i] = base[i] + Σ_e w[e]·h_t[col[e]]                                     (plain rows)
//       pre[i] = (Σ_e w[e]·h_t[col[e]]) · (1 + tanh(Σ_e w_mod[e]·h_t[mod_col[e]])) + base[i]   (rows with modulatory inputs, `neuromod`)
//   feat   = [h_t[output_idx] for t in readout_steps] ++ central_w·h_T[central_idx] + central_b   (gather + matvec shaders)
//   policy = policy_w · feat + policy_b                                                (matvec shader)
//   value  = tanh(value_w2 · act_v(value_w · feat + value_b) + value_b2)               (matvec shader + 256-term dot on the CPU)
//
// All recurrent arithmetic is f32 on the GPU; one command buffer per forward encodes the injections,
// all `steps` recurrent updates (ping-pong hidden-state buffers), the readouts and both heads, then a
// single mapAsync reads back the policy logits, the value-head hidden layer and (optionally) the
// retina drive, the whole final state and the activity trace. The blob's neuron order
// (header.neuron_order = 'rcm') is used as-is, like FlyBrain.
//
// Row lengths of the connectome CSR span 0…5000+, so a thread-per-row SpMV is latency-bound on the
// few long rows. Rows are therefore bucketed once by length (LANE_BUCKETS) and each timestep runs
// one dispatch per bucket: 1 lane per short row, 8 lanes per medium row, 64 lanes (a whole
// workgroup) per long row, partial sums combined through workgroup memory. Under `neuromod` every
// bucket runs a gated variant of the kernel: the rows with modulatory inputs are listed first, the
// leading workgroups that hold them also reduce the modulatory sum (a second lane-strided loop +
// tree reduction, skipped uniformly by the other workgroups) and every row is gated with it
// (mod = 0 leaves the ungated rows unchanged) — still three dispatches per timestep, which matters
// more than the arithmetic (separate gated buckets cost +45 % per step on the full brain). A blob
// without the feature runs exactly the pre-feature shaders. The buckets partition the rows, so the
// dispatches write disjoint entries of h_out. The input projection (2048 × 1280), the central
// summary and the heads use the same 64-lane matvec kernel.
//
// A blob whose heads read the final output activity only (no readout_steps / central_dim — e.g.
// fly2) feeds h_T[output_idx] straight into the head matvecs; otherwise the feature vector is
// assembled in a `feat` buffer (gather shader after every readout step, central matvec at the end)
// and the heads read it through an identity index.
//
// Trace mode (`forward(x, {trace: sampleIdx})`): after every timestep a gather shader copies the
// sampled neurons' activity into a trace buffer (steps × |sample|) that is read back with the rest
// in the same mapAsync — no per-step readback. Nothing is encoded when no trace is requested.
//
// `emulate*` are pure-JS, f32-rounded (Math.fround) mirrors of the WGSL, line by line, so the shader
// math is unit-testable under node (no WebGPU there); the browser test compares the real device
// against FlyBrain on the exported model. The only intended deviation between the emulation and the
// GPU is the compiler's freedom to fuse `s + w*h` into an FMA (one rounding instead of two).

import { modelFeatures } from './loader.js';

const WG = 64;                // workgroup size of every kernel
const REDUCE_LANES = 64;      // lanes per matvec row (== WG)
const NONE = 0xffffffff;      // "not an input neuron" marker in the inverse input map
/** SpMV row buckets: rows with length <= maxLen (and above the previous bucket) get `lanes` threads each. */
export const LANE_BUCKETS = [{ lanes: 1, maxLen: 32 }, { lanes: 8, maxLen: 512 }, { lanes: 64, maxLen: Infinity }];

/** Lanes used for a row of `len` synapses (shared by the WGSL generator, the dispatcher and the emulation). */
export function lanesForRow(len) {
  for (const b of LANE_BUCKETS) if (len <= b.maxLen) return b.lanes;
  return LANE_BUCKETS[LANE_BUCKETS.length - 1].lanes;
}

/**
 * Partition rows 0..n-1 into the lane buckets (ascending row ids inside each bucket). With a
 * modulatory CSR (`modIndptr`, neuromod) each bucket lists the rows that have modulatory inputs
 * FIRST (`modRows` of them, ascending), then the others (ascending): the gated step kernel then
 * runs the modulatory reduction only in the leading workgroups that hold such rows.
 */
export function bucketRows(indptr, n, modIndptr = null) {
  const lists = LANE_BUCKETS.map(() => []);
  const modLists = LANE_BUCKETS.map(() => []);
  for (let i = 0; i < n; i++) {
    const len = indptr[i + 1] - indptr[i];
    const k = LANE_BUCKETS.findIndex((b) => len <= b.maxLen);
    if (modIndptr && modIndptr[i + 1] > modIndptr[i]) modLists[k].push(i); else lists[k].push(i);
  }
  return lists.map((rows, k) => ({ lanes: LANE_BUCKETS[k].lanes, modRows: modLists[k].length, rows: Uint32Array.from([...modLists[k], ...rows]) }));
}

// ---------------------------------------------------------------------------------------------
// activation: WGSL source and the f32 JS mirror

/** Resolve the activation kind of a header ('relu'|'tanh'|'gelu'|'gelu_tanh'|'satrelu'). */
export function activationKind(header, name = header.activation ?? 'relu') {
  if (name === 'gelu' && /^tanh$/i.test(String(header.gelu_approximate || ''))) return 'gelu_tanh';
  if (!['relu', 'tanh', 'gelu', 'gelu_tanh', 'satrelu'].includes(name)) throw new Error(`unknown activation ${name}`);
  return name;
}

/** WGSL helpers shared by every kernel: tanh via exp (no reliance on backend tanh polyfills), erf (A&S 7.1.26). */
const WGSL_MATH = `
fn tanh_(x: f32) -> f32 {
  let a = min(abs(x), 20.0);
  let e = exp(2.0 * a);
  let t = 1.0 - 2.0 / (e + 1.0);
  return select(-t, t, x >= 0.0);
}
fn erf_(x: f32) -> f32 {
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return select(-y, y, x >= 0.0);
}
`;

/** WGSL body of `fn NAME(v: f32) -> f32` for an activation kind. */
export function activationWGSL(kind, sat, fnName = 'act') {
  const body = {
    relu: 'return max(v, 0.0);',
    tanh: 'return tanh_(v);',
    gelu: 'return 0.5 * v * (1.0 + erf_(v * 0.7071067811865476));',
    gelu_tanh: 'return 0.5 * v * (1.0 + tanh_(0.7978845608028654 * (v + 0.044715 * v * v * v)));',
    satrelu: `return select(0.0, ${f32lit(sat)} * tanh_(v / ${f32lit(sat)}), v > 0.0);`,
    identity: 'return v;',
  }[kind];
  if (!body) throw new Error(`unknown activation ${kind}`);
  return `fn ${fnName}(v: f32) -> f32 { ${body} }\n`;
}

function f32lit(v) {
  const s = String(Number(v));
  return /[.e]/i.test(s) ? s : s + '.0';
}

const f = Math.fround;
function tanhF32(x) {
  const a = Math.min(Math.abs(x), 20);
  const e = f(Math.exp(f(2 * a)));
  const t = f(1 - f(2 / f(e + 1)));
  return x >= 0 ? t : -t;
}
function erfF32(x) {
  const ax = Math.abs(x);
  const t = f(1 / f(1 + f(0.3275911 * ax)));
  const p = f(f(f(f(f(f(f(f(1.061405429 * t) - 1.453152027) * t) + 1.421413741) * t) - 0.284496736) * t) + 0.254829592);
  const y = f(1 - f(f(p * t) * f(Math.exp(f(-ax * ax)))));
  return x >= 0 ? y : -y;
}

/** f32 JS mirror of activationWGSL(kind). */
export function activationF32(kind, sat = 10) {
  sat = f(sat);
  switch (kind) {
    case 'relu': return (v) => (v > 0 ? v : 0);
    case 'tanh': return (v) => tanhF32(v);
    case 'gelu': return (v) => f(f(0.5 * v) * f(1 + erfF32(f(v * 0.7071067811865476))));
    case 'gelu_tanh': return (v) => f(f(0.5 * v) * f(1 + tanhF32(f(0.7978845608028654 * f(v + f(f(f(0.044715 * v) * v) * v))))));
    case 'satrelu': return (v) => (v > 0 ? f(sat * tanhF32(f(v / sat))) : 0);
    case 'identity': return (v) => v;
    default: throw new Error(`unknown activation ${kind}`);
  }
}

// ---------------------------------------------------------------------------------------------
// shaders

/**
 * Inverse of an index list: inv[neuron] = k or NONE. Throws on duplicates (FlyBrain would sum
 * them; the graph keeps input_idx and retina_idx unique and disjoint).
 */
export function inverseInputMap(inputIdx, n, label = 'input_idx') {
  const inv = new Uint32Array(n).fill(NONE);
  for (let k = 0; k < inputIdx.length; k++) {
    const i = inputIdx[k];
    if (i < 0 || i >= n) throw new Error(`${label}[${k}] = ${i} out of range`);
    if (inv[i] !== NONE) throw new Error(`${label} has duplicate neuron ${i}`);
    inv[i] = k;
  }
  return inv;
}

/** Row index of an invocation for a (possibly 2-D) dispatch — same formula as the WGSL. */
function dispatchDims(rows, maxPerDim) {
  const groups = Math.ceil(rows / WG);
  const gx = Math.min(groups, maxPerDim);
  return [gx, Math.ceil(groups / gx)];
}

/**
 * base[i] = bias[i] + inj[inv_in[i]] + drive[inv_ret[i]] (inj = w_in·x + b_in from the matvec kernel,
 * drive from the retina kernel); each injection is compiled in only when the blob has it.
 */
function baseWGSL(n, { sensory = true, retina = false, packed = false } = {}) {
  return `
@group(0) @binding(0) var<storage, read> bias: array<f32>;
${sensory ? `@group(0) @binding(1) var<storage, read> inv_in: array<u32>;
@group(0) @binding(2) var<storage, read> inj: array<f32>;` : ''}
@group(0) @binding(3) var<storage, read_write> base: array<f32>;
${retina ? `@group(0) @binding(4) var<storage, read> inv_ret: array<u32>;
@group(0) @binding(5) var<storage, read> drive: array<f32>;` : ''}
${packed ? `@group(0) @binding(6) var<storage, read_write> ba: array<f32>;` : ''}
@compute @workgroup_size(${WG})
fn base_(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.y * (nw.x * ${WG}u) + gid.x;
  if (i >= ${n}u) { return; }
  var s = bias[i];
${sensory ? `  let k = inv_in[i];
  if (k != ${NONE}u) { s = s + inj[k]; }` : ''}
${retina ? `  let r = inv_ret[i];
  if (r != ${NONE}u) { s = s + drive[r]; }` : ''}
  base[i] = s;${packed ? '\n  ba[2u * i] = s;' : ''}
}
`;
}

/** drive[k] = b_ret[k] + Σ_p w_ret[k, p] · x[p·64 + square[k]] — one thread per photoreceptor (planes summed in order). */
function retinaWGSL(nRet, planes) {
  return `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> square: array<u32>;
@group(0) @binding(2) var<storage, read> w_ret: array<f32>;
@group(0) @binding(3) var<storage, read> b_ret: array<f32>;
@group(0) @binding(4) var<storage, read_write> drive: array<f32>;
@compute @workgroup_size(${WG})
fn retina(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let k = gid.y * (nw.x * ${WG}u) + gid.x;
  if (k >= ${nRet}u) { return; }
  let sq = square[k];
  var s = b_ret[k];
  for (var p = 0u; p < ${planes}u; p = p + 1u) { s = s + w_ret[k * ${planes}u + p] * x[p * 64u + sq]; }
  drive[k] = s;
}
`;
}

/** dst[params.off + j] = src[idx[j]] for j < params.count — readout features and the activity trace. */
function gatherWGSL() {
  return `
struct Params { off: u32, count: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(${WG})
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let j = gid.y * (nw.x * ${WG}u) + gid.x;
  if (j >= params.count) { return; }
  dst[params.off + j] = src[idx[j]];
}
`;
}

/**
 * One recurrent update for the rows listed in `rows` (a lane bucket): LANES threads per row stride
 * over its synapses, lane partials are tree-reduced in workgroup memory (LANES == 1: plain loop).
 * `modGroups` (neuromod): the gated variant — the bucket's rows with modulatory inputs come first,
 * and the leading `modGroups` workgroups also reduce Σ w_mod·h over the mod CSR with the same lanes
 * (workgroup_id is uniform, so the barriers inside the branch are legal); every row then computes
 * pre = ion · (1 + tanh(mod)) + base (mod = 0 for the ungated rows), whereas the plain kernel folds
 * base into lane 0's partial sum.
 */
function stepWGSL(count, lanes, kind, sat, modGroups = -1) {
  const R = WG / lanes;   // rows per workgroup
  const reduce = (v, indent = '  ') => (lanes === 1 ? '' : `
${indent}part[lid.x] = ${v};
${indent}workgroupBarrier();
${indent}for (var stride = ${lanes / 2}u; stride > 0u; stride = stride >> 1u) {
${indent}  if (lane < stride) { part[lid.x] = part[lid.x] + part[lid.x + stride]; }
${indent}  workgroupBarrier();
${indent}}
${indent}${v} = part[lid.x];`);
  // The gated variant keeps to WebGPU's baseline of 8 storage buffers per stage: base and alpha
  // come interleaved in `ba` ([base_i, alpha_i], written by the base kernel), the row list carries
  // (row, mod_start, mod_end) triples and the modulatory CSR is packed as (col, bitcast(w)) pairs.
  if (modGroups >= 0) {
    return WGSL_MATH + activationWGSL(kind, sat, 'act') + `
@group(0) @binding(0) var<storage, read> indptr: array<u32>;
@group(0) @binding(1) var<storage, read> col: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> ba: array<f32>;
@group(0) @binding(4) var<storage, read> h_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> h_out: array<f32>;
@group(0) @binding(6) var<storage, read> rows3: array<u32>;
@group(0) @binding(7) var<storage, read> modcw: array<u32>;
${lanes === 1 ? '' : `var<workgroup> part: array<f32, ${WG}>;`}
@compute @workgroup_size(${WG})
fn step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let group = wg.y * nw.x + wg.x;
  let slot = group * ${R}u + lid.x / ${lanes}u;
  let lane = lid.x % ${lanes}u;
  let valid = slot < ${count}u;
  var i = 0u;
  var start = 0u;
  var end = 0u;
  var s = 0.0;
  if (valid) { i = rows3[3u * slot]; start = indptr[i]; end = indptr[i + 1u]; }
  for (var e = start + lane; e < end; e = e + ${lanes}u) { s = s + w[e] * h_in[col[e]]; }${reduce('s')}
  var m = 0.0;
  if (group < ${modGroups}u) {
    var mstart = 0u;
    var mend = 0u;
    if (valid) { mstart = rows3[3u * slot + 1u]; mend = rows3[3u * slot + 2u]; }
    for (var e = mstart + lane; e < mend; e = e + ${lanes}u) { m = m + bitcast<f32>(modcw[2u * e + 1u]) * h_in[modcw[2u * e]]; }${reduce('m', '    ')}
  }
  if (valid && lane == 0u) {
    s = s * (1.0 + tanh_(m)) + ba[2u * i];
    let a = ba[2u * i + 1u];
    h_out[i] = (1.0 - a) * h_in[i] + a * act(s);
  }
}
`;
  }
  return WGSL_MATH + activationWGSL(kind, sat, 'act') + `
@group(0) @binding(0) var<storage, read> indptr: array<u32>;
@group(0) @binding(1) var<storage, read> col: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<storage, read> alpha: array<f32>;
@group(0) @binding(5) var<storage, read> h_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> h_out: array<f32>;
@group(0) @binding(7) var<storage, read> rows: array<u32>;
${lanes === 1 ? '' : `var<workgroup> part: array<f32, ${WG}>;`}
@compute @workgroup_size(${WG})
fn step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let slot = (wg.y * nw.x + wg.x) * ${R}u + lid.x / ${lanes}u;
  let lane = lid.x % ${lanes}u;
  let valid = slot < ${count}u;
  var i = 0u;
  var start = 0u;
  var end = 0u;
  var s = 0.0;
  if (valid) { i = rows[slot]; start = indptr[i]; end = indptr[i + 1u]; s = select(0.0, base[i], lane == 0u); }
  for (var e = start + lane; e < end; e = e + ${lanes}u) { s = s + w[e] * h_in[col[e]]; }${reduce('s')}
  if (valid && lane == 0u) {
    let a = alpha[i];
    h_out[i] = (1.0 - a) * h_in[i] + a * act(s);
  }
}
`;
}

/**
 * y[off + m] = ACT(b[m] + Σ_j W[m, j] · h[idx[j]]) — one workgroup per row, 64-lane strided partial
 * sums + tree reduction. `off` places the rows inside a larger output (the central summary writes
 * behind the readout features).
 */
function matvecWGSL(nOut, kind, sat, off = 0) {
  return WGSL_MATH + activationWGSL(kind, sat, 'act') + `
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(1) var<storage, read> out_idx: array<u32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, ${REDUCE_LANES}>;
@compute @workgroup_size(${REDUCE_LANES})
fn matvec(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.y * nw.x + wg.x;
  let off = m * ${nOut}u;
  var s = 0.0;
  for (var j = lid.x; j < ${nOut}u; j = j + ${REDUCE_LANES}u) { s = s + W[off + j] * h[out_idx[j]]; }
  part[lid.x] = s;
  workgroupBarrier();
  for (var stride = ${REDUCE_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) { part[lid.x] = part[lid.x] + part[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { y[${off ? `m + ${off}u` : 'm'}] = act(part[0] + b[m]); }
}
`;
}

// ---------------------------------------------------------------------------------------------
// pure-JS f32 emulation of the kernels (for tests; mirrors the WGSL above line by line)

/** Model constants shared by the GPU engine and its emulation (including the SPEC §8 feature switches). */
export function modelShape({ header, arrays }) {
  const n = header.n ?? arrays.csr_indptr.length - 1;
  const D = header.input_dim ?? (header.num_planes ?? 20) * 64;
  const nOut = arrays.output_idx.length;
  const numMoves = header.num_moves ?? arrays.policy_b.length;
  const features = modelFeatures(header, arrays);
  const headIn = features.headIn;
  const valueHidden = arrays.value_w2 ? arrays.value_w.length / headIn : 0;
  const kind = activationKind(header);
  const valueKind = activationKind(header, header.value_activation || header.activation || 'relu');
  const sat = Number(header.activation_sat ?? 10);
  const steps = header.steps ?? 8;
  let alpha = arrays.alpha;
  if (alpha.length !== n) alpha = new Float32Array(n).fill(alpha[0]);
  if (arrays.w.length !== arrays.csr_indices.length || arrays.csr_indptr.length !== n + 1) throw new Error('CSR shape mismatch');
  if (features.sensoryInput && arrays.w_in.length !== arrays.input_idx.length * D) throw new Error('w_in shape mismatch');
  if (!features.sensoryInput && !features.vision) throw new Error('the brain has no input path (neither sensory_input nor vision)');
  if (arrays.policy_w.length !== numMoves * headIn) throw new Error('policy_w shape mismatch');
  if (arrays.value_w.length % headIn !== 0) throw new Error('value_w shape mismatch');
  // the heads read h_T[output_idx] directly unless the feature vector is assembled (readout steps / central summary)
  const direct = features.readoutSteps.length === 1 && features.readoutSteps[0] === steps && features.centralDim === 0;
  return { n, D, nOut, numMoves, valueHidden, kind, valueKind, sat, steps, alpha, features, headIn, direct };
}

/** retina kernel: drive[k] = b_ret[k] + Σ_p w_ret[k,p]·x[p·64 + square[k]] (planes in order, f32). */
export function emulateRetina({ arrays }, x, drive, planes = 20) {
  const nRet = drive.length, w = arrays.w_ret, b = arrays.b_ret, sq = arrays.retina_square;
  for (let k = 0; k < nRet; k++) {
    let s = b[k];
    for (let p = 0; p < planes; p++) s = f(s + f(w[k * planes + p] * x[p * 64 + sq[k]]));
    drive[k] = s;
  }
  return drive;
}

/**
 * inject: inj = w_in·x + b_in through the matvec kernel (when the blob has the sensory path), then
 * base[i] = bias[i] + inj[inv_in[i]] + drive[inv_ret[i]] (base kernel; `invRet`/`drive` only with vision).
 */
export function emulateInject({ arrays }, invIn, x, base, invRet = null, drive = null) {
  const n = base.length, D = x.length, nIn = arrays.w_in ? arrays.w_in.length / D : 0;
  const ident = Uint32Array.from({ length: D }, (_, j) => j);
  const inj = emulateMatvec(arrays.w_in, arrays.b_in, nIn, D, x, ident, (v) => v, new Float32Array(nIn));
  for (let i = 0; i < n; i++) {
    let s = arrays.bias[i];
    const k = invIn[i];
    if (k !== NONE) s = f(s + inj[k]);
    if (invRet) { const r = invRet[i]; if (r !== NONE) s = f(s + drive[r]); }
    base[i] = s;
  }
  return base;
}

/**
 * step kernels: hOut[i] = (1-α)·hIn[i] + α·act(pre) with each row's lane-strided partial sums and
 * tree reduction; pre = base + Σ w·hIn[col] (plain kernel) or, under neuromod for every row,
 * (Σ w·hIn[col])·(1 + tanh(Σ w_mod·hIn[mod_col])) + base (gated kernel), exactly as the kernels compute it.
 */
export function emulateStep({ header, arrays }, alpha, act, base, hIn, hOut) {
  const n = hOut.length, indptr = arrays.csr_indptr, col = arrays.csr_indices, w = arrays.w;
  const neuromod = !!header?.neuromod && !!arrays.mod_indices && arrays.mod_indices.length > 0;
  const mIndptr = arrays.mod_indptr, mCol = arrays.mod_indices, wMod = arrays.w_mod;
  const part = new Float32Array(WG), mpart = new Float32Array(WG);
  const reduce = (buf, lanes) => { for (let stride = lanes >> 1; stride > 0; stride >>= 1) for (let l = 0; l < stride; l++) buf[l] = f(buf[l] + buf[l + stride]); return buf[0]; };
  for (let i = 0; i < n; i++) {
    const start = indptr[i], end = indptr[i + 1];
    const lanes = lanesForRow(end - start);
    for (let lane = 0; lane < lanes; lane++) {
      let s = !neuromod && lane === 0 ? base[i] : 0;
      for (let e = start + lane; e < end; e += lanes) s = f(s + f(w[e] * hIn[col[e]]));
      part[lane] = s;
    }
    let s = reduce(part, lanes);
    if (neuromod) {   // gated kernel: every row, m = 0 (and the reduction skipped) for rows without modulatory inputs
      const ms = mIndptr[i], me = mIndptr[i + 1];
      let m = 0;
      if (me > ms) {
        for (let lane = 0; lane < lanes; lane++) {
          let acc = 0;
          for (let e = ms + lane; e < me; e += lanes) acc = f(acc + f(wMod[e] * hIn[mCol[e]]));
          mpart[lane] = acc;
        }
        m = reduce(mpart, lanes);
      }
      s = f(f(s * f(1 + tanhF32(m))) + base[i]);
    }
    const a = alpha[i];
    hOut[i] = f(f(f(1 - a) * hIn[i]) + f(a * act(s)));
  }
  return hOut;
}

/** gather kernel: dst[off + j] = src[idx[j]] for j < count. */
export function emulateGather(src, idx, count, off, dst) {
  for (let j = 0; j < count; j++) dst[off + j] = src[idx[j]];
  return dst;
}

/** matvec kernel: y[m] = act(b[m] + Σ_j W[m,j]·h[outIdx[j]]) with the workgroup's summation order. */
export function emulateMatvec(W, b, rows, nOut, h, outIdx, act, y) {
  const part = new Float32Array(REDUCE_LANES);
  for (let m = 0; m < rows; m++) {
    const off = m * nOut;
    for (let l = 0; l < REDUCE_LANES; l++) {
      let s = 0;
      for (let j = l; j < nOut; j += REDUCE_LANES) s = f(s + f(W[off + j] * h[outIdx[j]]));
      part[l] = s;
    }
    for (let stride = REDUCE_LANES >> 1; stride > 0; stride >>= 1) for (let l = 0; l < stride; l++) part[l] = f(part[l] + part[l + stride]);
    y[m] = act(f(part[0] + b[m]));
  }
  return y;
}

/** Final scalar of the value head from the matvec output (MLP: 256-term dot; linear: y[0]); done on the CPU in both engines. */
export function valueFromHidden(arrays, valueHidden, y) {
  let v;
  if (valueHidden > 0) {
    v = arrays.value_b2[0];
    for (let k = 0; k < valueHidden; k++) v += arrays.value_w2[k] * y[k];
  } else v = y[0];
  return Math.tanh(v);
}

const identityIdx = (len) => Uint32Array.from({ length: len }, (_, j) => j);

/**
 * Whole forward pass through the emulated kernels: same outputs as FlyBrainGPU.forward
 * (`opts.trace` = sampled neuron indices → `trace` Float32Array(steps × |trace|)).
 */
export function emulateForward(model, x, opts = {}) {
  const S = modelShape(model);
  const F = S.features;
  const a = model.arrays;
  const invIn = F.sensoryInput ? inverseInputMap(a.input_idx, S.n) : new Uint32Array(S.n).fill(NONE);
  const invRet = F.vision ? inverseInputMap(a.retina_idx, S.n, 'retina_idx') : null;
  const drive = F.vision ? emulateRetina(model, x, new Float32Array(F.nRet), F.numPlanes) : null;
  const base = emulateInject(model, invIn, x, new Float32Array(S.n), invRet, drive);
  const act = activationF32(S.kind, S.sat);
  const traceIdx = opts.trace || null;
  const L = traceIdx ? traceIdx.length : 0;
  const trace = traceIdx ? new Float32Array(S.steps * L) : null;
  const feat = new Float32Array(S.headIn);
  let h = new Float32Array(S.n), h2 = new Float32Array(S.n);
  for (let t = 0; t < S.steps; t++) {
    emulateStep(model, S.alpha, act, base, h, h2);
    const tmp = h; h = h2; h2 = tmp;
    const r = F.readoutSteps.indexOf(t + 1);
    if (!S.direct && r >= 0) emulateGather(h, a.output_idx, S.nOut, r * S.nOut, feat);
    if (trace) emulateGather(h, traceIdx, L, t * L, trace);
  }
  const ident = activationF32('identity');
  if (F.centralDim > 0) {
    emulateMatvec(a.central_w, a.central_b, F.centralDim, F.nCentral, h, a.central_idx, ident, feat.subarray(S.nOut * F.readoutSteps.length));
  }
  const [src, idx, K] = S.direct ? [h, a.output_idx, S.nOut] : [feat, identityIdx(S.headIn), S.headIn];
  const policy = emulateMatvec(a.policy_w, a.policy_b, S.numMoves, K, src, idx, ident, new Float32Array(S.numMoves));
  const vRows = S.valueHidden || 1;
  const vAct = S.valueHidden ? activationF32(S.valueKind, S.sat) : ident;
  const hid = emulateMatvec(a.value_w, a.value_b, vRows, K, src, idx, vAct, new Float32Array(vRows));
  return { policy, value: valueFromHidden(a, S.valueHidden, hid), activity: h, retinaDrive: drive, trace };
}

// ---------------------------------------------------------------------------------------------
// the WebGPU engine

const TS_BYTES = 16;   // two u64 timestamps at the head of the readback buffer

export class FlyBrainGPU {
  /**
   * Build the engine on the default WebGPU adapter. Rejects when WebGPU is unavailable, the adapter
   * limits cannot hold the model, or the model uses a feature the kernels do not cover.
   * @param {{header: object, arrays: Record<string, Int32Array|Float32Array|Uint8Array>}} model  as returned by loadBrain / parseArrays
   * @param {{adapter?: GPUAdapter, device?: GPUDevice, powerPreference?: string}} [opts]
   */
  static async create(model, opts = {}) {
    const gpu = (typeof navigator !== 'undefined' && navigator.gpu) || null;
    if (!gpu && !opts.device) throw new Error('WebGPU is not available in this context');
    let device = opts.device || null, adapter = opts.adapter || null;
    if (!device) {
      // Chromium can answer null while its GPU process is still starting: retry a few times
      for (let attempt = 0; !adapter && attempt < 4; attempt++) {
        if (attempt) await new Promise((r) => setTimeout(r, 200 * attempt));
        adapter = await gpu.requestAdapter({ powerPreference: opts.powerPreference || 'high-performance' });
      }
      if (!adapter) throw new Error('no WebGPU adapter');
      const S = modelShape(model);
      const largest = 4 * Math.max(S.n + 1, model.arrays.csr_indices.length, model.arrays.w_in.length, model.arrays.policy_w.length);
      const lim = adapter.limits;
      if (largest > lim.maxStorageBufferBindingSize || largest > lim.maxBufferSize) {
        throw new Error(`model needs ${largest} B storage bindings, adapter allows ${Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize)}`);
      }
      const requiredLimits = {};
      // default binding limit is 128 MiB; ask for more only when the model needs it
      if (largest > 134217728) { requiredLimits.maxStorageBufferBindingSize = largest; requiredLimits.maxBufferSize = Math.max(largest, 268435456); }
      const requiredFeatures = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
      device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    }
    const brain = new FlyBrainGPU(model, device, adapter);
    try {
      await brain._build();
    } catch (err) {
      brain.destroy();
      throw err;
    }
    return brain;
  }

  /** @private use FlyBrainGPU.create */
  constructor(model, device, adapter) {
    const S = modelShape(model);
    Object.assign(this, S);
    this.header = model.header;
    this.arrays = model.arrays;
    this.device = device;
    this.adapter = adapter;
    this.positions = model.arrays.positions ?? null;
    this.superClass = model.arrays.super_class ?? null;
    this.inputDim = S.D;
    this.nnz = model.arrays.csr_indices.length;
    this.nIn = model.arrays.input_idx.length;
    this.activation = S.kind;
    const F = S.features;
    this.vision = F.vision; this.sensoryInput = F.sensoryInput; this.readoutSteps = F.readoutSteps;
    this.neuromod = F.neuromod; this.centralDim = F.centralDim; this.nRet = F.nRet; this.nCentral = F.nCentral;
    this.lost = null;             // string once the device is gone / errored — forward() rejects afterwards
    this.lastStepMs = 0;          // GPU time per recurrent step (timestamp queries) or wall-clock estimate
    this.lastForwardMs = 0;
    this.timestamps = device.features.has('timestamp-query');
    this._chain = Promise.resolve();
    this._bufs = [];
    this._trace = null;           // {idx, L, buf, bgs, bufs} once a trace sample is set
    device.lost.then((info) => { this.lost = `device lost: ${info.message || info.reason}`; }).catch(() => {});
    device.addEventListener?.('uncapturederror', (ev) => { this.lost = `gpu error: ${ev.error?.message || ev.error}`; });
  }

  get backend() { return 'webgpu'; }

  _buffer(label, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const size = Math.ceil(Math.max(bytes.byteLength, 4) / 4) * 4;
    const buf = this.device.createBuffer({ label, size, usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(bytes);
    buf.unmap();
    this._bufs.push(buf);
    return buf;
  }

  _empty(label, bytes, usage) {
    const buf = this.device.createBuffer({ label, size: Math.ceil(Math.max(bytes, 4) / 4) * 4, usage });
    this._bufs.push(buf);
    return buf;
  }

  /** Uniform {off, count} for the gather kernel. */
  _params(label, off, count) {
    return this._buffer(label, Uint32Array.from([off, count]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  /** Bind group from [binding, buffer] pairs (or a plain buffer list bound 0..k). */
  _bg(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((e, k) => (Array.isArray(e) ? { binding: e[0], resource: { buffer: e[1] } } : { binding: k, resource: { buffer: e } })),
    });
  }

  async _build() {
    const dev = this.device, a = this.arrays, S = this, F = this.features;
    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    // parameters (uploaded once)
    const bIndptr = this._buffer('csr_indptr', a.csr_indptr);
    const bCol = this._buffer('csr_indices', a.csr_indices);
    const bW = this._buffer('w', a.w);
    const bBias = this._buffer('bias', a.bias);
    const bAlpha = this._buffer('alpha', S.alpha);
    const bIdent = this._buffer('identity_idx', identityIdx(Math.max(S.D, S.headIn)));
    const bOutIdx = this._buffer('output_idx', a.output_idx);
    const bPolicyW = this._buffer('policy_w', a.policy_w);
    const bPolicyB = this._buffer('policy_b', a.policy_b);
    const bValueW = this._buffer('value_w', a.value_w);
    const bValueB = this._buffer('value_b', a.value_b);
    let bInvIn = null, bWIn = null, bBIn = null;
    if (F.sensoryInput) {
      bInvIn = this._buffer('inv_in', inverseInputMap(a.input_idx, S.n));
      bWIn = this._buffer('w_in', a.w_in);
      bBIn = this._buffer('b_in', a.b_in);
    }
    let bInvRet = null, bSquare = null, bWRet = null, bBRet = null;
    if (F.vision) {
      bInvRet = this._buffer('inv_ret', inverseInputMap(a.retina_idx, S.n, 'retina_idx'));
      bSquare = this._buffer('retina_square', Uint32Array.from(a.retina_square));
      bWRet = this._buffer('w_ret', a.w_ret);
      bBRet = this._buffer('b_ret', a.b_ret);
    }
    let bModCW = null, bBA = null;
    if (F.neuromod) {
      // (col, bitcast(w)) pairs of the modulatory CSR and the interleaved [base, alpha] pairs (base filled per forward)
      const cw = new Uint32Array(2 * a.mod_indices.length);
      const wBits = new Uint32Array(a.w_mod.buffer, a.w_mod.byteOffset, a.w_mod.length);
      for (let e = 0; e < a.mod_indices.length; e++) { cw[2 * e] = a.mod_indices[e]; cw[2 * e + 1] = wBits[e]; }
      bModCW = this._buffer('mod_col_w', cw);
      const ba = new Float32Array(2 * S.n);
      for (let i = 0; i < S.n; i++) ba[2 * i + 1] = S.alpha[i];
      bBA = this._buffer('base_alpha', ba);
    }
    const buckets = bucketRows(a.csr_indptr, S.n, F.neuromod ? a.mod_indptr : null).filter((b) => b.rows.length > 0);
    const rowsBuffer = (b) => {
      if (!F.neuromod) return this._buffer(`rows_${b.lanes}`, b.rows);
      const triples = new Uint32Array(3 * b.rows.length);
      for (let k = 0; k < b.rows.length; k++) { const i = b.rows[k]; triples[3 * k] = i; triples[3 * k + 1] = a.mod_indptr[i]; triples[3 * k + 2] = a.mod_indptr[i + 1]; }
      return this._buffer(`rows3_${b.lanes}`, triples);
    };
    // gated kernel (neuromod): the modulatory reduction runs in the leading workgroups that hold gated rows
    this.buckets = buckets.map((b) => ({ lanes: b.lanes, count: b.rows.length, modGroups: F.neuromod ? Math.ceil(b.modRows / (WG / b.lanes)) : -1, buf: rowsBuffer(b) }));
    // per-forward state
    this.bX = this._empty('x', 4 * S.D, ST);
    const bInj = F.sensoryInput ? this._empty('inj', 4 * this.nIn, GPUBufferUsage.STORAGE) : null;
    this.bDrive = F.vision ? this._empty('retina_drive', 4 * F.nRet, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
    const bBase = this._empty('base', 4 * S.n, ST);
    const hA = this._empty('h_a', 4 * S.n, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const hB = this._empty('h_b', 4 * S.n, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.hA = hA; this.hB = hB;
    this.hFinal = S.steps % 2 === 0 ? hA : hB;
    this.hAfter = (t) => (t % 2 === 0 ? hB : hA);      // buffer holding h_{t+1} after step t
    this.bFeat = S.direct ? null : this._empty('feat', 4 * S.headIn, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.vRows = S.valueHidden || 1;
    this.bPolicy = this._empty('policy', 4 * S.numMoves, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.bValue = this._empty('value_hidden', 4 * this.vRows, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    // readback: [timestamps | policy | value hidden | retina drive | h | trace]
    this.offPolicy = TS_BYTES;
    this.offValue = this.offPolicy + 4 * S.numMoves;
    this.offRet = this.offValue + 4 * this.vRows;
    this.offH = this.offRet + 4 * F.nRet;
    this.offTrace = this.offH + 4 * S.n;
    this.readback = null;
    this._readbackBytes = 0;
    this._ensureReadback(this.offTrace);
    if (this.timestamps) {
      this.querySet = dev.createQuerySet({ type: 'timestamp', count: 2 });
      this.bQuery = this._empty('timestamps', 256, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
    }

    // pipelines (compiled asynchronously; compilation errors surface here)
    const make = async (label, code, entry) => {
      const module = dev.createShaderModule({ label, code });
      const info = await module.getCompilationInfo?.();
      const errs = (info?.messages || []).filter((m) => m.type === 'error');
      if (errs.length) throw new Error(`${label} shader: ${errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
      return dev.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: entry } });
    };
    const vKind = S.valueHidden ? S.valueKind : 'identity';
    const K = S.direct ? S.nOut : S.headIn;      // width of the head input
    const featOff = S.nOut * F.readoutSteps.length;
    const [pBase, pMatvecIn, pPolicy, pValue, pRetina, pGather, pCentral, ...pSteps] = await Promise.all([
      make('base', baseWGSL(S.n, { sensory: F.sensoryInput, retina: F.vision, packed: F.neuromod }), 'base_'),
      F.sensoryInput ? make('inject', matvecWGSL(S.D, 'identity', S.sat), 'matvec') : null,
      make('policy', matvecWGSL(K, 'identity', S.sat), 'matvec'),
      make('value', matvecWGSL(K, vKind, S.sat), 'matvec'),
      F.vision ? make('retina', retinaWGSL(F.nRet, F.numPlanes), 'retina') : null,
      make('gather', gatherWGSL(), 'gather'),
      F.centralDim > 0 ? make('central', matvecWGSL(F.nCentral, 'identity', S.sat, featOff), 'matvec') : null,
      ...this.buckets.map((b) => make(`step_${b.lanes}${b.modGroups >= 0 ? '_gated' : ''}`, stepWGSL(b.count, b.lanes, S.kind, S.sat, b.modGroups), 'step')),
    ]);
    this.pBase = pBase; this.pInject = pMatvecIn; this.pPolicy = pPolicy; this.pValue = pValue;
    this.pRetina = pRetina; this.pGather = pGather; this.pCentral = pCentral;
    const bg = (pipeline, buffers) => this._bg(pipeline, buffers);
    const maxWG = dev.limits.maxComputeWorkgroupsPerDimension;
    const dims2 = (groups) => { const gx = Math.min(groups, maxWG); return [gx, Math.ceil(groups / gx)]; };
    this._dims2 = dims2;
    if (F.sensoryInput) this.bgInject = bg(pMatvecIn, [this.bX, bIdent, bWIn, bBIn, bInj]);
    if (F.vision) this.bgRetina = bg(pRetina, [this.bX, bSquare, bWRet, bBRet, this.bDrive]);
    const baseEntries = [[0, bBias], [3, bBase]];
    if (F.sensoryInput) baseEntries.push([1, bInvIn], [2, bInj]);
    if (F.vision) baseEntries.push([4, bInvRet], [5, this.bDrive]);
    if (F.neuromod) baseEntries.push([6, bBA]);
    this.bgBase = bg(pBase, baseEntries);
    this.buckets.forEach((b, k) => {
      b.pipeline = pSteps[k];
      b.bg = b.modGroups >= 0 ? [
        bg(pSteps[k], [bIndptr, bCol, bW, bBA, hA, hB, b.buf, bModCW]),   // even t: read A, write B
        bg(pSteps[k], [bIndptr, bCol, bW, bBA, hB, hA, b.buf, bModCW]),   // odd t
      ] : [
        bg(pSteps[k], [bIndptr, bCol, bW, bBase, bAlpha, hA, hB, b.buf]),
        bg(pSteps[k], [bIndptr, bCol, bW, bBase, bAlpha, hB, hA, b.buf]),
      ];
      b.dims = dims2(Math.ceil(b.count / (WG / b.lanes)));
    });
    // readout gathers (one per readout step) and the central summary write the feature vector
    this.readouts = [];
    if (!S.direct) {
      F.readoutSteps.forEach((step, r) => {
        const t = step - 1;
        this.readouts.push({ t, bg: bg(pGather, [this.hAfter(t), bOutIdx, this.bFeat, this._params(`readout_${r}`, r * S.nOut, S.nOut)]), dims: dims2(Math.ceil(S.nOut / WG)) });
      });
      if (F.centralDim > 0) {
        this.bgCentral = bg(pCentral, [this.hFinal, this._buffer('central_idx', a.central_idx), this._buffer('central_w', a.central_w), this._buffer('central_b', a.central_b), this.bFeat]);
        this.dimsCentral = dims2(F.centralDim);
      }
    }
    const [hSrc, hIdx] = S.direct ? [this.hFinal, bOutIdx] : [this.bFeat, bIdent];
    this.bgPolicy = bg(pPolicy, [hSrc, hIdx, bPolicyW, bPolicyB, this.bPolicy]);
    this.bgValue = bg(pValue, [hSrc, hIdx, bValueW, bValueB, this.bValue]);
    this.dimsN = dispatchDims(S.n, maxWG);
    this.dimsInject = dims2(this.nIn);
    this.dimsRetina = dims2(Math.ceil(F.nRet / WG));
    this.dimsPolicy = dims2(S.numMoves);
    this.dimsValue = dims2(this.vRows);
    // run one forward so pipeline warm-up does not land on the first move
    await this._forward(new Float32Array(S.D), { activity: false });
  }

  /** (Re)allocate the readback buffer to hold at least `bytes`. */
  _ensureReadback(bytes) {
    if (this.readback && this._readbackBytes >= bytes) return;
    if (this.readback) { try { this.readback.destroy(); } catch { /* ignore */ } this._bufs = this._bufs.filter((b) => b !== this.readback); }
    this._readbackBytes = bytes;
    this.readback = this._empty('readback', bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  }

  /**
   * Prepare the trace buffers for a set of sampled neurons (idempotent for an unchanged sample):
   * one gather bind group per timestep, writing h_{t+1}[idx] to trace[t * L ..].
   */
  _ensureTrace(idx) {
    const cur = this._trace;
    if (cur && cur.L === idx.length) {
      let same = cur.idx === idx;
      if (!same) { same = true; for (let j = 0; j < idx.length; j++) if (cur.idx[j] !== idx[j]) { same = false; break; } }
      if (same) return cur;
    }
    if (cur) { for (const b of cur.bufs) { try { b.destroy(); } catch { /* ignore */ } } this._bufs = this._bufs.filter((b) => !cur.bufs.includes(b)); }
    const L = idx.length;
    for (let j = 0; j < L; j++) if (!(idx[j] >= 0 && idx[j] < this.n)) throw new Error(`trace index ${idx[j]} out of range`);
    const before = this._bufs.length;
    const bIdx = this._buffer('trace_idx', Uint32Array.from(idx));
    const bTrace = this._empty('trace', 4 * this.steps * L, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const bgs = [], dims = this._dims2(Math.ceil(L / WG));
    for (let t = 0; t < this.steps; t++) bgs.push(this._bg(this.pGather, [this.hAfter(t), bIdx, bTrace, this._params(`trace_${t}`, t * L, L)]));
    this._trace = { idx: Int32Array.from(idx), L, buf: bTrace, bgs, dims, bufs: this._bufs.slice(before) };
    this._ensureReadback(this.offTrace + 4 * this.steps * L);
    return this._trace;
  }

  /**
   * One forward pass (serialised: concurrent calls run one after another on the same buffers).
   * @param {Float32Array} x  flattened planes (input_dim)
   * @param {{activity?: boolean, trace?: Int32Array|Uint32Array|number[]|null}} [opts]
   *   activity: false skips the final-state (and retina drive) readback — the search path;
   *   trace: neuron indices sampled after every timestep → `trace` Float32Array(steps × |trace|), [t][j].
   * @returns {Promise<{policy: Float32Array, value: number, activity: Float32Array|null, retinaDrive: Float32Array|null, trace: Float32Array|null}>}  fresh arrays each call
   */
  forward(x, opts = {}) {
    const run = this._chain.then(() => this._forward(x, opts));
    this._chain = run.catch(() => {});
    return run;
  }

  async _forward(x, { activity = true, trace = null } = {}) {
    if (this.lost) throw new Error(this.lost);
    if (x.length !== this.D) throw new Error(`input must have ${this.D} values, got ${x.length}`);
    const dev = this.device;
    const t0 = performance.now();
    const tr = trace && trace.length ? this._ensureTrace(trace) : null;
    dev.queue.writeBuffer(this.bX, 0, x.buffer, x.byteOffset, x.byteLength);
    const enc = dev.createCommandEncoder();
    enc.clearBuffer(this.hA);                       // h_0 = 0
    const passDesc = this.timestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {};
    const pass = enc.beginComputePass(passDesc);
    if (this.sensoryInput) {
      pass.setPipeline(this.pInject);
      pass.setBindGroup(0, this.bgInject);
      pass.dispatchWorkgroups(this.dimsInject[0], this.dimsInject[1]);
    }
    if (this.vision) {
      pass.setPipeline(this.pRetina);
      pass.setBindGroup(0, this.bgRetina);
      pass.dispatchWorkgroups(this.dimsRetina[0], this.dimsRetina[1]);
    }
    pass.setPipeline(this.pBase);
    pass.setBindGroup(0, this.bgBase);
    pass.dispatchWorkgroups(this.dimsN[0], this.dimsN[1]);
    let nextReadout = 0;
    for (let t = 0; t < this.steps; t++) {
      for (const b of this.buckets) {
        pass.setPipeline(b.pipeline);
        pass.setBindGroup(0, b.bg[t & 1]);
        pass.dispatchWorkgroups(b.dims[0], b.dims[1]);
      }
      if (nextReadout < this.readouts.length && this.readouts[nextReadout].t === t) {
        const r = this.readouts[nextReadout++];
        pass.setPipeline(this.pGather);
        pass.setBindGroup(0, r.bg);
        pass.dispatchWorkgroups(r.dims[0], r.dims[1]);
      }
      if (tr) {
        pass.setPipeline(this.pGather);
        pass.setBindGroup(0, tr.bgs[t]);
        pass.dispatchWorkgroups(tr.dims[0], tr.dims[1]);
      }
    }
    pass.end();
    const heads = enc.beginComputePass();
    if (this.bgCentral) {
      heads.setPipeline(this.pCentral);
      heads.setBindGroup(0, this.bgCentral);
      heads.dispatchWorkgroups(this.dimsCentral[0], this.dimsCentral[1]);
    }
    heads.setPipeline(this.pPolicy);
    heads.setBindGroup(0, this.bgPolicy);
    heads.dispatchWorkgroups(this.dimsPolicy[0], this.dimsPolicy[1]);
    heads.setPipeline(this.pValue);
    heads.setBindGroup(0, this.bgValue);
    heads.dispatchWorkgroups(this.dimsValue[0], this.dimsValue[1]);
    heads.end();
    if (this.timestamps) {
      enc.resolveQuerySet(this.querySet, 0, 2, this.bQuery, 0);
      enc.copyBufferToBuffer(this.bQuery, 0, this.readback, 0, TS_BYTES);
    }
    enc.copyBufferToBuffer(this.bPolicy, 0, this.readback, this.offPolicy, 4 * this.numMoves);
    enc.copyBufferToBuffer(this.bValue, 0, this.readback, this.offValue, 4 * this.vRows);
    const withState = activity || !!tr;
    if (withState) {
      if (this.vision) enc.copyBufferToBuffer(this.bDrive, 0, this.readback, this.offRet, 4 * this.nRet);
      enc.copyBufferToBuffer(this.hFinal, 0, this.readback, this.offH, 4 * this.n);
    }
    if (tr) enc.copyBufferToBuffer(tr.buf, 0, this.readback, this.offTrace, 4 * this.steps * tr.L);
    dev.queue.submit([enc.finish()]);
    const mapBytes = tr ? this.offTrace + 4 * this.steps * tr.L : withState ? this.offTrace : this.offRet;
    await this.readback.mapAsync(GPUMapMode.READ, 0, mapBytes);
    if (this.lost) { try { this.readback.unmap(); } catch { /* gone */ } throw new Error(this.lost); }
    const mapped = this.readback.getMappedRange(0, mapBytes);
    const policy = new Float32Array(mapped.slice(this.offPolicy, this.offPolicy + 4 * this.numMoves));
    const hid = new Float32Array(mapped.slice(this.offValue, this.offValue + 4 * this.vRows));
    const act = withState ? new Float32Array(mapped.slice(this.offH, this.offH + 4 * this.n)) : null;
    const retinaDrive = withState && this.vision ? new Float32Array(mapped.slice(this.offRet, this.offRet + 4 * this.nRet)) : null;
    const traceOut = tr ? new Float32Array(mapped.slice(this.offTrace, this.offTrace + 4 * this.steps * tr.L)) : null;
    let gpuNs = 0;
    if (this.timestamps) {
      const ts = new BigUint64Array(mapped.slice(0, TS_BYTES));
      gpuNs = Number(ts[1] - ts[0]);
    }
    this.readback.unmap();
    this.lastForwardMs = performance.now() - t0;
    this.lastStepMs = gpuNs > 0 ? gpuNs / 1e6 / this.steps : this.lastForwardMs / this.steps;
    const value = valueFromHidden(this.arrays, this.valueHidden, hid);
    return { policy, value, activity: act, retinaDrive, trace: traceOut };
  }

  /** Evaluate several inputs (sequentially); each result owns its arrays. */
  async forwardBatched(xs, opts) {
    const out = [];
    for (const x of xs) out.push(await this.forward(x, opts));
    return out;
  }

  /** Release every GPU buffer; the device stays usable by others. */
  destroy() {
    for (const b of this._bufs) { try { b.destroy(); } catch { /* already destroyed */ } }
    this._bufs = [];
    try { this.querySet?.destroy(); } catch { /* ignore */ }
    if (!this.lost) this.lost = 'destroyed';
  }
}
