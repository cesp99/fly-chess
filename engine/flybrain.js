// flybrain.js — the fly brain as a pure typed-array recurrent network (SPEC §4).
//
// Same math as flychess/model/flybrain.py (reference: flychess.export.web.numpy_forward):
//   h_0 = 0
//   for t in 0..steps:  ion = W·h ; mod = W_mod·h                       (mod only under neuromod)
//                       pre = ion ⊙ (1 + tanh(mod)) + bias + inj ; h = (1-α)⊙h + α⊙act(pre)
//   feat   = concat(h_t[output_idx] for t in readout_steps, central_w · h_T[central_idx] + central_b)
//   policy = policy_w · feat + policy_b
//   value  = tanh(value_w2 · act(value_w1 · feat + value_b1) + value_b2)   (or linear head)
// where W is the connectome CSR (rows = post-synaptic neuron; the ionotropic synapses only under
// neuromod — the DA/SER/OCT ones are the mod_* CSR), inj[input_idx[k]] = w_in[k]·x + b_in[k] (the
// dense sensory path, `sensory_input`) and inj[retina_idx[k]] += w_ret[k]·planes[:, square_k] + b_ret[k]
// (the retina, `vision`: photoreceptor k looks at one board square through the 20 planes).
// The board x is constant across the timesteps, so both injections are computed once per forward.
// A blob without a feature (loader.modelFeatures) runs exactly the pre-feature fast path.

import { modelFeatures } from './loader.js';

/** erf via Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7) — enough for the 1e-2 parity tolerance. */
export function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}

// 'gelu' is the exact (erf) form by default, as torch.nn.functional.gelu; the export header may set
// gelu_approximate: 'tanh' to select the tanh approximation the model was trained with.
const ACTS = {
  relu: (v) => (v > 0 ? v : 0),
  tanh: Math.tanh,
  gelu: (v) => 0.5 * v * (1 + erf(v * 0.7071067811865476)),
  gelu_tanh: (v) => 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))),
  // saturating rectifier: sat * tanh(relu(v) / sat) — non-negative rates with a ceiling (header.activation_sat)
  satrelu: (sat) => (v) => (v > 0 ? sat * Math.tanh(v / sat) : 0),
};

export class FlyBrain {
  /**
   * @param {{header: object, arrays: Record<string, Int32Array|Float32Array|Uint8Array>}} model
   */
  constructor({ header, arrays }) {
    this.header = header;
    this.n = header.n ?? arrays.csr_indptr.length - 1;
    this.nnz = header.nnz ?? arrays.csr_indices.length;
    this.steps = header.steps ?? 8;
    this.activation = header.activation ?? 'relu';
    const pickAct = (name) => {
      const geluTanh = name === 'gelu' && /^tanh$/i.test(String(header.gelu_approximate || ''));
      const fn = name === 'satrelu' ? ACTS.satrelu(Number(header.activation_sat ?? 10)) : geluTanh ? ACTS.gelu_tanh : ACTS[name];
      if (!fn) throw new Error(`unknown activation ${name}`);
      return fn;
    };
    this.act = pickAct(this.activation);
    // the value MLP's hidden non-linearity may be exported separately (header.value_activation)
    this.valueAct = pickAct(header.value_activation || this.activation);
    this.inputDim = header.input_dim ?? (header.num_planes ?? 20) * 64;
    this.numMoves = header.num_moves ?? arrays.policy_b.length;

    this.indptr = arrays.csr_indptr;
    this.col = arrays.csr_indices;
    this.w = arrays.w;
    this.bias = arrays.bias;
    this.inputIdx = arrays.input_idx;
    this.outputIdx = arrays.output_idx;
    this.nIn = this.inputIdx.length;
    this.nOut = this.outputIdx.length;
    this.wIn = arrays.w_in;
    this.bIn = arrays.b_in;
    this.policyW = arrays.policy_w;
    this.policyB = arrays.policy_b;
    this.valueW = arrays.value_w;
    this.valueB = arrays.value_b;
    this.valueW2 = arrays.value_w2 ?? null;
    this.valueB2 = arrays.value_b2 ?? null;
    this.positions = arrays.positions ?? null;
    this.superClass = arrays.super_class ?? null;

    // optional features (SPEC §8): all off for a blob that does not carry them
    const F = modelFeatures(header, arrays);
    this.features = F;
    this.vision = F.vision;
    this.sensoryInput = F.sensoryInput;
    this.readoutSteps = F.readoutSteps;
    this.neuromod = F.neuromod;
    this.centralDim = F.centralDim;
    this.nRet = F.nRet;
    this.nCentral = F.nCentral;
    this.numPlanes = F.numPlanes;
    this.headIn = F.headIn;                    // n_out * |readout_steps| + central_dim
    this.retinaIdx = F.vision ? arrays.retina_idx : null;
    this.retinaSquare = F.vision ? arrays.retina_square : null;
    this.wRet = F.vision ? arrays.w_ret : null;
    this.bRet = F.vision ? arrays.b_ret : null;
    this.modIndptr = F.neuromod ? arrays.mod_indptr : null;
    this.modCol = F.neuromod ? arrays.mod_indices : null;
    this.wMod = F.neuromod ? arrays.w_mod : null;
    this.centralIdx = F.centralDim ? arrays.central_idx : null;
    this.centralW = F.centralDim ? arrays.central_w : null;
    this.centralB = F.centralDim ? arrays.central_b : null;
    this.valueHidden = this.valueW2 ? this.valueW.length / this.headIn : 0;
    // readout flags per 0-based timestep: readoutSlot[t] = index of the feature block written after step t, else -1
    this.readoutSlot = new Int8Array(this.steps).fill(-1);
    F.readoutSteps.forEach((t, r) => { this.readoutSlot[t - 1] = r; });

    // per-neuron leak α (accept a scalar export too)
    const a = arrays.alpha;
    if (a.length === this.n) this.alpha = a;
    else { this.alpha = new Float32Array(this.n).fill(a[0]); }
    this.oneMinusAlpha = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) this.oneMinusAlpha[i] = 1 - this.alpha[i];

    if (this.w.length !== this.nnz || this.indptr.length !== this.n + 1) throw new Error('CSR shape mismatch');
    if (this.sensoryInput && this.wIn.length !== this.nIn * this.inputDim) throw new Error('w_in shape mismatch');
    if (!this.sensoryInput && !this.vision) throw new Error('the brain has no input path (neither sensory_input nor vision)');
    if (this.policyW.length !== this.numMoves * this.headIn) throw new Error('policy_w shape mismatch');
    if (this.valueW.length % this.headIn !== 0) throw new Error('value_w shape mismatch');

    // scratch buffers (double-buffered hidden state)
    this._h = new Float32Array(this.n);
    this._h2 = new Float32Array(this.n);
    this._base = new Float32Array(this.n);   // bias + injection, fixed for a forward pass
    this._feat = new Float32Array(this.headIn);
    this._hid = new Float32Array(Math.max(this.valueHidden, 1));
    this._retDrive = new Float32Array(this.nRet);   // per-photoreceptor input drive of the last forward
    this.lastStepMs = 0;
  }

  /**
   * One full forward pass.
   * @param {Float32Array} x  flattened planes, length input_dim (1280)
   * @param {{trace?: Int32Array|Uint32Array|number[]|null, activity?: boolean}} [opts]
   *   trace: neuron indices to sample after every timestep → result.trace = Float32Array(steps * trace.length),
   *   laid out [t][j] (t = 0 is the state after the first step). `activity` is accepted for API symmetry
   *   with FlyBrainGPU and ignored (the final state costs nothing here).
   * @returns {{policy: Float32Array, value: number, activity: Float32Array, retinaDrive: Float32Array|null, trace: Float32Array|null}}
   *   activity is the final hidden state and retinaDrive the per-photoreceptor drive (null without
   *   vision) — both views that are reused on the next call (copy if you keep them); policy and trace are fresh.
   */
  forward(x, opts) {
    if (x.length !== this.inputDim) throw new Error(`input must have ${this.inputDim} values, got ${x.length}`);
    const n = this.n, act = this.act, D = this.inputDim;
    const indptr = this.indptr, col = this.col, w = this.w, alpha = this.alpha, oma = this.oneMinusAlpha;
    const base = this._base;
    base.set(this.bias);
    // sensory injection: constant across timesteps
    if (this.sensoryInput) {
      const wIn = this.wIn, bIn = this.bIn, inputIdx = this.inputIdx;
      for (let k = 0, off = 0; k < this.nIn; k++, off += D) {
        let s = bIn[k];
        for (let j = 0; j < D; j++) s += wIn[off + j] * x[j];
        base[inputIdx[k]] += s;
      }
    }
    // retina: photoreceptor k sees the 20 planes of one square (what the fly sees = retinaDrive)
    if (this.vision) {
      const P = this.numPlanes, wRet = this.wRet, bRet = this.bRet, sq = this.retinaSquare, ridx = this.retinaIdx, drive = this._retDrive;
      for (let k = 0, off = 0; k < this.nRet; k++, off += P) {
        let s = bRet[k];
        const s0 = sq[k];
        for (let p = 0, xi = s0; p < P; p++, xi += 64) s += wRet[off + p] * x[xi];
        drive[k] = s;
        base[ridx[k]] += s;
      }
    }
    const traceIdx = opts && opts.trace ? opts.trace : null;
    const L = traceIdx ? traceIdx.length : 0;
    const trace = traceIdx ? new Float32Array(this.steps * L) : null;
    const feat = this._feat, outputIdx = this.outputIdx, nOut = this.nOut, slot = this.readoutSlot;
    let h = this._h, h2 = this._h2;
    h.fill(0);
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const relu = act === ACTS.relu;
    if (this.neuromod) {
      // gated rows: pre = ion * (1 + tanh(mod)) + base; rows without modulatory inputs keep pre = ion + base
      const mIndptr = this.modIndptr, mCol = this.modCol, wMod = this.wMod;
      for (let t = 0; t < this.steps; t++) {
        let e = indptr[0], me = mIndptr[0];
        for (let i = 0; i < n; i++) {
          const end = indptr[i + 1], mend = mIndptr[i + 1];
          let s = 0;
          for (; e < end; e++) s += w[e] * h[col[e]];
          if (me < mend) {
            let m = 0;
            for (; me < mend; me++) m += wMod[me] * h[mCol[me]];
            s *= 1 + Math.tanh(m);
          }
          s += base[i];
          h2[i] = oma[i] * h[i] + alpha[i] * act(s);
        }
        const tmp = h; h = h2; h2 = tmp;
        if (slot[t] >= 0) { const off = slot[t] * nOut; for (let j = 0; j < nOut; j++) feat[off + j] = h[outputIdx[j]]; }
        if (trace) { const off = t * L; for (let j = 0; j < L; j++) trace[off + j] = h[traceIdx[j]]; }
      }
    } else {
      for (let t = 0; t < this.steps; t++) {
        let e = indptr[0];
        if (relu) {
          for (let i = 0; i < n; i++) {
            const end = indptr[i + 1];
            let s = base[i];
            for (; e < end; e++) s += w[e] * h[col[e]];
            h2[i] = oma[i] * h[i] + (s > 0 ? alpha[i] * s : 0);
          }
        } else {
          for (let i = 0; i < n; i++) {
            const end = indptr[i + 1];
            let s = base[i];
            for (; e < end; e++) s += w[e] * h[col[e]];
            h2[i] = oma[i] * h[i] + alpha[i] * act(s);
          }
        }
        const tmp = h; h = h2; h2 = tmp;
        if (slot[t] >= 0) { const off = slot[t] * nOut; for (let j = 0; j < nOut; j++) feat[off + j] = h[outputIdx[j]]; }
        if (trace) { const off = t * L; for (let j = 0; j < L; j++) trace[off + j] = h[traceIdx[j]]; }
      }
    }
    this.lastStepMs = ((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0) / this.steps;
    // keep the final state in this._h so the returned activity stays valid until the next forward
    this._h = h; this._h2 = h2;
    // central summary: Linear on the final activity of the central-brain neurons, appended last
    if (this.centralDim > 0) {
      const C = this.centralDim, J = this.nCentral, cw = this.centralW, cb = this.centralB, cidx = this.centralIdx;
      const off0 = nOut * this.readoutSteps.length;
      for (let c = 0, off = 0; c < C; c++, off += J) {
        let s = cb[c];
        for (let j = 0; j < J; j++) s += cw[off + j] * h[cidx[j]];
        feat[off0 + c] = s;
      }
    }
    const res = this._heads(feat);
    res.activity = h;
    res.retinaDrive = this.vision ? this._retDrive : null;
    res.trace = trace;
    return res;
  }

  _heads(feat) {
    const K = this.headIn;
    const policy = new Float32Array(this.numMoves);
    const pw = this.policyW, pb = this.policyB;
    const n4 = K & ~3;
    for (let m = 0, off = 0; m < this.numMoves; m++, off += K) {
      let s0 = pb[m], s1 = 0, s2 = 0, s3 = 0;
      let j = 0;
      for (; j < n4; j += 4) {
        s0 += pw[off + j] * feat[j]; s1 += pw[off + j + 1] * feat[j + 1];
        s2 += pw[off + j + 2] * feat[j + 2]; s3 += pw[off + j + 3] * feat[j + 3];
      }
      for (; j < K; j++) s0 += pw[off + j] * feat[j];
      policy[m] = s0 + s1 + s2 + s3;
    }
    let value;
    if (this.valueHidden > 0) {
      const H = this.valueHidden, vw = this.valueW, vb = this.valueB, hid = this._hid, act = this.valueAct;
      for (let k = 0, off = 0; k < H; k++, off += K) {
        let s = vb[k];
        for (let j = 0; j < K; j++) s += vw[off + j] * feat[j];
        hid[k] = act(s);
      }
      let v = this.valueB2[0];
      for (let k = 0; k < H; k++) v += this.valueW2[k] * hid[k];
      value = Math.tanh(v);
    } else {
      let v = this.valueB[0];
      const vw = this.valueW;
      for (let j = 0; j < K; j++) v += vw[j] * feat[j];
      value = Math.tanh(v);
    }
    return { policy, value };
  }

  /** Simple loop over inputs; each result gets its own copy of the activity. */
  forwardBatched(xs, opts) {
    const res = [];
    for (const x of xs) {
      const r = this.forward(x, opts);
      res.push({ ...r, activity: r.activity.slice(), retinaDrive: r.retinaDrive ? r.retinaDrive.slice() : null });
    }
    return res;
  }
}

/**
 * Softmax over the legal move indices only.
 * @param {Float32Array} policy  raw logits (numMoves)
 * @param {ArrayLike<number>} legalIdx
 * @param {number} [temperature=1]
 * @returns {Float32Array} probabilities aligned with legalIdx
 */
export function policyForLegal(policy, legalIdx, temperature = 1) {
  const L = legalIdx.length;
  const p = new Float32Array(L);
  if (L === 0) return p;
  const invT = 1 / Math.max(temperature, 1e-6);
  let mx = -Infinity;
  for (let i = 0; i < L; i++) { const v = policy[legalIdx[i]] * invT; p[i] = v; if (v > mx) mx = v; }
  let sum = 0;
  for (let i = 0; i < L; i++) { const e = Math.exp(p[i] - mx); p[i] = e; sum += e; }
  for (let i = 0; i < L; i++) p[i] /= sum;
  return p;
}

/** Index of the largest entry of a typed array. */
export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/** Sample an index from a probability vector (uses Math.random unless `rnd` given). */
export function sampleIndex(probs, rnd = Math.random) {
  let r = rnd();
  for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; }
  return probs.length - 1;
}

/** Indices of the k largest values, descending. */
export function topK(arr, k) {
  const idx = Array.from({ length: arr.length }, (_, i) => i);
  idx.sort((a, b) => arr[b] - arr[a]);
  return idx.slice(0, k);
}
