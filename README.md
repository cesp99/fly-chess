# web/ — play chess against the fly brain, in the browser

A static site (ES modules, no build step, no CDN) that runs the exact network trained in Python:
the FlyWire connectome as a recurrent net, evaluated inside a Web Worker — on the GPU through WebGPU
compute shaders when the browser has it, otherwise with plain typed-array CSR kernels (same math,
automatic fallback). Nothing but the network picks moves.

```
index.html / style.css / app.js   landing, loading screen, game UI, party mode, share card, leaderboard, sounds
board.js                          dependency-free SVG chess board (drag / click, legal dots, promotion picker)
eye.js                            "fly's eye": both compound eyes from the blob's retina map, coloured by the board
                                  the fly sees / the input current it receives; gaze commentary from the retina drive
brainviz.js                       the live brain canvas + the thought replay (activity trace scrubbed through the
                                  timesteps) and the per-class activity strip chart
engine/loader.js                  fetch brain.json + brain.flyb(.gz), stream progress, gunzip, Cache API, parse (§8);
                                  modelFeatures(): which §8 optional features a blob carries (old blobs: none)
engine/flybrain.js                FlyBrain.forward(): SPEC §4 dynamics in pure JS (retina, neuromod, readout_steps,
                                  central summary, activity trace); softmax helpers
engine/flybrain-gpu.js            FlyBrainGPU: the same forward pass as WGSL compute shaders (async forward), plus
                                  f32 JS mirrors of every kernel (emulate*) so the shader math is testable under node
engine/mcts.js                    PUCT MCTS (c_puct 1.5) for "superfly"; every evaluation is the brain (sync or async)
engine/worker.js                  Web Worker protocol (load / move / eval), difficulty semantics (§9), backend choice + fallback
engine/encoding.js                board -> planes, move <-> index (shared with Python)          [other module]
vendor/chess.js                   chess.js 1.4.0 (BSD-2)                                        [other module]
assets/fly.svg                    the fly (own artwork; pieces are drawn in board.js)
model/                            gitignored: brain.json + brain.flyb + brain.flyb.gz from `fly export-web`
test/eye.test.mjs                 node --test: the pure parts of eye.js / brainviz.js (layout, colours, gaze, trace means)
test/app.test.mjs                 node --test: the page in headless Chromium — game flow, retina / retina-less blobs, replay
test/flybrain.test.mjs            node --test: forward vs. reference, loader layout, MCTS on a stub brain
test/flybrain-gpu.test.mjs        node --test: WGSL kernel math (f32 emulation) vs. float64 reference; async MCTS
test/features.test.mjs            node --test: the §8 optional features in both engines vs. a float64 reference
                                  (support/reference.mjs), trace mode, loader.modelFeatures
test/parity.test.mjs              node --test: both engines (FlyBrain + the WebGPU mirror) == Python reference on
                                  the exported brain (SPEC §8); FLY_MODEL_DIR / FLY_VECTORS pick another export
test/browser/gpu-parity.mjs       node script (not --test): real model in headless Chromium, WebGPU vs JS (and vs
                                  the Python vectors) parity, trace + retina drive, latency, worker backends and
                                  device-loss fallback; FLY_MODEL_DIR / FLY_VECTORS pick another export
```

## Run locally

```bash
fly export-web --run <run-name>      # writes web/model/brain.json, brain.flyb, brain.flyb.gz
scripts/serve-web.sh                 # http://localhost:8000  (python -m http.server on web/)
```

A model must be served from `model/` next to `index.html`. Without it the landing page explains
what to do. The first visit downloads the `.gz` blob (streamed, gunzipped in the worker) and caches
the decoded buffer in the Cache API, keyed by `run_name` + `exported_at` + `blob_sha256`; later visits load from cache.
Blob requests carry the same version tag as a query string (`brain.flyb.gz?v=…`) and are fetched with
`cache: 'no-cache'`, and every blob (downloaded or cached) is checked against the header's `total_bytes` and
`blob_sha256`, so a re-deploy never pairs the new `brain.json` with a stale `brain.flyb` from an HTTP cache.

## Worker protocol

```
→ {type:'load', baseUrl, gpu?}                            baseUrl absolute (app passes new URL('model/', location)); gpu:false = JS only
← {type:'progress', loaded, total, phase, n, nnz, runName}
← {type:'ready', header, legend, features, retina, sample:{idx, xy, cls}, silhouette:{xy, cls}, fromCache, bytes, backend, gpu}
                                                          backend: 'webgpu' | 'js'; gpu: {timestamps} or {error: why not}
                                                          features: {vision, sensoryInput, readoutSteps, neuromod, centralDim, nRet, nCentral, steps}
                                                          retina (vision blobs, else null): {n, uv:Float32Array(2n), eye:Uint8Array(n),
                                                          type:Uint8Array(n), legend:['R1-6','R7','R8'], square:Uint8Array(n), idx:Int32Array(n)}
→ {type:'move', id, fen, moves:[uci…], difficulty, trace?}  moves = full game history (repetition plane); trace:true asks for the activity trace
← {type:'thinking', id, done, total}                      superfly only, every 10 simulations
← {type:'move', id, move, san, policyTop:[{uci,san,p}], value, activitySample:Float32Array(2048), retinaDrive, trace, traceSteps, thinkMs, sims, stepMs, backend}
                                                          retinaDrive: Float32Array(n_ret), the input drive of every photoreceptor for the
                                                          position the fly looked at (what the fly sees; null without vision)
→ {type:'eval', id, fen, moves}
← {type:'eval', id, value, policyTop, activitySample, retinaDrive, trace:Float32Array(steps×2048), traceSteps, backend}
                                                          trace[t*2048 + j] = activity of sample neuron j after timestep t (t = 0 first step)
→ {type:'bench', id, reps?, backend?, activity?, trace?}  per-forward latency of one engine
← {type:'bench', id, forwardMs, stepMs, backend, reps, trace}
← {type:'backend', backend:'js', reason}                  unsolicited: the WebGPU device was lost, JS answers from now on
← {type:'error', id?, message}
```

**Backends.** On `load` the worker always builds the plain-JS `FlyBrain` and then tries `FlyBrainGPU.create`
(skipped with `gpu:false`, bounded to 20 s): WebGPU adapter + device, the CSR / heads uploaded once
(~55 MB of f32 storage buffers), one compute pipeline per kernel. Every forward then goes through one
wrapper: WebGPU while it is alive, JS otherwise. A GPU failure (device lost, out of memory, shader
error) is reported once as `{type:'backend'}` and the JS engine answers that request and all later ones;
`backend` in each reply says which engine produced it, and the page shows it in the specimen line
("engine WebGPU" / "engine JS"). Where WebGPU is missing (Firefox without the flag, `--disable-gpu`,
insecure origins) `ready` carries `gpu: {error}` and everything runs in JS as before.

`FlyBrainGPU.forward(x, {activity})` is asynchronous (one command buffer: injection, all timesteps
with ping-pong hidden-state buffers, both heads; one `mapAsync` readback of 4168 logits + the value
hidden layer + optionally the full final state, 537 KB). `mcts.js` accepts either kind of brain:
`step()` returns a boolean for `FlyBrain` and a promise for `FlyBrainGPU`; `runMCTSAsync` awaits
only when it is handed a promise, so the JS path is unchanged. The recurrent SpMV runs rows bucketed
by synapse count (1 / 8 / 64 lanes per row, `LANE_BUCKETS`) so the ~150 rows with thousands of
inputs do not serialise a whole timestep.

`value` is always from the side to move's perspective at the root (= the fly, when it is asked to move).
`activitySample` holds the final-step activity of 2048 fixed neurons (chosen deterministically at load — a
stratified sample: every super class and the retina's photoreceptors get up to 160 members, the rest of the
budget is a spread random sample, so the descending / motor side of the thought replay is populated; their
2-D connectome positions and super-class are sent once in `ready`); `trace` holds the same
neurons after *every* timestep (the last row equals `activitySample`). The trace is a mode of both engines'
forward pass — `forward(x, {trace: Int32Array})` — costing one gather dispatch per timestep on the GPU
(read back with everything else in the single mapAsync, ≈ +0.2 ms on fly2) and a 2048-element copy per
step in JS; it is never requested by the search.

**Blob features (SPEC §8).** Both engines run every optional feature a blob declares, with exact parity to
`flychess.export.web.numpy_forward`: the retina (`vision`: photoreceptor k receives
`w_ret[k] · planes[:, retina_square[k]] + b_ret[k]` each step — a small kernel over the photoreceptors on the
GPU, folded into `base` next to the sensory injection), `sensory_input=false` (no `w_in` path), neuromodulation
(`pre = ion · (1 + tanh(W_mod · h)) + bias + inj`: the rows with modulatory inputs run a gated variant of the
step kernel in their own lane buckets — 8 storage bindings, the WebGPU baseline — and the other rows keep the
plain kernel), `readout_steps` (the heads read the concatenated `output_idx` activity of several timesteps,
gathered into a feature buffer as the steps go by) and the central summary (`central_w · h_T[central_idx] +
central_b`, a matvec appended to the feature vector). `loader.modelFeatures` resolves the switches from the
header flags *and* the arrays, so a pre-feature blob such as `fly2` runs the byte-identical pre-feature shaders.

**What the fly sees (the page).** With a vision blob the game screen gains a *fly's eye* panel under the
avatar: `eye.js` draws both compound eyes from `retina_uv` (left eye `u < 0.5`, right eye `u ≥ 0.5`, dorsal
up; the photoreceptors of one ommatidial column sit as R7 / R8 at the centre with the R1-6 on a ring), each
photoreceptor coloured by the board square it watches *from the fly's side* (`retina_square` is in the mover's
perspective, so a black fly's squares are mirrored with `sq ^ 56`): light / dark tones for empty squares,
amber for its own pieces, blue for yours, brighter for bigger pieces, the last move's squares pulsing. The
*feels* toggle colours them by `retinaDrive` instead — the input current `w_ret · planes + b_ret` the engines
report for the position the fly last looked at — green excitatory, red inhibitory, scaled to the view's 98th
percentile; hovering names the photoreceptor (`R7 · left eye · sees e4 · your knight · drive +0.83`). The
commentary occasionally reports the (eye, square) whose mean input current changed most since the fly's
previous glance ("the fly's right eye is fixed on your knight on f6") — read off the retina drive, never
from chess heuristics. A blob without a retina (fly2) shows a one-line note instead. Every fly move is
requested with `trace: true`; `brainviz.js` replays the trace on the brain canvas (play / scrub / step
counter, ≈ 150 ms per timestep) and draws a strip chart of the mean |activity| per super class (plus a
`retina` row for sampled photoreceptors) so the wave eyes → optic lobe → central brain → descending neurons
is visible; *look again* sends an `eval` for the current position and replays that thought.

Difficulties (SPEC §9): **larva** samples the legal-masked policy at temperature 1.2; **fly** takes the
top-3 policy moves, plays each, asks the value head how the opponent likes the result and keeps the move
that is worst for them; **superfly** runs PUCT simulations with a backend-dependent budget (400 on WebGPU,
40 in plain JS; `DIFFICULTY.superfly.sims` in worker.js).

## Tests

```bash
node --test web/test/*.mjs
for f in web/*.js web/engine/*.js; do node --check "$f"; done
node web/test/browser/gpu-parity.mjs        # needs web/model/ and a Chromium with WebGPU (CHROME=… to pick one)
# another export (e.g. a scratch fly3-style blob) and its Python vectors, for both scripts:
FLY_MODEL_DIR=/tmp/site3/model FLY_VECTORS=/tmp/site3/model-vectors.json node --test web/test/parity.test.mjs
FLY_MODEL_DIR=/tmp/site3/model FLY_VECTORS=/tmp/site3/model-vectors.json node web/test/browser/gpu-parity.mjs
```

The test builds a tiny random brain as a real `.flyb` blob, parses it with the loader and checks
`FlyBrain.forward` against a float64 reference for relu/tanh/gelu, linear and MLP value heads,
then runs MCTS with a constant-policy stub brain on chess.js (mate-in-one must be found).
`flybrain-gpu.test.mjs` checks the WGSL kernels through their f32 JS mirrors (`emulateInject` /
`emulateStep` / `emulateMatvec` / `emulateForward`, `Math.fround` after every operation, same
summation order as the shader lanes) against the float64 reference and `FlyBrain` for
relu / tanh / gelu / gelu-tanh / satrelu and both value heads, and that MCTS gives the same search
with a promise-returning brain. `test/browser/gpu-parity.mjs` serves `web/` with the real model,
starts headless Chromium (`--headless=new --enable-unsafe-webgpu --ignore-gpu-blocklist
--enable-features=Vulkan --use-angle=vulkan`; `GPU_ANGLE=swiftshader` for a software run) and
compares `FlyBrainGPU` with `FlyBrain` on 5 positions (|Δlogit| ≤ 1e-2 on all 4168 logits,
|Δvalue| ≤ 1e-2, same legal argmax), then drives `worker.js`: backend report, moves at every
difficulty, bench, and a forced device loss followed by moves from the JS fallback.
`features.test.mjs` writes synthetic blobs with every combination of the §8 features (retina with and without
the sensory path, neuromod, readout steps, central summary, a legacy blob without the arrays) and checks
`FlyBrain`, the GPU mirror and the trace / retina-drive outputs against a float64 port of `numpy_forward`.
Cross-language parity with Python: `node --test web/test/parity.test.mjs` loads `web/model/` with the loader and
checks every logit / value of **both** engines (FlyBrain, and the WebGPU engine through `emulateForward`) against
`tests/vectors/model.json` (written by `fly export-web` from the same blob); it skips when either file is missing
or belongs to another export. `FLY_MODEL_DIR` / `FLY_VECTORS` point both it and `gpu-parity.mjs` at another
export; the browser script then also compares the real WebGPU engine with the Python vectors. Measured: fly2
(no features) max |Δlogit| 1.8e-4 (JS), 2.4e-4 (WebGPU) vs Python; a tiny all-features export (vision +
neuromod + readout_steps [2,4] + central_dim 16) 4.8e-7 for both engines, JS mirror and real device alike.

## Performance

Measured with Node 22 on a desktop CPU for a synthetic full-size brain (134k neurons, 2.7M synapses,
random column pattern — a worst case for cache locality): ≈3.5 ms per recurrent timestep, ≈37 ms per
forward pass at 8 steps including the 4168×2048 policy head. Superfly (200 simulations) therefore
takes ≈7–8 s per move; the UI shows the simulation count ticking.

WebGPU, exported `fly2` (134,209 neurons, 2,700,513 connections, 16 satrelu steps), headless Chromium 149
on an RTX 5080 via Vulkan: 0.09–0.16 ms per recurrent step (timestamp queries), 2.5–3.3 ms per forward
including submit + readback (≈2.5 ms without the 537 KB activity readback), against 79–88 ms in plain JS
in the same browser (4.5 ms per step) — ≈24×. Superfly (400 simulations on WebGPU, 40 in JS) answers in ≈1.5–3 s either way.
GPU vs JS on 5 positions: |Δlogit| ≤ 1.9e-4, |Δvalue| ≤ 1.1e-6, |Δactivity| ≤ 3e-6, same argmax.
The §8 feature support changed none of this for fly2 (same shaders; measured before / after on a GPU shared
with a training run: 7.8 / 7.9 ms per forward, 70 / 70 ms in node JS); the trace adds ≈ 0.2 ms on WebGPU
and nothing measurable in JS. The tiny all-features export (2000 neurons, 4 steps) runs in 3.3 ms on WebGPU
(dispatch-bound) and 1.1–1.5 ms in JS. A fly3 export (ckpt-4000: vision + neuromod + readout_steps [8, 16] +
central_dim 128, 2,663,502 ionotropic + 37,011 modulatory synapses, heads 4168 × 2958) costs 8.7 ms per
forward on the same shared GPU (fly2 7.9 ms; step 0.32 vs 0.26 ms — the gated kernel, the wider heads and the
128 × 32,292 central summary) and 88–100 ms in JS (fly2 70–83 ms); running the gated rows as separate
dispatches instead had cost 12.1 ms. GPU vs Python on the fly3 vectors: |Δlogit| ≤ 8.4e-5, JS ≤ 9.5e-6.

## Deploy

`scripts/deploy-pages.sh` copies `web/` (with `web/model/`) into a temporary worktree on the
`gh-pages` branch, commits and pushes. Every file must stay below GitHub's 100 MB limit.
Each deploy is a single snapshot commit that replaces the branch (force push) so the ~75 MB of
model blobs are not accumulated in history; `KEEP_HISTORY=1` appends instead. Re-running with an
unchanged `web/` is a no-op (`BUILD.txt` is keyed on the source commit and `brain.json`'s
`exported_at`, not on the wall clock).

## Credits

FlyWire connectome — Dorkenwald et al. 2024 and Schlegel et al. 2024 (*Nature*), data CC BY-NC 4.0.
chess.js — BSD-2. Piece and fly artwork were drawn for this project.
