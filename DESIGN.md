# Crossmodel Latency Model — Design

Coefficient fitting pipeline for the [inference-sim](https://github.com/inference-sim/inference-sim) crossmodel step-time model. Fits 10 parameters (α₀–α₂, β₁–β₇) from vLLM journey trace data.

- Model specification: [inference-sim/inference-sim#489](https://github.com/inference-sim/inference-sim/issues/489#issuecomment-4013680061)
- Fitting specification: [inference-sim/training#3](https://github.com/inference-sim/training/issues/3)

---

## What the simulator needs

inference-sim replays a workload through a simulated vLLM scheduler. It handles queueing and scheduling internally. What it CANNOT predict are three types of overhead that depend on the real system:

1. **API processing overhead** — time from request arrival at the API server to handoff to the engine (tokenization, validation). Modeled by α₀.
2. **GPU step time** — time to execute one scheduler step (prefill compute, decode compute, weight loading, communication). Modeled by β₁–β₇.
3. **Post-decode overhead** — time from scheduler completion to response delivery (detokenization, API response). Modeled by α₁ + α₂·output_tokens.

## Request lifecycle and where each parameter acts

```
api.ARRIVED ──α₀──► journey.QUEUED ──simulator──► journey.SCHEDULED
                                                       │
                                              ┌────────┘
                                              ▼
                                    step 1: β₁..β₇ (prefill)
                                    step 2: β₁..β₇ (decode)
                                      ...
                                    step N: β₁..β₇ (decode)
                                              │
                              journey.FINISHED─┘
                                    │
                                    ├──α₁ + α₂·n──► api.DEPARTED
```

- **α₀**: constant API processing overhead (µs). Measured from `api.ARRIVED.ts → journey.QUEUED.ts`.
- **β₁–β₇**: GPU step-time model. Predicts each scheduler step's duration from roofline basis functions.
- **α₁ + α₂·n**: post-decode CPU overhead (µs). Measured from `journey.FINISHED.ts → api.DEPARTED.ts`. α₁ is fixed per-request overhead, α₂ is per-output-token cost (detokenization).
- **Scheduler queueing** (`QUEUED → SCHEDULED`): NOT modeled here — the simulator produces this internally via its own scheduler implementation.

## Step-time formula

```
StepTime(s) = β₁ · max(T_pf_compute, T_pf_kv)     prefill roofline bottleneck
            + β₂ · max(T_dc_compute, T_dc_kv)     decode roofline bottleneck
            + β₃ · T_weight                         weight loading
            + β₄ · T_tp                             TP communication
            + β₅ · L                                per-layer overhead
            + β₆ · batch_size + β₇                 scheduling overhead
```

All basis functions output microseconds. β₁–β₄ are dimensionless corrections to the roofline model (analytical prior = 1.0). β₅ is µs/layer, β₆ is µs/request, β₇ is µs/step.

## End-to-end prediction

For a request r with output_tokens n:

```
api_overhead(r)     = α₀                                                    [constant]
gpu_processing(r)   = Σ_{s ∈ active_steps(r)} StepTime(s)                  [sum of step times]
postdecode(r)       = α₁ + α₂ · n                                          [fixed + per-token]
```

The simulator composes these into end-to-end predictions:
```
ttft(r)  = api_overhead + simulator_queueing + prefill_step_time
e2e(r)   = api_overhead + simulator_queueing + gpu_processing + postdecode
```

`simulator_queueing` is produced by the simulator's scheduler, not learned.

Note on ground-truth alignment: `RequestLabel.processing_us` = FINISHED.ts − SCHEDULED.ts − gaps (covers scheduler steps only, no API or post-decode overhead). `RequestLabel.e2e_us` = FINISHED.ts − QUEUED.ts (includes scheduler wait but not API overhead or post-decode). Each α/β is evaluated against its own signal, not against e2e directly.

---

## Pipeline stages

### Stage 1 — Validation (`validate_traces.py`)

**Done.** Validates journey trace integrity across all 16 experiments.

5 correctness checks per request:

1. Lifecycle completeness — QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED all present
2. Timestamp ordering — Q ≤ S ≤ F ≤ D (monotonic)
3. Step index ordering — S ≤ F ≤ D
4. Preemption pairing — every PREEMPTED has a matching SCHEDULED(RESUME)
5. Single-step prefill — FIRST_TOKEN.step - SCHEDULED.step ≤ 1

Gate: exit code 1 if any experiment exceeds 1% failure rate.

### Stage 2 — Step reconstruction (`reconstruct_steps.py`)

**Done.** Reconstructs per-step batch composition from journey events using teacher-forced reconstruction.

4-phase internal pipeline:

```
traces.json → parse_events → RequestTimeline
            → _build_intervals → intervals + decode_tokens_before
            → _compute_label → RequestLabel (ground-truth timing)
            → _reconstruct_steps → list[ReconstructedStep] (batch per step)
```

Key design decisions:

- **Teacher-forced**: uses real batch compositions from the actual execution, not simulated. Prevents circular dependency where predictions alter scheduling.
- **Greedy-fill prefill**: mirrors vLLM's `max_num_batched_tokens` budget. Decode gets 1 token first, remaining budget goes to prefill in arrival order.
- **Sweep-line algorithm**: O(N log N) for step reconstruction. Maintains active prefill/decode sets via sorted start/end events.
- **Preemption handling**: decode context_length excludes gap steps via `decode_tokens_before` cumulative tracking. Prefill-phase preemption re-enters PREFILL on resume (tracked via `prefill_complete` flag). Per-interval token counts from `prefill.done_tokens`.
- **Event sorting**: events sorted by (step, ts) before state machine processing — correctness requirement for out-of-order OTEL exports.
- **Algorithm-local state**: `prefill_remaining` is a local dict in `_reconstruct_steps`, not stored on `RequestTimeline`.

Output types (frozen):

- `ReconstructedStep`: step_id, prefill_reqs (tuple[PrefillEntry]), decode_reqs (tuple[DecodeEntry]), totals
- `RequestLabel`: timing in µs (queueing, ttft, processing, e2e), preemption count, failed flag
- `ExperimentReconstruction`: steps + labels + max_num_batched_tokens

### Stage 3 — Basis functions (`basis_functions.py`)

**Done.** Computes 7 analytical roofline basis functions per step in microseconds.

Each basis function is a standalone pure function. Adding a new one = 1 function + 1 field on StepBasisValues + 1 line in `compute_step_basis()`.

#### Basis function formulas

All inputs from: model `config.json` (architecture), `h100-sxm.json` (hardware peaks), `ReconstructedStep` (batch composition).

**T_pf_compute** (prefill compute):
```
FLOPs_proj = L · 2 · T_pf · d · (2·d + 2·d_kv) / TP
FLOPs_attn = L · Σᵢ 4 · (H/TP) · T_pf_i · (S_pf_i + T_pf_i/2) · d_h
FLOPs_ffn  = L · T_pf · k_eff · 6 · d · d_ff / TP
result     = (FLOPs_proj + FLOPs_attn + FLOPs_ffn) / (FLOPS_peak · 1e6)   [µs]
```

**T_dc_compute** (decode compute):
```
FLOPs_proj = L · 2 · T_dc · d · (2·d + 2·d_kv) / TP
FLOPs_attn = L · Σⱼ 4 · (H/TP) · S_dc_j · d_h
FLOPs_ffn  = L · T_dc · k_eff · 6 · d · d_ff / TP
result     = (FLOPs_proj + FLOPs_attn + FLOPs_ffn) / (FLOPS_peak · 1e6)   [µs]
```

**T_pf_kv** (prefill KV write bandwidth):
```
Bytes = L · 2 · (kv_heads/TP) · d_h · T_pf · 2
result = Bytes / (BW_hbm · 1e6)   [µs]
```

**T_dc_kv** (decode KV read+write bandwidth):
```
Bytes = L · 2 · (kv_heads/TP) · d_h · 2 · (Σⱼ S_dc_j + T_dc)
result = Bytes / (BW_hbm · 1e6)   [µs]
```

**T_weight** (weight loading):
```
N_eff = 1 (dense) or min(N, max(k, B·k)) (MoE)
Bytes_attn = L · d · (2·d + 2·d_kv) · 2 / TP
Bytes_ffn  = L · N_eff · 3 · d · d_ff · 2 / TP
result     = (Bytes_attn + Bytes_ffn) / (BW_hbm · 1e6)   [µs]
```

**T_tp** (tensor-parallel communication bandwidth):
```
msg_bytes = (T_pf + T_dc) · d · 2
TP = 1:  0
TP = 2:  L · 2 · msg_bytes / (BW_nvlink · 1e3)      [point-to-point]
TP ≥ 4:  L · 2 · 2 · msg_bytes / (BW_nvlink · 1e3)  [ring all-reduce]
```

Note: the projection formula `(2·d + 2·d_kv)` includes the output projection O(d×d), corrected from the original design after cross-referencing vLLM source (`LlamaAttention.o_proj`). See [issue #3 comment](https://github.com/inference-sim/training/issues/3).

#### Notation

From `config.json`: L (layers), d (hidden), H (heads), kv_heads, d_h = d/H, d_kv = kv_heads·d_h, d_ff (FFN dim), N (experts), k (active experts), k_eff = max(1, k).

From `h100-sxm.json`: FLOPS_peak = fp16_tensor_core/2 (dense TFLOPS), BW_hbm (TB/s), BW_nvlink (GB/s).

From `ReconstructedStep`: T_pf (total prefill tokens), T_dc (decode request count), T_pf_i (tokens this step for prefill request i = `PrefillEntry.tokens_this_step`), S_pf_i (total prompt tokens for prefill request i = `PrefillEntry.prompt_tokens`), S_dc_j (context length for decode request j = `DecodeEntry.context_length`), batch_size.

### Stage 4 — Coefficient fitting (`fit_coefficients.py`)

**Planned.** Three-phase NNLS fitting.

#### Phase 1: API processing overhead (α₀)

Estimates the constant delay between request arrival at the API server and handoff to the engine scheduler.

```
α₀ = mean( QUEUED.ts − ARRIVED.ts )   across training requests   [µs]
```

- Signal: `api.ARRIVED` on `llm_request` span → `journey.QUEUED` on `llm_core` span
- This is API-side work: request parsing, tokenization, validation, engine handoff
- Estimated as simple mean (not regression) because prompt token range in this dataset is too narrow (557–808) to identify a per-token slope
- Requires extending `trace_parser.py` to extract `api.ARRIVED` events from `llm_request` spans

Note: the scheduler queueing delay (QUEUED → SCHEDULED) is NOT modeled — the simulator produces this internally.

#### Phase 2: Post-decode overhead (α₁, α₂)

Estimates the delay between scheduler completion and response delivery to the client.

```
post_decode_overhead(r) = α₁ + α₂ · output_tokens(r)
minimize ||Xα - y||²   subject to α ≥ 0
```

- Signal: `journey.FINISHED` on `llm_core` span → `api.DEPARTED` on `llm_request` span
- α₁: fixed per-request cost (µs) — response setup, final API processing
- α₂: per-token detokenization cost (µs/token)
- No regularization
- Solver: `scipy.optimize.nnls`
- Requires extending `trace_parser.py` to extract `api.DEPARTED` events from `llm_request` spans

Preliminary measurement on llama-2-7b-general: mean ≈ 10 µs/token total overhead.

#### Phase 3: GPU processing model (β₁–β₇)

Fits the step-time model coefficients on GPU processing time.

```
X_r = [Σ max(t_pf_compute, t_pf_kv),   # β₁
       Σ max(t_dc_compute, t_dc_kv),   # β₂
       Σ t_weight,                       # β₃
       Σ t_tp,                           # β₄
       Σ L,                              # β₅
       Σ batch_size,                     # β₆
       num_active_steps]                 # β₇

minimize ||Xβ - y||² + λ · Σᵢ₌₁⁴ (βᵢ - 1)²   subject to β ≥ 0
```

- Target: `RequestLabel.processing_us` (= FINISHED.ts − SCHEDULED.ts − preemption_gaps). This already excludes post-decode overhead because it ends at FINISHED, not DEPARTED. No subtraction of α₁/α₂ needed.
- Per-request features: sum of per-step basis values along the request's trajectory
- Regularization: Tikhonov on β₁–β₄ only (toward prior of 1.0). β₅–β₇ unregularized.
  - Physics-informed: regularize where we have physics (roofline corrections), unconstrained where we don't (system overheads)
- λ tuned on validation set (grid search, pick lowest validation MSE)
- Solver: constrained QP (`scipy.optimize.lsq_linear` or `minimize` with bounds)

#### Why NNLS over MAPE + gradient-free

The original design (issue #3) proposed MAPE with Nelder-Mead. We switched to MSE + NNLS because:

- `max()` is precomputed on raw basis values, making the regression linear in β
- Request-level aggregation (summing step predictions) preserves linearity
- NNLS is convex → guaranteed global optimum, fast, deterministic
- MSE is unbiased (MAPE biases predictions low)
- Non-negativity is physically correct for all parameters

#### Parameter summary

| Parameter | Phase | What it models | Signal | Constraint | Regularization | Expected |
|-----------|-------|---------------|--------|-----------|----------------|----------|
| α₀ | 1 | API processing overhead (fixed) | QUEUED.ts − ARRIVED.ts | ≥ 0 | None | ~5–7 ms |
| α₁ | 2 | Post-decode per-request overhead | DEPARTED.ts − FINISHED.ts | ≥ 0 | None | ~0–500 µs |
| α₂ | 2 | Post-decode per-token cost (detokenization) | DEPARTED.ts − FINISHED.ts | ≥ 0 | None | ~1–10 µs/tok |
| β₁ | 3 | Prefill roofline correction (1/MFU_prefill) | processing_us | ≥ 0 | λ·(β₁−1)² | 1.5–3.0 |
| β₂ | 3 | Decode roofline correction (1/MFU_decode) | processing_us | ≥ 0 | λ·(β₂−1)² | 5–15 |
| β₃ | 3 | Weight loading correction (1/BW_eff) | processing_us | ≥ 0 | λ·(β₃−1)² | 1.0–3.0 |
| β₄ | 3 | TP communication correction | processing_us | ≥ 0 | λ·(β₄−1)² | 0.5–2.0 |
| β₅ | 3 | Per-layer kernel launch + NCCL latency | processing_us | ≥ 0 | None | ~10–50 µs/layer |
| β₆ | 3 | Per-request CPU scheduling cost | processing_us | ≥ 0 | None | ~50 µs/req |
| β₇ | 3 | Fixed per-step overhead | processing_us | ≥ 0 | None | ~500 µs |

### Stage 5 — Evaluation (`evaluate.py`)

**Planned.** MAPE / MSE / MAE metrics on validate and test splits.

Evaluation targets (each component evaluated independently against its own signal):
- API overhead: predicted α₀ vs observed `QUEUED.ts − ARRIVED.ts`
- Post-decode: predicted (α₁ + α₂·n) vs observed `DEPARTED.ts − FINISHED.ts`
- GPU processing: predicted Σ StepTime vs observed `RequestLabel.processing_us`
- β diagnostic: check fitted values against expected ranges (all β should be hardware constants, not model-dependent — this is the "crossmodel" property)

---

## Data split

Defined in `split.py` (single source of truth, validated on import).

| Split | Experiments | Requests | Purpose |
|-------|------------|----------|---------|
| Train | 10 | ~107K | Fit all parameters |
| Validate | 3 | ~16K | Tune λ (regularization strength) |
| Test | 3 | ~10K | Final evaluation (never touch during fitting) |

Train covers 4 architectures × 3 profiles (general, codegen, roleplay). No reasoning profile in train — reserved for generalization testing. Validate tests cross-profile and overload. Test exercises reasoning/saturation regime.

## Dataset

16 experiments: 4 models × 4 profiles, collected with [inference-perf](https://github.com/kubernetes-sigs/inference-perf) against instrumented vLLM on H100 SXM GPUs.

| Model | TP | Attention | MoE |
|-------|----|-----------|-----|
| Llama-2-7b | 1 | MHA (kv_heads=32) | No |
| Llama-2-70b | 4 | GQA (kv_heads=8) | No |
| Mixtral-8x7B | 2 | GQA (kv_heads=8) | Yes (N=8, k=2) |
| CodeLlama-34b | 2 | GQA (kv_heads=8) | No |
