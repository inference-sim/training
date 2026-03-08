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

Note on ground-truth alignment: `RequestLabel.processing_us` = FINISHED.ts − SCHEDULED.ts − gaps (covers scheduler steps only, no API or post-decode overhead). Decomposition: `prefill_processing_us` = FIRST_TOKEN.ts − SCHEDULED.ts − prefill_gaps and `decode_processing_us` = FINISHED.ts − FIRST_TOKEN.ts − decode_gaps. `RequestLabel.e2e_us` = FINISHED.ts − QUEUED.ts (includes scheduler wait but not API overhead or post-decode). Each α/β is evaluated against its own signal, not against e2e directly.

---

## Pipeline stages

### Stage 1 — Validation (`validate_traces.py`)

**Done.** Validates journey trace integrity across all 13 active experiments.

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
- `RequestLabel`: timing in µs (queueing, ttft, processing, prefill_processing, decode_processing, e2e), preemption count, failed flag.
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

**Done.** Three-phase NNLS fitting with stacked prefill/decode formulation. Uses request-level splitting (SHA-256 hash, 70/15/15) and excludes 3 overload experiments (>10% failure).

#### Timing decomposition

`RequestLabel` includes `prefill_processing_us` and `decode_processing_us` fields, decomposing `processing_us` using the `FIRST_TOKEN` event as the boundary:

```
prefill_processing_us = (FIRST_TOKEN.ts − SCHEDULED.ts − prefill_gaps) · 1e6
decode_processing_us  = (FINISHED.ts − FIRST_TOKEN.ts − decode_gaps) · 1e6
```

where preemption gaps are partitioned by comparing `PREEMPTED.ts` against `FIRST_TOKEN.ts`:
- Prefill gap: `PREEMPTED.ts < FIRST_TOKEN.ts` (preemption occurred before first token)
- Decode gap: `PREEMPTED.ts > FIRST_TOKEN.ts` (preemption occurred after first token)

**Invariant:** `prefill_processing_us + decode_processing_us = processing_us` (exact, by construction — the gap partition is exhaustive and the timestamp arithmetic telescopes).

The decomposition is valid because:
- `FIRST_TOKEN` occurs exactly once per non-failed request (guaranteed by the lifecycle completeness check in Stage 1 and the event model — one first-token event per request)
- Events are sorted by `(step, ts)` before processing, so temporal ordering is reliable
- Preemption during prefill is supported (the state machine re-enters PREFILL on resume if `prefill_complete` is False)

#### Phase 1: API processing overhead (α₀)

Estimates the constant delay between request arrival at the API server and handoff to the engine scheduler.

```
α₀ = mean( QUEUED.ts − ARRIVED.ts )   across training requests   [µs]
```

- Signal: `api.ARRIVED` on `llm_request` span → `journey.QUEUED` on `llm_core` span
- This is API-side work: request parsing, tokenization, validation, engine handoff
- Estimated as simple mean (not regression) because prompt token range in this dataset is too narrow (557–808) to identify a per-token slope
- Requires `parse_api_events()` in `trace_parser.py` to extract `api.ARRIVED` events from `llm_request` spans

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
- Requires `parse_api_events()` in `trace_parser.py` to extract `api.DEPARTED` events from `llm_request` spans

#### Phase 3: GPU processing model (β₁–β₇)

Fits the step-time model coefficients using separate prefill and decode targets.

##### Why separate prefill/decode objectives

The current formulation uses a single target per request: `processing_us = Σ StepTime(s)` across all active steps. This causes β₁ (prefill roofline correction) to be zeroed out because:
- Prefill occupies ~1 step while decode occupies ~247 steps per request
- β₁'s feature column is ~250x smaller in magnitude than β₂'s
- NNLS zeros out β₁ because it has negligible leverage on total processing time

Separating the objective into `MSE(prefill_time) + MSE(decode_time)` gives prefill and decode equal structural weight in the loss function. Each row in the system predicts a quantity whose scale matches its feature magnitudes.

##### Stacked NNLS formulation

For each non-failed request r, compute two feature vectors by summing per-step basis values over the steps where r appears in prefill vs. decode:

```
X_pf(r) = [Σ max(t_pf_compute, t_pf_kv),   # β₁   (prefill roofline, from prefill entries only)
            0,                               # β₂   (decode roofline — not accumulated for prefill)
            Σ t_weight,                       # β₃   (shared, summed over prefill steps of r)
            Σ t_tp,                           # β₄
            Σ L,                              # β₅
            Σ batch_size,                     # β₆
            num_prefill_steps]               # β₇

X_dc(r) = [0,                               # β₁   (prefill roofline — not accumulated for decode)
            Σ max(t_dc_compute, t_dc_kv),   # β₂   (decode roofline, from decode entries only)
            Σ t_weight,                       # β₃   (shared, summed over decode steps of r)
            Σ t_tp,                           # β₄
            Σ L,                              # β₅
            Σ batch_size,                     # β₆
            num_decode_steps]                # β₇
```

Note: β₁ and β₂ use **selective accumulation** — prefill entries only accumulate β₁ (prefill roofline), decode entries only accumulate β₂ (decode roofline). This matches the original single-row formulation semantics and preserves the sum invariant. The shared columns β₃-β₇ are accumulated by any entry in the step.

The stacked system solves a single NNLS problem:

```
⌈ X_pf(r₁) ⌉       ⌈ prefill_processing_us(r₁) ⌉
│ X_pf(r₂) │       │ prefill_processing_us(r₂) │
│   ...     │       │          ...               │
│ X_dc(r₁) │ β  ≈  │ decode_processing_us(r₁)  │
│ X_dc(r₂) │       │ decode_processing_us(r₂)  │
⌊   ...     ⌋       ⌊          ...               ⌋

minimize ||X_stacked · β − y_stacked||² + λ · Σᵢ₌₁⁴ (βᵢ − 1)²   subject to β ≥ 0
```

All 7 β coefficients are **shared** between the prefill and decode rows. This is physically correct: weight loading, TP communication, per-layer overhead, and scheduling costs are properties of the step, not of whether the step is doing prefill or decode.

**Key properties of the stacked formulation:**
- β₁ (prefill roofline) is accumulated ONLY by prefill entries. β₂ (decode roofline) is accumulated ONLY by decode entries. This selective accumulation matches the original single-row formulation.
- In mixed steps (prefill + decode in same step): the prefilling request accumulates β₁ and the shared β₃-β₇; the decoding request accumulates β₂ and the shared β₃-β₇. Both see the same step's shared overhead (weight loading, TP comm, per-layer, scheduling).
- β₃–β₇ are shared: accumulated by any entry (prefill or decode) in each step.

**Invariant:** `X_pf(r) + X_dc(r) = X_total(r)` where `X_total` is the original single-row feature vector. This ensures the stacked formulation is a refinement, not a redefinition, of the original model.

##### Regularization

Tikhonov regularization on β₁–β₄ only (toward prior of 1.0). β₅–β₇ unregularized. Implemented via augmented matrix: append √λ·I₄ₓ₇ rows to X_stacked, √λ·1₄ to y_stacked.

λ tuned on validation set (grid search over [0, 0.01, 0.1, 1, 10, 100], pick lowest validation MSE). Validation MSE is the combined prefill + decode MSE.

Solver: `scipy.optimize.nnls` on the augmented system.

#### Why NNLS over MAPE + gradient-free

The original design (issue #3) proposed MAPE with Nelder-Mead. We switched to MSE + NNLS because:

- `max()` is precomputed on raw basis values, making the regression linear in β
- Request-level aggregation (summing step predictions) preserves linearity
- The stacked prefill/decode system is still a single linear NNLS problem
- NNLS is convex → guaranteed global optimum, fast, deterministic
- MSE is unbiased (MAPE biases predictions low)
- Non-negativity is physically correct for all parameters

#### Diagnostics

Output to `output/fit/`:
- `coefficients.json` — all 10 fitted parameters
- `lambda_tuning.json` — λ vs validation MSE curve
- `residuals.json` — per-experiment residual summary

MSE reported for all three phases independently:
- Phase 1: MSE of α₀ vs observed (QUEUED.ts − ARRIVED.ts)
- Phase 2: MSE of (α₁ + α₂·n) vs observed (DEPARTED.ts − FINISHED.ts)
- Phase 3: combined prefill + decode MSE, and each reported separately

This three-phase reporting ensures each component's prediction quality is visible. A poor α₀ estimate does not mask a good β fit.

#### Parameter summary

| Parameter | Phase | What it models | Signal | Constraint | Regularization | Expected |
|-----------|-------|---------------|--------|-----------|----------------|----------|
| α₀ | 1 | API processing overhead (fixed) | QUEUED.ts − ARRIVED.ts | ≥ 0 | None | ~5–7 ms |
| α₁ | 2 | Post-decode per-request overhead | DEPARTED.ts − FINISHED.ts | ≥ 0 | None | ~0–500 µs |
| α₂ | 2 | Post-decode per-token cost (detokenization) | DEPARTED.ts − FINISHED.ts | ≥ 0 | None | ~1–10 µs/tok |
| β₁ | 3 | Prefill roofline correction (1/MFU_prefill) | prefill_processing_us | ≥ 0 | λ·(β₁−1)² | 1.5–3.0 |
| β₂ | 3 | Decode roofline correction (1/MFU_decode) | decode_processing_us | ≥ 0 | λ·(β₂−1)² | 5–15 |
| β₃ | 3 | Weight loading correction (1/BW_eff) | prefill + decode | ≥ 0 | λ·(β₃−1)² | 1.0–3.0 |
| β₄ | 3 | TP communication correction | prefill + decode | ≥ 0 | λ·(β₄−1)² | 0.5–2.0 |
| β₅ | 3 | Per-layer kernel launch + NCCL latency | prefill + decode | ≥ 0 | None | ~10–50 µs/layer |
| β₆ | 3 | Per-request CPU scheduling cost | prefill + decode | ≥ 0 | None | 50–200 µs/req |
| β₇ | 3 | Fixed per-step overhead | prefill + decode | ≥ 0 | None | 100–2000 µs |

### Stage 5 — Evaluation (`evaluate.py`)

**Done.** Comprehensive evaluation of all 5 prediction targets across 3 data splits with 3 metrics each (45 cells total).

#### Evaluation matrix

5 measures × 3 metrics × 3 splits = 45 cells.

| Measure | Predicted | Observed | Signal |
|---------|-----------|----------|--------|
| Pre-queueing | α₀ (constant) | (QUEUED.ts − ARRIVED.ts) × 1e6 | API overhead per request |
| Post-decode | α₁ + α₂·n | (DEPARTED.ts − FINISHED.ts) × 1e6 | Detokenization per request |
| GPU prefill | X_pf @ β | prefill_processing_us | Prefill step time per request |
| GPU decode | X_dc @ β | decode_processing_us | Decode step time per request |
| GPU combined | (X_pf + X_dc) @ β | processing_us | Total GPU time per request |

Metrics: **MAPE** (%), **RMSE** (µs), **MAE** (µs). Splits: **train**, **validate**, **test**.

#### Alpha data collection

`collect_alpha_data(split_filter)` generalizes the original training-only collection to accept any split (or `None` for all data). For each split, pre-queueing and post-decode metrics are computed from API + journey timestamps.

#### GPU data collection

When `_collect_beta_data()` or `_collect_gpu_eval_data()` calls `build_stacked_feature_matrix()` per experiment and vstacks the results, the combined matrix has the layout `[exp1_pf; exp1_dc; exp2_pf; exp2_dc; ...]`. Each per-experiment block has first-n rows for prefill and last-n rows for decode. To correctly separate prefill and decode predictions, `_collect_gpu_eval_data()` processes each experiment individually, extracting the two halves per experiment before concatenating across experiments.

Combined predictions are `pf_pred + dc_pred` per request, matching the sum invariant `X_pf(r) + X_dc(r) == X_total(r)`.

#### MAPE guard

MAPE excludes observations where `|observed| < 1.0 µs` to avoid division by near-zero. This threshold is safe — real timing values are all > 100 µs.

#### Output

- `output/evaluate/metrics.json` — full 5×3×3 evaluation matrix
- `output/evaluate/metrics.csv` — one row per (measure, split) for visualization
- Printed table to stdout with MAPE (%), RMSE (µs), MAE (µs), n for each cell

#### Public API

`evaluate(coeffs, hw) -> EvaluationResult` — evaluate all 5 measures across train/validate/test splits.

---

## Dataset

16 experiments (13 active + 3 excluded overload): 4 models × 4 profiles, collected with [inference-perf](https://github.com/kubernetes-sigs/inference-perf) against instrumented vLLM on H100 SXM GPUs.

| Model | TP | Attention | MoE |
|-------|----|-----------|-----|
| Llama-2-7b | 1 | MHA (kv_heads=32) | No |
| Llama-2-70b | 4 | GQA (kv_heads=8) | No |
| Mixtral-8x7B | 2 | GQA (kv_heads=8) | Yes (N=8, k=2) |
| CodeLlama-34b | 2 | GQA (kv_heads=8) | No |

### Overload exclusion

3 of the 16 experiments are in the **overload regime** (high failure rates due to preemption cascades and timeouts):

| Experiment | Failure rate |
|------------|-------------|
| llama-2-7b reasoning | 85% |
| llama-2-70b reasoning | 33% |
| mixtral-8x7b reasoning | 69% |

These experiments are excluded from the active dataset. The linear step-time model cannot capture the nonlinear dynamics of overload (preemption cascades amplify latency non-linearly), and including them contaminates validation MSE (the mixtral-reasoning experiment alone contributed 7-second RMSE, dominating the validation signal).

codellama-34b reasoning (0.08% failure, 4796 successful requests) is NOT overloaded — it is a long-request saturation regime where the model has sufficient capacity. It is retained.

**Active dataset: 13 experiments, ~137K successful requests.**

## Data split

Defined in `split.py` (single source of truth, validated on import).

### Request-level splitting

**Split unit: individual requests**, not experiments. Since the fitting pipeline uses teacher-forced reconstruction (real batch compositions, not simulated), per-request features are independent — request N's feature vector does not depend on request M's split assignment. This eliminates the regime bias of experiment-level splitting, where an entire workload profile (reasoning) would be absent from training.

After excluding the 3 overload experiments, split assignment uses a deterministic hash of the request ID (SHA-256 mod 100) for reproducibility across runs and platforms. Ratio: 70% train, 15% validate, 15% test.

| Split | Requests | Purpose |
|-------|----------|---------|
| Train | ~91K (70%) | Fit all 10 parameters |
| Validate | ~19.5K (15%) | Tune λ (regularization strength) |
| Test | ~19.5K (15%) | Final evaluation (never touch during fitting) |

Every split contains requests from all 4 model architectures and all active profiles (general, codegen, roleplay, reasoning). This ensures:
- β₁ (prefill roofline) sees prefill steps from all architectures in training
- β₂ (decode roofline) sees decode steps across MHA, GQA, and MoE
- Validation and test are representative of the full operating range

**Invariants:**
- Every active (non-overload) request appears in exactly one split.
- Split assignment depends only on request ID — deterministic, no randomness.
- The 3 overload experiments contribute zero requests to any split.

Note: experiment-level splitting remains appropriate for Stage 5 (simulator replay evaluation), where the simulator must replay a complete arrival stream. The request-level split is specific to Stage 4 coefficient fitting.
