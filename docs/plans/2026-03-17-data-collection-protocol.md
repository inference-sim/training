# Data Collection Protocol for 9-Parameter Step-Time Model

Specification: [inference-sim/training#5](https://github.com/inference-sim/training/issues/5)
Evaluation: [inference-sim/inference-sim#598](https://github.com/inference-sim/inference-sim/discussions/598)

---

## Goal

Collect training data on H100 SXM to fit 9 coefficients of the extended step-time model:

```
step_time(s) = beta_1 * T_pf_compute + beta_2 * T_dc_compute + beta_3 * T_all_hbm
             + beta_4 * T_tp + beta_5 * T_ep_a2a + beta_6 * T_ep_imbalance
             + beta_7 * T_kv_cpu + beta_8 * L + beta_9 * B
```

Design priorities: maximize identifiability of all 9 coefficients, minimize sample complexity and GPU-hours, avoid data leakage from the evaluation set.

---

## Constraints

1. **No reuse of existing experiments.** The 13 active experiments from the current pipeline are not included.
2. **No overlap with evaluation set.** The evaluation protocol (#598) uses standard workload profiles (general, codegen, roleplay, reasoning, general-lite). Training uses synthetic identifiability-focused workloads only.
3. **H100 SXM only.** Single-node experiments (1-8 GPUs). Multi-node cluster available for parallelism across experiments.
4. **Sub-saturation target.** Most experiments at 30-70% of safe capacity. At most 1-2 intentional overload experiments per model.

---

## Tooling

### Load generation

`blis observe` ([#659](https://github.com/inference-sim/inference-sim/issues/659), [#660](https://github.com/inference-sim/inference-sim/issues/660)): sends workload-spec-driven requests to a real vLLM endpoint at precise arrival times, records client-side timing to TraceV2 CSV.

### Data collection (3 simultaneous streams)

| Stream | Source | Format | Content |
|--------|--------|--------|---------|
| Journey traces | vLLM OTEL (`--enable-journey-tracing`) | JSONL | Per-request lifecycle: QUEUED, SCHEDULED, FIRST_TOKEN, PREEMPTED, FINISHED with `ts.monotonic` and `scheduler.step` |
| KV events | vLLM ZMQ publisher → `kv_events_subscriber.py` | JSONL | Per-step cache operations: BlockStored, BlockRemoved, CacheEviction, TransferInitiated/Completed |
| Client trace | `blis observe` | CSV | Per-request client timing: ArrivalTimeUs, FirstTokenTimeUs, LastTokenTimeUs, InputTokens, OutputTokens, Status |

Journey traces and KV events correlate via `scheduler_step`. Sample rate = 1.0 (100% — training data, not production monitoring).

### Reconstruction pipeline

Existing: `validate_traces.py` -> `reconstruct_steps.py` -> `basis_functions.py` -> `fit_coefficients.py` -> `evaluate.py`.

Modified for issue #5: `basis_functions.py` computes 9 features instead of 7 (see Post-Processing section).

---

## Training Workload Profiles

Six synthetic workload profiles designed for coefficient isolation. These are NOT the standard benchmark profiles used in evaluation.

| ID | Name | Input tokens (mu, sigma) | Output tokens (mu, sigma) | Purpose |
|----|------|--------------------------|---------------------------|---------|
| W1 | prefill-heavy | 3000, 500 | 8, 2 | Isolate beta_1 (prefill compute dominates step time) |
| W2 | decode-heavy | 32, 8 | 1500, 300 | Isolate beta_2 (decode compute dominates) |
| W3 | balanced-short | 256, 64 | 128, 32 | Low per-request overhead, moderate batching |
| W4 | balanced-long | 1024, 256 | 512, 128 | High KV cache pressure, tests beta_3 scaling |
| W5 | batch-stressor | 64, 16 | 32, 8 | Very short requests at high rate -> large B, isolates beta_9 |
| W6 | kv-pressure | 512, 128 | 2048, 512 | Long decode with growing context -> beta_3 KV component, beta_7 offload |

All use Poisson arrivals. Rates are model-specific (calibrated per experiment).

---

## Rate Calibration

Before each experiment, run a single-request calibration probe:

```
decode_ms_per_token = (t_last - t_second) / (n - 2) * 1000
R_seq = 1000 / (decode_ms_per_token * mean_output_tokens)
R_mbt = max_num_batched_tokens / mean_input_tokens / decode_ms_per_token * 1000
safe_rps = 0.5 * min(R_mbt, R_seq)
```

Rate targets as percentage of safe_rps: 30%, 50%, 70%, 100%, 130% (last only for intentional overload).

---

## Post-Hoc Saturation Detection

Five timestamp-based checks on journey traces (no fitted coefficients needed):

| Check | Signal | Threshold |
|-------|--------|-----------|
| 1. Failure rate | Aborted or failed / total | > 5% |
| 2. Queue growth | Slope of (SCHEDULED.ts - QUEUED.ts) over time | Positive slope, R^2 > 0.5 |
| 3. Preemption rate | Requests with >= 1 PREEMPTED / total | > 15% |
| 4. TTFT degradation | Median TTFT last 20% / median TTFT first 20% | Ratio > 2.0 |
| 5. Tail latency blowup | P99 E2E / P50 E2E | Ratio > 10 |

Decision: if >= 2 of 5 checks flag, classify experiment as saturated and exclude from training set.

---

## Phase 1: Experiment Matrix (40 experiments)

### Coefficient identifiability map

| Beta | Feature | Activated by | Isolation strategy |
|------|---------|-------------|-------------------|
| beta_1 | T_pf_compute | Prefill tokens | Prefill-heavy workloads (W1), vary across models |
| beta_2 | T_dc_compute | Decode tokens x context | Decode-heavy workloads (W2), long-context (W6) |
| beta_3 | T_all_hbm | Always (weights + KV) | Vary batch size to shift compute/bandwidth ratio; quantization changes bpw |
| beta_4 | T_tp | TP > 1 | Same model at TP=1/2/4/8 with identical workload |
| beta_5 | T_ep_a2a | EP > 1 | MoE with EP enabled vs TP-only |
| beta_6 | T_ep_imbalance | EP with varied batch | EP at high vs low batch sizes |
| beta_7 | T_kv_cpu | CPU offload enabled | Same model with/without offload |
| beta_8 | L (per-layer) | Varies with model | Models with L in {32, 40, 48, 56, 80} |
| beta_9 | B (per-request) | Varies with batch size | Rate sweep (low -> high load = varied B) |

### Layer 1: Dense Model Diversity (12 experiments)

Identifies beta_1, beta_2, beta_3, beta_8, beta_9 across varied architectures.

| # | Model | TP | Workload | Rate | Identifies |
|---|-------|----|----------|------|-----------|
| 1 | Llama-3.1-8B-Instruct | 1 | W1 (prefill-heavy) | 50% safe | beta_1 dominance |
| 2 | Llama-3.1-8B-Instruct | 1 | W2 (decode-heavy) | 50% safe | beta_2 dominance |
| 3 | Llama-3.1-8B-Instruct | 1 | W5 (batch-stressor) | 70% safe | beta_9 scaling |
| 4 | Qwen3-14B | 1 | W1 (prefill-heavy) | 50% safe | beta_1 at different L/d |
| 5 | Qwen3-14B | 1 | W2 (decode-heavy) | 50% safe | beta_2 at different L/d |
| 6 | Qwen3-14B | 1 | W4 (balanced-long) | 50% safe | beta_3 KV bandwidth |
| 7 | Llama-2-70B-hf | 4 | W1 (prefill-heavy) | 30% safe | beta_1 + beta_4 at TP=4 |
| 8 | Llama-2-70B-hf | 4 | W2 (decode-heavy) | 50% safe | beta_2 + beta_4 at TP=4 |
| 9 | Llama-2-70B-hf | 4 | W6 (kv-pressure) | 50% safe | beta_3 at large model |
| 10 | CodeLlama-34B-Instruct-hf | 2 | W3 (balanced-short) | 50% safe | beta_8 (48 layers vs 32/40/80) |
| 11 | CodeLlama-34B-Instruct-hf | 2 | W4 (balanced-long) | 50% safe | beta_3 at TP=2 |
| 12 | Llama-3.1-8B-Instruct | 1 | W3 (balanced-short) | 30% safe | Low-load baseline |

Model selection rationale: Llama-3.1-8B (L=32, GQA), Qwen3-14B (L=40, GQA, different d_ff), CodeLlama-34B (L=48, GQA), Llama-2-70B (L=80, GQA). Four distinct layer counts break the L correlation with other features.

### Layer 2: TP Isolation (6 experiments)

Same model and workload, vary TP only. Cleanly separates beta_4.

| # | Model | TP | Workload | Rate | Identifies |
|---|-------|----|----------|------|-----------|
| 13 | Qwen3-14B | 1 | W4 (balanced-long) | 50% safe | beta_4=0 reference |
| 14 | Qwen3-14B | 2 | W4 (balanced-long) | 50% safe | beta_4 at TP=2 (point-to-point) |
| 15 | Qwen3-14B | 4 | W4 (balanced-long) | 50% safe | beta_4 at TP=4 (ring all-reduce) |
| 16 | Llama-2-70B-hf | 4 | W3 (balanced-short) | 50% safe | beta_4 at TP=4, different L |
| 17 | Llama-2-70B-hf | 8 | W3 (balanced-short) | 50% safe | beta_4 at TP=8 |
| 18 | Llama-3.1-8B-Instruct | 2 | W4 (balanced-long) | 50% safe | beta_4 at TP=2, small model |

Note: experiment 13 duplicates 6 (same model+workload at TP=1) to provide the zero-TP-comm baseline within the TP sweep.

### Layer 3: MoE + Expert Parallelism (8 experiments)

Identifies beta_5 (EP dispatch) and beta_6 (EP load imbalance).

| # | Model | TP | EP | Workload | Rate | Identifies |
|---|-------|----|-----|----------|------|-----------|
| 19 | Mixtral-8x7B-v0.1 | 2 | 1 | W1 (prefill-heavy) | 50% safe | MoE baseline, no EP |
| 20 | Mixtral-8x7B-v0.1 | 2 | 1 | W2 (decode-heavy) | 50% safe | MoE decode baseline |
| 21 | Mixtral-8x7B-v0.1 | 1 | 2 | W1 (prefill-heavy) | 50% safe | beta_5/beta_6 at EP=2 |
| 22 | Mixtral-8x7B-v0.1 | 1 | 2 | W5 (batch-stressor) | 70% safe | beta_6 at high B (reduces imbalance) |
| 23 | Llama-4-Scout-17B-16E (FP8) | 2 | 1 | W3 (balanced-short) | 50% safe | 16-expert MoE baseline |
| 24 | Llama-4-Scout-17B-16E (FP8) | 1 | 2 | W3 (balanced-short) | 50% safe | beta_5/beta_6 with 16 experts |
| 25 | Mixtral-8x22B-Instruct-v0.1 | 4 | 2 | W4 (balanced-long) | 30% safe | Large MoE with EP |
| 26 | Mixtral-8x22B-Instruct-v0.1 | 8 | 1 | W4 (balanced-long) | 30% safe | Large MoE TP-only |

Key isolation: experiments 19 vs 21 compare TP=2/EP=1 vs TP=1/EP=2 on the same model and workload. The only basis function difference is beta_4 (TP comm) vs beta_5/beta_6 (EP comm).

### Layer 4: Quantization (6 experiments)

Three quantization levels: FP16 (bpw=2.0, eta_quant=1.0), FP8 (bpw=1.0, eta_quant~0.9), INT4/W4A16 (bpw=0.5, eta_quant~0.65).

| # | Model | TP | Quant | Workload | Rate | Identifies |
|---|-------|----|-------|----------|------|-----------|
| 27 | Llama-3.1-8B-Instruct | 1 | FP8 | W1 (prefill-heavy) | 50% safe | eta_quant(FP8) effect on beta_1 |
| 28 | Llama-3.1-8B-Instruct | 1 | FP8 | W2 (decode-heavy) | 50% safe | bpw(FP8) effect on beta_3 |
| 29 | Llama-3.1-8B-Instruct | 1 | INT4 (GPTQ) | W1 (prefill-heavy) | 50% safe | eta_quant(INT4) effect on beta_1 |
| 30 | Llama-3.1-8B-Instruct | 1 | INT4 (GPTQ) | W2 (decode-heavy) | 50% safe | bpw(INT4) effect on beta_3 |
| 31 | Qwen3-14B | 1 | FP8 | W4 (balanced-long) | 50% safe | FP8 at different architecture |
| 32 | CodeLlama-34B-Instruct-hf | 2 | FP8 | W3 (balanced-short) | 50% safe | FP8 at TP=2 |

Comparison pairs: exp 1 (FP16) vs 27 (FP8) vs 29 (INT4) isolate eta_quant; exp 2 (FP16) vs 28 (FP8) vs 30 (INT4) isolate bpw on T_all_hbm. Scout-17B (experiments 23-24) provides additional FP8 MoE data.

### Layer 5: CPU KV Offload (4 experiments)

Identifies beta_7 (CPU-to-GPU KV transfer bandwidth).

| # | Model | TP | Offload | Workload | Rate | Identifies |
|---|-------|----|---------|----------|------|-----------|
| 33 | Llama-3.1-8B-Instruct | 1 | Yes | W6 (kv-pressure) | 50% safe | beta_7 activation |
| 34 | Llama-3.1-8B-Instruct | 1 | No | W6 (kv-pressure) | 50% safe | beta_7=0 reference |
| 35 | Qwen3-14B | 1 | Yes | W6 (kv-pressure) | 50% safe | beta_7 at different model |
| 36 | Qwen3-14B | 1 | No | W6 (kv-pressure) | 50% safe | beta_7=0 reference |

KV-pressure workload (long decode) maximizes KV blocks transferred, making beta_7's signal strongest.

### Layer 6: Rate Sweep (4 experiments)

Validates beta_9 scaling with batch size and tests saturation boundary.

| # | Model | TP | Workload | Rate | Identifies |
|---|-------|----|----------|------|-----------|
| 37 | Llama-3.1-8B-Instruct | 1 | W3 (balanced-short) | 30% safe | Low batch size |
| 38 | Llama-3.1-8B-Instruct | 1 | W3 (balanced-short) | 70% safe | Medium batch size |
| 39 | Llama-3.1-8B-Instruct | 1 | W3 (balanced-short) | 100% safe | High batch size |
| 40 | Llama-3.1-8B-Instruct | 1 | W3 (balanced-short) | 130% safe | Over-saturation (intentional) |

### Summary

| Layer | Experiments | GPUs/exp | Purpose |
|-------|------------|----------|---------|
| 1: Dense diversity | 12 | 1-4 | beta_1, beta_2, beta_3, beta_8, beta_9 |
| 2: TP isolation | 6 | 1-8 | beta_4 |
| 3: MoE + EP | 8 | 2-8 | beta_5, beta_6 |
| 4: Quantization | 6 | 1-2 | bpw, eta_quant effects |
| 5: CPU offload | 4 | 1 | beta_7 |
| 6: Rate sweep | 4 | 1 | beta_9 + saturation boundary |
| **Total** | **40** | | |

Estimated GPU-hours: 40 experiments x ~15 min avg x ~2.5 avg GPUs = ~25 GPU-hours.

---

## Phase 2: Adaptive Gap-Filling

After Phase 1 fitting:

1. **Condition number.** Compute kappa(X) of the stacked feature matrix. If kappa > 1000, near-collinear features exist.
2. **Bootstrap confidence.** Resample requests with replacement, fit NNLS 100 times. Report CV = std(beta_i) / beta_i for each coefficient.
3. **Design targeted experiments.** For each coefficient with CV > 50%, design 2-3 experiments maximizing that coefficient's feature variance relative to others.
4. **Iterate** until all coefficients have CV < 30% or total reaches ~50 experiments.

Expected: 5-10 additional experiments in Phase 2.

---

## Per-Experiment Execution Protocol

### vLLM server configuration

```bash
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_HF_ID \
  --tensor-parallel-size $TP \
  --enable-journey-tracing \
  --otlp-traces-endpoint http://localhost:4317 \
  --journey-tracing-sample-rate 1.0 \
  --kv-events-publisher-addr tcp://*:5557 \
  --kv-events-replay-addr tcp://*:5558 \
  ${QUANT_FLAGS} \
  ${OFFLOAD_FLAGS} \
  ${EP_FLAGS}
```

Sample rate = 1.0 (100%). Training data requires complete step reconstruction.

### Execution sequence

```
 1. Start OTEL collector (file exporter to traces.json)
 2. Start KV events subscriber -> kv_events.jsonl
 3. Start vLLM server with experiment config
 4. Wait for server ready (health check)
 5. Run calibration probe (1 request) -> compute safe_rps -> calibration.json
 6. Run blis observe with workload-spec YAML at target rate
 7. Wait for drain (all in-flight requests complete)
 8. Shutdown vLLM server
 9. Stop KV events subscriber
10. Stop OTEL collector
11. Run saturation detection -> saturation.json
12. Package all artifacts into experiment directory
```

### Output directory structure

```
experiments/
  <experiment-name>/
    traces.json          # Journey traces (OTEL spans)
    kv_events.jsonl      # KV cache events
    observe-trace.csv    # blis observe client-side timing
    exp-config.yaml      # Full experiment configuration
    workload-spec.yaml   # blis observe workload spec
    calibration.json     # Rate calibration results
    saturation.json      # Post-hoc saturation check results
```

### Experiment naming

```
<model-short>-tp<N>-ep<N>-<quant>-<workload>-r<rate-pct>
```

Examples: `llama31-8b-tp1-ep1-fp16-W1-r50`, `mixtral-8x7b-tp1-ep2-fp16-W1-r50`, `llama31-8b-tp1-ep1-fp8-W2-r50`.

### Experiment configuration schema

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
tp: 1
ep: 1
max_num_batched_tokens: 2048
bpw: 2.0            # Bytes per weight (FP16=2.0, FP8=1.0, INT4=0.5)
eta_quant: 1.0       # Quantization throughput factor
kv_dtype_bytes: 2    # KV cache precision (FP16=2, FP8=1)
cpu_offload: false
pcie_bandwidth_gbps: 64
```

---

## Post-Processing Pipeline

```
Raw Data -> Stage 0: Saturation Filter (NEW)
         -> Stage 1: Validation (existing)
         -> Stage 2: Step Reconstruction (existing)
         -> Stage 3: Basis Functions (MODIFIED for 9 terms)
         -> Stage 4: Coefficient Fitting (MODIFIED for 9 beta)
         -> Stage 5: Evaluation (MODIFIED for per-experiment/per-model reporting)
```

### Basis function changes (Stage 3)

| Current 7-term | New 9-term | Change |
|----------------|-----------|--------|
| max(T_pf_compute, T_pf_kv) | T_pf_compute | Separate, no roofline max |
| max(T_dc_compute, T_dc_kv) | T_dc_compute | Separate, no roofline max |
| T_weight | T_all_hbm = T_weight + T_pf_kv + T_dc_kv | Unified memory traffic |
| T_tp | T_tp | Unchanged |
| -- | T_ep_a2a | NEW |
| -- | T_ep_imbalance | NEW |
| -- | T_kv_cpu | NEW |
| L | L | Unchanged |
| B | B | Unchanged |

The max() operations are removed from the basis functions. T_pf_compute and T_all_hbm become separate features; NNLS learns their relative weights through beta_1 and beta_3.

### Stacked matrix structure (Stage 4)

Same 2-rows-per-request structure as current fitting. Each request r produces:

```
X_pf(r) = [sum T_pf_compute,   # beta_1 (prefill only)
            0,                  # beta_2 (decode only)
            sum T_all_hbm,      # beta_3 (shared)
            sum T_tp,           # beta_4 (shared)
            sum T_ep_a2a,       # beta_5 (shared)
            sum T_ep_imbalance, # beta_6 (shared)
            sum T_kv_cpu,       # beta_7 (shared)
            sum L,              # beta_8 (shared)
            num_pf_steps]       # beta_9

X_dc(r) = [0,                  # beta_1 (prefill only)
            sum T_dc_compute,   # beta_2 (decode only)
            sum T_all_hbm,      # beta_3 (shared)
            sum T_tp,           # beta_4 (shared)
            sum T_ep_a2a,       # beta_5 (shared)
            sum T_ep_imbalance, # beta_6 (shared)
            sum T_kv_cpu,       # beta_7 (shared)
            sum L,              # beta_8 (shared)
            num_dc_steps]       # beta_9
```

Sum invariant preserved: X_pf(r) + X_dc(r) = X_total(r) for all 9 features.
