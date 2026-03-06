# Training Workloads for Corrected Roofline β/α Fitting

**Companion to:** `2026-03-06-corrected-roofline-features-design.md` §12
**Infrastructure:** [tektonc-data-collection](https://github.com/inference-sim/tektonc-data-collection) + [instrumented vLLM](https://github.com/inference-sim/vllm)

---

## Overview

5 workload profiles, each targeting specific β identifiability requirements. Run each profile per model/TP config. Collect `step.BATCH_SUMMARY` + `REQUEST_SNAPSHOT` from instrumented vLLM.

**Models:** Llama-3.1-8B, Llama-3.1-70B (dense), Mixtral-8x22B (MoE), DeepSeek-V3 (MoE+MLA)
**TP configs:** 1, 2, 4 per model (where GPU memory permits)

**Data processing note:** At low request rates (W1, W2), vLLM's scheduler may execute idle polling iterations between requests. Filter out empty steps (batch_size=0) from the collected `step.BATCH_SUMMARY` data before training.

---

## W1: Prefill Length Sweep

**Targets:** β₁ (prefill compute/bandwidth), β₇ (intercept anchor)
**Why:** Varies F_pf_compute and F_pf_kv across orders of magnitude while F_dc_* = 0.

Low rate, single-request batches, no decode. Vary input length.

```yaml
# Run 6 times with question_len ∈ {128, 512, 1024, 2048, 4096, 8192}
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 0.5            # Low rate → B=1 batches (one prefill at a time)
      duration: 120         # 2 min per length → ~60 steps each
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true          # Critical: prevents early EOS truncation
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 128       # SWEEP: {128, 512, 1024, 2048, 4096, 8192}
    output_len: 1           # Minimal decode — effectively pure prefill measurement
```

**Expected measurements:** 6 × ~60 = ~360 steps. F_pf_compute varies ~64× across sweep. With output_len=1, vLLM completes the request in a single forward pass (prefill generates the first token by sampling), so no separate decode step is measured.

---

## W2: Decode Context Sweep

**Targets:** β₂ (decode compute/bandwidth), β₇ (intercept anchor)
**Why:** Varies decode KV bandwidth (dominant decode cost) by varying context length.

Short prefill to seed context, then long generation. Vary prompt length to control decode context.

```yaml
# Run 6 times with question_len ∈ {128, 512, 1024, 2048, 4096, 8192}
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 0.5            # Low rate → B=1 decode-dominated batches
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true          # Ensures full 512 output tokens are generated
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 128       # SWEEP: {128, 512, 1024, 2048, 4096, 8192}
    output_len: 512         # Long generation → many decode steps per request
```

**Expected measurements:** At rate=0.5 for 120s, ~60 requests arrive. Each request produces ~512 decode steps, so ~30,000 decode steps per sweep point (~180,000 total across 6 sweep points). Context S grows from question_len to question_len+512 within each request. At higher question_len values (e.g., 8192), decode step times increase and B may briefly reach 2 if a new request arrives before the previous completes.

---

## W3: Batch Size Scaling

**Targets:** β₂ vs β₆ separation, β₄ (N_eff for MoE models)
**Why:** Varies B while holding per-request shape constant. Separates CPU overhead (β₆ · B) from decode phase (β₂). For MoE models, N_eff = min(N, B·k) transitions from k to N.

Fixed prompt/output length, vary rate to control steady-state batch size. Note: the mapping from rate to batch size is indirect and model-dependent (B ≈ rate × per-request service time). Actual batch sizes should be read from the `step.BATCH_SUMMARY` data post-hoc.

```yaml
# Run 7 times with rate ∈ {1, 4, 8, 16, 32, 64, 128}
# Higher rate → larger steady-state batches
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 8.0             # SWEEP: {1, 4, 8, 16, 32, 64, 128}
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 256       # Fixed moderate prompt
    output_len: 256         # Fixed moderate generation
```

**Expected measurements:** 7 sweep points. Steady-state B depends on model throughput; at rate=128 with ~0.5s service time, expect B ≈ 64 (not 128). MoE models: N_eff transitions visible in step.BATCH_SUMMARY token counts.

---

## W4: Prefill/Decode Ratio Mix

**Targets:** β₁ vs β₂ separation
**Why:** At fixed batch size, varies the fraction of prefill vs decode work. This is the key workload for separating compute-phase corrections.

Use high request rate with different output lengths to control the prefill:decode mix in steady-state batches. Note: `num_unique_system_prompts` and `num_users_per_system_prompt` control prompt data diversity, not concurrency — concurrency is determined by rate and request service time.

```yaml
# Profile A: Prefill-heavy (short outputs → requests cycle fast, high prefill fraction)
# With output_len=4, requests complete in ~4 steps. At rate=128, ~128 new prefill
# requests arrive per second, each generating only 4 decode steps before completing.
# Aggregate token mix is heavily prefill-dominated.
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 128.0           # High rate needed because requests are short-lived
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 1024      # Long prompts
    output_len: 4           # Very short output → requests finish fast → high prefill fraction
```

```yaml
# Profile B: Balanced
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 32.0
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 256
    output_len: 256
```

```yaml
# Profile C: Decode-heavy (short prompts, long outputs → mostly decoding)
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 32.0
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 64        # Short prompts
    output_len: 1024        # Long output → sustained decode
```

**Expected measurements:** 3 profiles × ~200 steps each. The per-step prefill:decode token ratio varies across profiles — exact fractions depend on model throughput and should be measured from `step.BATCH_SUMMARY` data rather than assumed from config. Profile A uses rate=128 with output_len=4 (instead of rate=32 with output_len=16) to ensure sufficient concurrent requests for meaningful batch mixing — short-lived requests at low rates yield very small batches with little mixing.

---

## W5: TP Communication Scaling

**Targets:** β₅ (TP all-reduce)
**Why:** F_tp = 0 at TP=1, nonzero at TP≥2. Varies message size via batch size at each TP config.

Same workload as W3, but **run at each TP config** (TP=1,2,4). The difference in step time between TP configs at matched batch sizes isolates F_tp. Note: batch size matching must be done post-hoc (same rate may yield different B at different TP configs due to throughput differences).

```yaml
# Same spec as W3, run at TP=1, TP=2, TP=4
# Compare step times at same B across TP configs
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 16.0            # SWEEP rate: {1, 4, 8, 16, 32, 64}
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 256
    output_len: 256
```

**Expected measurements:** 6 rates × 3 TP configs × variable steps = ~3600 steps total. β₅ identified from the TP>1 vs TP=1 delta at matched batch sizes.

---

## Summary

| Profile | Sweeps | Steps/model/TP | Primary β targets |
|---------|--------|----------------|-------------------|
| W1: Prefill length | 6 input lengths | ~360 | β₁, β₇ |
| W2: Decode context | 6 context lengths | ~180,000 | β₂, β₇ |
| W3: Batch scaling | 7 rates | ~1400 | β₂ vs β₆, β₄ |
| W4: Pf/dc ratio | 3 profiles | ~600 | β₁ vs β₂ |
| W5: TP scaling | 6 rates × 3 TP | ~3600 | β₅ |

**Total per model family:** W2 dominates with ~180K decode steps; other profiles contribute ~6K. Total useful data points depend on sampling/filtering strategy — the 450 minimum for cross-model fitting is easily exceeded.
**Full dataset (4 model families):** Well above minimum. The large W2 dataset provides dense coverage of the decode context dimension.
