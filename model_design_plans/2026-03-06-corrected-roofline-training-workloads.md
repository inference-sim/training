# Training Workloads for Corrected Roofline β/α Fitting

**Companion to:** `2026-03-06-corrected-roofline-features-design.md` §12
**Infrastructure:** [tektonc-data-collection](https://github.com/inference-sim/tektonc-data-collection) + [instrumented vLLM](https://github.com/inference-sim/vllm)

---

## Overview

5 workload profiles, each targeting specific β identifiability requirements. Run each profile per model/TP config. Collect `step.BATCH_SUMMARY` + `REQUEST_SNAPSHOT` from instrumented vLLM.

**Models:** Llama-3.1-8B, Llama-3.1-70B (dense), Mixtral-8x22B (MoE), DeepSeek-V3 (MoE+MLA)
**TP configs:** 1, 2, 4 per model (where GPU memory permits)

---

## W1: Prefill Length Sweep

**Targets:** β₁ (prefill compute/bandwidth), β₇ (intercept anchor)
**Why:** Varies F_pf_compute and F_pf_kv across orders of magnitude while F_dc_* = 0.

Low rate, single-request batches, no decode. Vary input length.

```yaml
# Run 6 times with question_len ∈ {128, 512, 1024, 2048, 4096, 8192}
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 0.5          # Low rate → B=1 batches (one prefill at a time)
      duration: 120       # 2 min per length → ~60 steps each
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 128     # SWEEP: {128, 512, 1024, 2048, 4096, 8192}
    output_len: 1         # Minimal decode — effectively pure prefill measurement
```

**Expected measurements:** 6 × ~60 = ~360 steps. F_pf_compute varies ~64× across sweep.

---

## W2: Decode Context Sweep

**Targets:** β₂ (decode compute/bandwidth), β₇ (intercept anchor)
**Why:** Varies decode KV bandwidth (dominant decode cost) by varying context length.

Short prefill to seed context, then long generation. Vary prompt length to control decode context.

```yaml
# Run 6 times with question_len ∈ {128, 512, 1024, 2048, 4096, 8192}
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 0.5          # Low rate → B=1 decode-dominated batches
      duration: 120
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 128     # SWEEP: {128, 512, 1024, 2048, 4096, 8192}
    output_len: 512       # Long generation → many decode steps per request
```

**Expected measurements:** 6 × ~500 decode steps. Context S grows from question_len to question_len+512.

---

## W3: Batch Size Scaling

**Targets:** β₂ vs β₆ separation, β₄ (N_eff for MoE models)
**Why:** Varies B while holding per-request shape constant. Separates CPU overhead (β₆ · B) from decode phase (β₂). For MoE models, N_eff = min(N, B·k) transitions from k to N.

Fixed prompt/output length, vary rate to control steady-state batch size.

```yaml
# Run 7 times with rate ∈ {1, 4, 8, 16, 32, 64, 128}
# Higher rate → larger steady-state batches
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 8.0           # SWEEP: {1, 4, 8, 16, 32, 64, 128}
      duration: 120
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 256     # Fixed moderate prompt
    output_len: 256       # Fixed moderate generation
```

**Expected measurements:** 7 × ~200 steps. B ranges from 1 to ~128 across sweep. MoE models: N_eff transitions visible in step.BATCH_SUMMARY token counts.

---

## W4: Prefill/Decode Ratio Mix

**Targets:** β₁ vs β₂ separation
**Why:** At fixed batch size, varies the fraction of prefill vs decode work. This is the key workload for separating compute-phase corrections.

Use multiple concurrent clients with different output lengths to control the prefill:decode mix in steady-state batches.

```yaml
# Profile A: Prefill-heavy (short outputs → requests cycle fast, mostly prefilling)
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 32.0
      duration: 120
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 1024    # Long prompts
    output_len: 16        # Very short output → requests finish fast → high prefill fraction
```

```yaml
# Profile B: Balanced
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 32.0
      duration: 120
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 256
    output_len: 256
```

```yaml
# Profile C: Decode-heavy (short prompts, long outputs → mostly decoding)
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 32.0
      duration: 120
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 64      # Short prompts
    output_len: 1024      # Long output → sustained decode
```

**Expected measurements:** 3 × ~200 steps. Prefill token fraction in batches varies from ~90% (A) to ~10% (C).

---

## W5: TP Communication Scaling

**Targets:** β₅ (TP all-reduce)
**Why:** F_tp = 0 at TP=1, nonzero at TP≥2. Varies message size via batch size at each TP config.

Same workload as W3, but **run at each TP config** (TP=1,2,4). The difference in step time between TP configs at matched batch sizes isolates F_tp.

```yaml
# Same spec as W3, run at TP=1, TP=2, TP=4
# Compare step times at same B across TP configs
version: "1"
seed: 42
inference_perf:
  stages:
    - rate: 16.0          # SWEEP rate: {1, 4, 8, 16, 32, 64}
      duration: 120
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 256
    output_len: 256
```

**Expected measurements:** 6 rates × 3 TP configs × ~200 steps = ~3600 steps. β₅ identified from the TP>1 vs TP=1 delta at matched batch sizes.

---

## Summary

| Profile | Sweeps | Steps/model/TP | Primary β targets |
|---------|--------|----------------|-------------------|
| W1: Prefill length | 6 input lengths | ~360 | β₁, β₇ |
| W2: Decode context | 6 context lengths | ~3000 | β₂, β₇ |
| W3: Batch scaling | 7 rates | ~1400 | β₂ vs β₆, β₄ |
| W4: Pf/dc ratio | 3 profiles | ~600 | β₁ vs β₂ |
| W5: TP scaling | 6 rates × 3 TP | ~3600 | β₅ |

**Total per model family:** ~9000 step measurements (~3 hours GPU time at ~1 step/sec).
**Full dataset (4 model families):** ~36,000 steps. Well above the 450 minimum for cross-model fitting.
