# Experiment Plan: Roofline β Fitting Data Collection

**Workload specs:** [`corrected-roofline-training-workloads.md`](2026-03-06-corrected-roofline-training-workloads.md)
**Active learning:** [`corrected-roofline-active-learning.md`](2026-03-06-corrected-roofline-active-learning.md)

---

## Model × TP Matrix

### Training (fit βs on these)

| Model | TP=1 | TP=2 | TP=4 | Purpose |
|-------|:----:|:----:|:----:|---------|
| Llama-3.1-8B | ✅ | ✅ | ✅ | Small dense; 3 TP points anchor β₅ |
| Llama-3.1-70B | — | — | ✅ | Large dense (separates β₃ from β₇) |
| Mixtral-8x22B | — | — | ✅ | MoE (exercises β₄) |
| DeepSeek-V3 | — | — | ✅ | MoE+MLA (absorbed attention, compressed KV) |

**6 training combos.** Llama-8B at TP=1,2,4 gives 3 TP data points — needed because AllReduce algorithm may differ between TP=2 (one-shot) and TP=4 (two-shot/ring).

### Validation (held out — never used for fitting)

| Model | TP | Why |
|-------|:--:|-----|
| CodeLlama-34B | 1 | Mid-size dense — interpolation between 8B and 70B |
| Llama-2-7B | 1 | Same scale as 8B, different generation — tests β transfer |
| Mixtral-8x7B | 4 | Different expert count/size — tests β₄ generalization |

**3 validation combos.**

---

## Runs Per Combo

### Training sweep (15 runs per combo)

| Profile | Sweep parameter | Values | `output_len` | What each step measures | Runs |
|---------|----------------|--------|:---:|---|:----:|
| W1: Prefill sweep | `question_len` | 128, 1K, 4K, 8K | **1** | Pure prefill — `question_len` controls how many tokens the GPU prefills per step | 4 |
| W2: Decode context | `question_len` | 128, 1K, 4K, 8K | **512** | Pure decode — `question_len` controls the starting KV cache context length that each decode step must read | 4 |
| W3: Batch scaling | `rate` | 1, 8, 32, 64, 128 | 256 | Mixed — fixed `question_len=256`; varies concurrent requests via rate | 5 |
| W4: Pf/dc ratio | profile | A, C | 4 / 1024 | Mixed — prefill-heavy (A) vs decode-heavy (C) | 2 |
| | | | | **Total per combo** | **15** |

W4-B (balanced) dropped — redundant with W3 at moderate rate. W5 subsumed by running W3 at each TP config.

### Validation sweep (3 runs per combo)

| Profile | Sweep value | Runs |
|---------|------------|:----:|
| W2 | question_len=1024 | 1 |
| W3 | rate ∈ {1, 64} | 2 |
| | **Total per combo** | **3** |

Covers decode bandwidth + batch scaling. Prefill coverage comes from W3 mixed batches.

### Within-model validation

Hold out 20% of training steps by random sampling (not by sweep point) for within-model error estimation.

---

## Full Experiment List

**Total: 99 runs** (90 training + 9 validation). Each run is 2 minutes → **~3.3 hours sequential**.

### Training

| Combo | Runs |
|-------|:----:|
| Llama-3.1-8B — TP=1 | 15 |
| Llama-3.1-8B — TP=2 | 15 |
| Llama-3.1-8B — TP=4 | 15 |
| Llama-3.1-70B — TP=4 | 15 |
| Mixtral-8x22B — TP=4 | 15 |
| DeepSeek-V3 — TP=4 | 15 |

Each combo runs:

| # | Profile | Sweep value |
|---|---------|------------|
| 1–4 | W1 | question_len ∈ {128, 1024, 4096, 8192} |
| 5–8 | W2 | question_len ∈ {128, 1024, 4096, 8192} |
| 9–13 | W3 | rate ∈ {1, 8, 32, 64, 128} |
| 14 | W4-A | rate=128, question_len=1024, output_len=4 |
| 15 | W4-C | rate=32, question_len=64, output_len=1024 |

DeepSeek-V3: W3 at rate=128 may hit KV cache limits — reduce if OOM.

### Validation

| Combo | Runs |
|-------|:----:|
| CodeLlama-34B — TP=1 | 3 |
| Llama-2-7B — TP=1 | 3 |
| Mixtral-8x7B — TP=4 | 3 |

Each combo runs:

| # | Profile | Sweep value |
|---|---------|------------|
| 1 | W2 | question_len=1024 |
| 2–3 | W3 | rate ∈ {1, 64} |

---

## β Identifiability Checklist

| β | Required data | Covered by |
|---|--------------|------------|
| β₁ (prefill) | Varied prefill token count | W1 (4 lengths), W4-A vs W4-C |
| β₂ (decode) | Varied decode context | W2 (4 lengths), W4-C |
| β₃ (static weights) | ≥3 models with different weight sizes | Llama-8B, Llama-70B, Mixtral, DeepSeek |
| β₄ (MoE weights) | MoE models with varied B | W3 on Mixtral + DeepSeek |
| β₅ (TP comms) | TP>1 vs TP=1 at matched B | Llama-8B at TP=1,2,4 (W3 across TP) |
| β₆ (CPU overhead) | Varied B independent of tokens | W3 vs W1 (large T_pf, small B) |
| β₇ (intercept) | Small-B anchors, cross-model | W1 + W2 at rate=0.5 (B=1) |

---

## Execution Notes

- All configs require `server.ignore_eos: true` (already in workload specs).
- Filter empty steps (`batch_size=0`) from collected data before training.
- Actual batch sizes are determined by model throughput, not rate — record from `step.BATCH_SUMMARY` post-hoc.
- Run Llama-8B first (cheapest) to validate the pipeline end-to-end before larger models.
- If β confidence intervals are wide after initial fit, add runs per the [active learning pipeline](2026-03-06-corrected-roofline-active-learning.md).
