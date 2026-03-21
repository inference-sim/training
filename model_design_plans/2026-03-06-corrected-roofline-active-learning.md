# Active Learning Pipeline for Roofline Model

**Goal:** Iteratively improve β fitting accuracy with minimal additional GPU time, instead of collecting all data upfront.

**Depends on:**
- [`corrected-roofline-features-design.md`](2026-03-06-corrected-roofline-features-design.md) — formula definitions
- [`corrected-roofline-training-workloads.md`](2026-03-06-corrected-roofline-training-workloads.md) — workload specs (W1–W4 training, V1 validation)
- [`corrected-roofline-data-collection-plan.md`](2026-03-06-corrected-roofline-data-collection-plan.md) — 99-run initial collection plan

---

## Overview

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐     ┌────────────────┐
│ Collect data │────▶│ Fit βs    │────▶│ Diagnose     │────▶│ Target next    │──┐
│ (initial or  │     │ + validate│     │ weaknesses   │     │ experiments    │  │
│  targeted)   │     │           │     │              │     │                │  │
└─────────────┘     └───────────┘     └──────────────┘     └────────────────┘  │
       ▲                                                                        │
       └────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Bootstrap

Run the [99-run initial collection](2026-03-06-corrected-roofline-data-collection-plan.md):
- **90 training runs** — 6 (model, TP) combos × 15 runs each (W1–W4)
- **9 validation runs** — 3 held-out models × 3 runs each (V1)

Fit β₁–β₇ on training data. Evaluate on validation data. This gives a working model and a baseline error.

---

## Phase 1: Diagnose

After each fit, run three automated diagnostics:

### 1.1 β health check

| Condition | Diagnosis | Action |
|-----------|-----------|--------|
| Any β > 5.0 | Feature is a poor physics prior | Investigate formula; may need structural change |
| Any β < 0.0 | Feature has wrong sign | Check for collinearity or data issue |
| β₄ < 0.5 | N_eff overestimates loaded experts | Switch to group-aware or birthday-problem N_eff |
| β₃ CI width > 2× its mean | Static weight loading poorly identified | Need more model diversity (add a mid-size dense model) |
| β₅ CI width > 2× its mean | TP comms poorly identified | Need more TP data at varied batch sizes |
| β₆ CI width > 2× its mean | CPU overhead poorly identified | Need more prefill-heavy batches (high T_pf, low B) |

### 1.2 Residual analysis

Compute prediction error on both validation types:
- **Held-out models** (V1 data from CodeLlama-34B, Llama-2-7B, Mixtral-8x7B) — tests cross-model generalization
- **Held-out batches** (20% random step sample from training data) — tests within-model interpolation

Bucket residuals by:

| Bucket | What systematic error reveals |
|--------|------------------------------|
| By model | β₃ mis-fit (weight loading) or missing architecture feature |
| By phase (prefill vs decode fraction) | β₁/β₂ imbalance |
| By batch size | β₆ (CPU overhead) or N_eff error for MoE |
| By context length | KV bandwidth feature gap (paged attention scatter) |
| By TP config | β₅ (TP comms) error |

Flag any bucket with >15% MAPE as needing targeted data.

### 1.3 Leverage analysis

For each data point, compute its leverage (influence on β estimates). Identify:
- **High-leverage gaps:** Regions of the feature space with no data but high potential to change βs (e.g., no data at B=1 for MoE models → β₄ anchor missing).
- **Redundant data:** Regions with many low-leverage points (e.g., thousands of mid-context decode steps all saying the same thing). These can be downsampled in future fits.

---

## Phase 2: Target next experiments

Based on diagnosis, select from this menu. All use the same YAML structure as the [workload specs](2026-03-06-corrected-roofline-training-workloads.md) with adjusted sweep values.

### 2.1 Model diversity (addresses β₃, β₇ separation)

Promote a validation model to training. Run the full 15-run training sweep on it.

Priority order:
1. CodeLlama-34B — fills the gap between 8B and 70B weight scales
2. Mixtral-8x7B — validates β₄ generalization across expert configurations
3. DeepSeek-V2 — validates MLA feature generalization

**Cost:** 15 runs per new model/TP combo (~30 min).

### 2.2 Range extension (addresses high-residual buckets)

| Gap identified | Targeted experiment |
|---------------|-------------------|
| Long context (>8K) | W2 with question_len ∈ {16K, 32K, 64K} |
| Very large batches | W3 with rate ∈ {256, 512} |
| Small chunked prefill | W1 with question_len ∈ {16, 32, 64} |
| High TP | W3 at TP=8 |

**Cost:** 3–4 runs per gap, ~8 min each.

### 2.3 Adversarial validation (addresses overconfident predictions)

Run on a held-out model at configurations designed to stress specific simplifications:

| Stress test | Simplification tested | Config |
|-------------|----------------------|--------|
| B=1, question_len=128K | Decode bandwidth extrapolation | W2 at 128K |
| B=128, short prompts on MoE | N_eff saturation | W3 at rate=256 on Mixtral |
| Mixed extreme context variance | Batch-aggregated attention (S7) | Custom: question_len mix of 32K and 128 |

If error >20%, the simplification being stressed needs revisiting.

**Cost:** 3 runs, ~6 min total.

---

## Phase 3: Refit and repeat

1. Add new data to the training set
2. Refit β₁–β₇
3. Re-run Phase 1 diagnostics
4. Stop when: all βs in [0.3, 5.0], MAPE <10% on validation set, no bucket >15% MAPE

---

## Triggers for new collection rounds

Beyond the iterative loop, certain events should trigger data collection:

| Trigger | What to collect |
|---------|----------------|
| **New model deployed to production** | V1 sweep (3 runs) at production TP. Compare predicted vs actual. If MAPE >10%, add full 15-run training sweep |
| **New GPU hardware** | Few-shot calibration: 1 model × 10 batch sizes (§9 of features design). Estimate η_c, η_m scalars |
| **vLLM major version upgrade** | Re-run V1 on 1 model. Scheduling/kernel changes may shift β₆, β₇ |
| **β drift detected in production** | Compare live step times to predictions over a rolling window. If drift >10% for >1 hour, trigger targeted collection at the drifting batch composition |
| **New architecture class** (e.g., encoder-decoder, sliding window) | Requires new features — design doc update first, then full collection for the new architecture |

---

## Budget guideline

| Round | Runs | GPU hours | Purpose |
|-------|:----:|:---------:|---------|
| Initial (Phase 0) | 99 | ~3.3 | Bootstrap β fitting (90 training + 9 validation) |
| Iteration 1 | 15–30 | ~0.5–1 | Fill gaps from Phase 1 diagnosis |
| Iteration 2 | 5–10 | ~0.2–0.3 | Targeted fixes for remaining buckets |
| Per new model (ongoing) | 3–15 | ~0.1–0.5 | V1 to validate, or full sweep to extend |
| Per new GPU (ongoing) | 10 | ~0.3 | Few-shot calibration |

Expect 2–3 iterations to reach <10% MAPE across all validation models. Total: ~5–6 GPU hours including the initial round.
