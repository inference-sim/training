# Experiment Plan: Roofline β Fitting Data Collection

**Workload specs:** [`corrected-roofline-training-workloads.md`](2026-03-06-corrected-roofline-training-workloads.md)

---

## Model × TP Matrix

| Model | TP=1 | TP=2 | TP=4 |
|-------|:----:|:----:|:----:|
| Llama-3.1-8B | ✅ | ✅ | ✅ |
| Llama-3.1-70B | — | ✅ | ✅ |
| Mixtral-8x22B | — | — | ✅ |
| DeepSeek-V3 | — | — | ✅ |

**10 combos.** Llama-8B runs at all TP to anchor β₅. Larger models only at TP configs that fit in GPU memory (H100 80GB assumed). Adjust per actual hardware.

---

## Runs Per Combo

| Profile | Sweep parameter | Values | `output_len` | What each step measures | Runs |
|---------|----------------|--------|:---:|---|:----:|
| W1: Prefill sweep | `question_len` | 128, 512, 1K, 2K, 4K, 8K | **1** | Pure prefill — `question_len` controls how many tokens the GPU prefills per step | 6 |
| W2: Decode context | `question_len` | 128, 512, 1K, 2K, 4K, 8K | **512** | Pure decode — `question_len` controls the starting KV cache context length that each decode step must read | 6 |
| W3: Batch scaling | `rate` | 1, 4, 8, 16, 32, 64, 128 | 256 | Mixed — fixed `question_len=256`; varies number of concurrent requests via rate | 7 |
| W4: Pf/dc ratio | profile | A, B, C | 4 / 256 / 1024 | Mixed — varies the prefill vs decode fraction per step | 3 |
| | | | | **Total per combo** | **22** |

W5 (TP scaling) is subsumed by running W3 at each TP config.

---

## Full Experiment List

**Total: 220 runs** (10 combos × 22 runs). Each run is 2 minutes → ~7.3 hours sequential.

### Llama-3.1-8B — TP=1 (22 runs)

| # | Profile | Sweep value |
|---|---------|------------|
| 1–6 | W1 | question_len ∈ {128, 512, 1024, 2048, 4096, 8192} |
| 7–12 | W2 | question_len ∈ {128, 512, 1024, 2048, 4096, 8192} |
| 13–19 | W3 | rate ∈ {1, 4, 8, 16, 32, 64, 128} |
| 20 | W4-A | rate=128, question_len=1024, output_len=4 |
| 21 | W4-B | rate=32, question_len=256, output_len=256 |
| 22 | W4-C | rate=32, question_len=64, output_len=1024 |

### Llama-3.1-8B — TP=2 (22 runs)

Same 22 runs as above.

### Llama-3.1-8B — TP=4 (22 runs)

Same 22 runs as above.

### Llama-3.1-70B — TP=2 (22 runs)

Same 22 runs as above.

### Llama-3.1-70B — TP=4 (22 runs)

Same 22 runs as above.

### Mixtral-8x22B — TP=4 (22 runs)

Same 22 runs as above.

### DeepSeek-V3 — TP=4 (22 runs)

Same 22 runs as above. W3 high-rate sweeps (rate=64, 128) may hit KV cache limits at TP=4 — reduce max rate if OOM.

---

## β Identifiability Checklist

| β | Required data | Covered by |
|---|--------------|------------|
| β₁ (prefill) | Varied prefill token count | W1 (6 lengths), W4-A vs W4-C |
| β₂ (decode) | Varied decode context | W2 (6 lengths), W4-C |
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
- For W2 at high `question_len` (8K), decode step times increase; B may briefly reach 2.
- Run Llama-8B first (cheapest) to validate the pipeline end-to-end before larger models.
