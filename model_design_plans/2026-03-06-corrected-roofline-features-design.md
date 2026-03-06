# Design: Corrected Roofline Feature Set

**Date:** 2026-03-06 (Rev 7: 2026-03-06 — formula design review fixes)
**Species:** Specification
**Extension Type:** Policy Template (new algorithm behind existing frozen LatencyModel interface)
**Builds on:** Issue #489 Rev 6 design doc (corrected roofline concept)
**Scope:** Feature vector design for step time prediction
**Validated against:** vLLM v0.8–v0.10 source code, DeepSeek-V2/V3 papers, DuetServe, Vidur, Pope et al.

---

## 1. Goal

Design physically-motivated basis functions for:

```
StepTime = Σ βᵢ · fᵢ(batch, architecture, hardware, parallelism)
```

Covering dense, MoE (routed + shared experts), MLA (compressed KV via absorbed attention), and hybrid (mixed dense/MoE layers). Parallelism: TP only (DP/EP deferred — see §13).

**Phased approach:** A minimal version (dense + standard MoE, 5 features — drop MLA-specific terms) would cover Llama and Mixtral families. MLA support adds ~40% formula complexity (absorbed attention, compressed KV, v_head_dim/qk_nope_head_dim distinction, 4 new config fields) but is needed for DeepSeek-V2/V3 — the fastest-growing model family in production. The full 7-feature design is specified here; implementation may phase MLA as a second PR if needed.

---

## 2. Formula

### 7 features, 7 β parameters (+ 3 α = 10 total)

```
StepTime = β₁ · max(F_pf_compute, F_pf_kv)     prefill phase bottleneck
         + β₂ · max(F_dc_compute, F_dc_kv)     decode phase bottleneck
         + β₃ · F_weight_static                  attention + dense/shared FFN weights
         + β₄ · F_weight_moe                     routed expert weights
         + β₅ · F_tp                             TP all-reduce (custom IPC kernel)
         + β₆ · batch_size                       CPU scheduling overhead (per-request)
         + β₇                                    constant overhead (intercept)
```

- Compute features = FLOPs / `TFlops_peak`. Bandwidth features = Bytes / `BW_peak`. Both yield µs (time at 100% hardware utilization). β > 1 corrects for real-world utilization (always slower than peak).
- β₁–β₅ learned from data (warm-start 1.0). β₆ µs/request. β₇ µs (intercept).
- **Alpha model unchanged:** The 3 α coefficients (queueing, preprocessing, output processing) are identical to existing blackbox/crossmodel backends. Alpha models pre-scheduling and post-processing delays that are independent of GPU step time — improved step time accuracy does not require re-fitting α.

**Sequential-phase assumption:** The formula adds prefill and decode phases. In vLLM v0.8+, a single step processes mixed batches where prefill and decode tokens are scheduled together into one `model_runner.execute_model()` call. The GPU may interleave prefill compute warps with decode memory-bound warps. The additive treatment is conservative (overestimates) — β₁, β₂ fit < 1.0 to compensate for any overlap. This is the standard approach in DuetServe, Vidur, and Pope et al.

**Why β₆ · batch_size:** vLLM's `prepare_model_input()` iterates every request to build block tables, slot mappings, and sampling parameters. This CPU work scales linearly with B and is architecture-independent (doesn't scale with hidden_dim or num_layers), making it identifiable in cross-model fitting. For single-model fits, β₆ is confounded with the decode terms — harmless, β₂ absorbs it.

---

## 3. Notation

### From config.json

| Symbol | Field | Meaning |
|--------|-------|---------|
| L | `num_hidden_layers` | Total decoder layers |
| d | `hidden_size` | Hidden dimension |
| H | `num_attention_heads` | Query heads |
| kv_heads | `num_key_value_heads` | KV heads (GQA) |
| d_h | d / H | Head dimension |
| d_kv | kv_heads · d_h | Full KV dimension |
| d_ff | `intermediate_size` | Dense/shared-expert FFN intermediate dim |
| d_ff_expert | `moe_intermediate_size` | **NEW.** Routed/shared expert FFN intermediate dim (may differ from d_ff; e.g., DeepSeek-V3: d_ff=18432, d_ff_expert=2048 — 9× smaller) |
| N | `num_local_experts` | Routed experts (0 = dense) |
| k | `num_experts_per_tok` | Active experts per token |
| n_shared | `n_shared_experts` | **NEW.** Shared experts (0 = none) |
| kv_lora_rank | `kv_lora_rank` | **NEW.** MLA compressed KV dim (0 = standard) |
| qk_rope_head_dim | `qk_rope_head_dim` | **NEW.** MLA decoupled RoPE key dim (0 = standard) |
| v_head_dim | `v_head_dim` | **NEW.** MLA per-head V output dim (0 = use qk_nope_head_dim). Architecturally independent from qk_nope_head_dim; equal in DeepSeek-V2/V3 (both 128) |
| first_k_dense | `first_k_dense_replace` | **NEW.** Initial layers with dense FFN (0 = all MoE) |

### Derived

| Symbol | Formula | Notes |
|--------|---------|-------|
| k_eff | max(1, k) | 1 for dense models |
| N_eff(B) | min(N, max(k, B·k)) | Batch-dependent loaded experts; 1 for dense |
| L_moe | L − first_k_dense (if N > 0, else 0) | MoE FFN layers |
| L_dense | L − L_moe | Dense FFN layers |
| d_kv_cache | standard: 2·kv_heads·d_h; MLA: kv_lora_rank + qk_rope_head_dim | Cached KV elements per token per layer (pre-TP) |
| d_kv_eff | standard: d_kv_cache / TP; MLA: d_kv_cache (not TP-sharded) | Per-rank cached KV elements |
| bpp | bytes_per_param | Weight precision (typically 2 for fp16) |
| bytes | bytes_per_elem | KV cache precision (typically 2 for fp16) |

**MLA KV cache:** Stores compressed latent `c_kv` (dim `kv_lora_rank`) plus decoupled RoPE key `k_R` (dim `qk_rope_head_dim`). Total = 576 elements for DeepSeek-V2/V3 (512 + 64), vs 32,768 for standard MHA — a ~57× compression. The RoPE key must be cached separately because positional encoding cannot be applied to the compressed representation.

### Parallelism

| Symbol | Meaning | vLLM relationship |
|--------|---------|-------------------|
| TP | Tensor parallelism | `--tensor-parallel-size`. Shards attention heads + FFN/expert weight columns |

All experts present on every GPU, weights TP-sharded. 2× AllReduce per layer (post-attention, post-FFN/MoE). No all-to-all. DP/EP support deferred (§12).

### Per-batch (from `[]*Request`)

- **Prefill requests**: ProgressIndex < len(InputTokens). Context S = ProgressIndex. New tokens T_new = NumNewTokens.
- **Decode requests**: ProgressIndex ≥ len(InputTokens). Context S = ProgressIndex. New tokens = 1.
- T_pf = total new prefill tokens. T_dc = total decode tokens. T_total = T_pf + T_dc. B = batch size.

---

## 4. Feature Definitions

### 4.1 F_pf_compute — Prefill GPU Compute Time

Time to execute all prefill FLOPs at peak throughput. Architecture-aware with different parallelism divisors per component.

```
F_pf_compute = [A_pf + D_pf + R_pf + S_pf] / (TFlops_peak · 1e12)   [µs at peak]
```

**A_pf — Attention FLOPs** (all L layers, divided by TP):

Per prefill request (context S, new tokens T_new):

*Standard MHA/GQA:*

| Component | FLOPs per layer |
|-----------|----------------|
| QKV projection | 2 · T_new · d · (d + 2·d_kv) / TP |
| Attention scores | 4 · (H/TP) · T_new · (S + T_new/2) · d_h |
| Output projection | 2 · T_new · d² / TP |

*MLA (absorbed attention):*

In MLA, W_UK and W_UV are algebraically absorbed into Q and O projections. vLLM computes Q normally, then applies per-head absorption — it does NOT fuse W_Q·W_UK into a single matrix (that would be H× more FLOPs). Each head attends to the full `kv_lora_rank`-dim compressed c_kv (not divided by H).

**Projection GEMMs (per layer):**

| Component | FLOPs per layer | Notes |
|-----------|----------------|-------|
| KV compression (W_DKV) | 2 · T_new · d · (kv_lora_rank + qk_rope_head_dim) | Fused c_kv + RoPE key; replicated (not TP-sharded) |
| Q projection (W_Q) | 2 · T_new · d² / TP | Approximation: actual is two-step q_a(d→q_lora_rank) + q_b(q_lora_rank→H·(noPE+RoPE)), ≈d² total |
| Q absorption (per-head W_UK^T) | 2 · T_new · H · qk_nope_head_dim · kv_lora_rank / TP | q_nope_h (qk_nope_head_dim) → q_compressed_h (kv_lora_rank). **Not** d/H — DeepSeek-V3: qk_nope_head_dim=128, d/H=56 |
| V absorption (per-head W_UV) | 2 · T_new · H · kv_lora_rank · v_head_dim / TP | c_kv (kv_lora_rank) → v_h (v_head_dim) |
| Output projection (W_O) | 2 · T_new · H · v_head_dim · d / TP | Non-square for MLA: (H·v_head_dim) → d. DeepSeek-V3: 16384→7168 |

**Attention score FLOPs (per layer):**

| Component | FLOPs per layer | Notes |
|-----------|----------------|-------|
| Attention scores (noPE) | 4 · (H/TP) · T_new · (S + T_new/2) · kv_lora_rank | Inner dim = full kv_lora_rank per head |
| Attention scores (RoPE) | 2 · (H/TP) · T_new · (S + T_new/2) · qk_rope_head_dim | QK^T only (RoPE doesn't participate in AV) |

MLA trades ~57× KV bandwidth reduction for ~4× more attention score FLOPs per head (kv_lora_rank=512 vs d_h=128). Net win for memory-bound decode. The absorbed formulation is what vLLM uses in production.

**Worked example — DeepSeek-V3, 1 prefill request, T_new=128, S=0, TP=1, single layer:**
- W_DKV: 2 · 128 · 7168 · 576 = 1.06G FLOPs
- W_Q: 2 · 128 · 7168² = 13.15G FLOPs (approx)
- Q absorption: 2 · 128 · 128 · 128 · 512 = 2.15G FLOPs
- V absorption: 2 · 128 · 128 · 512 · 128 = 2.15G FLOPs
- W_O: 2 · 128 · 128 · 128 · 7168 = 30.06G FLOPs
- Attention scores (noPE): 4 · 128 · 128 · 64 · 512 = 2.15G FLOPs
- Attention scores (RoPE): 2 · 128 · 128 · 64 · 64 = 0.13G FLOPs
- **Total per layer:** ~50.9G FLOPs (attention only)
- At 989 TFlops (H100): ~50.9/989 ≈ **51 µs/layer** at peak

**Note:** DeepSeek also compresses Q (via q_lora_rank). We approximate Q compression as standard Q projection — the two-step down+up has similar total FLOPs (q_lora_rank ≈ d/4). β₁/β₂ absorb any discrepancy.

```
A_pf = L · Σ_req [attention FLOPs per layer] / TP
```

**D_pf — Dense FFN FLOPs** (L_dense layers, divided by TP):

```
D_pf = L_dense · T_pf · 6 · d · d_ff / TP
```

Factor 6 = 3 weight matrices × 2 FLOPs/element, assuming SwiGLU/GeGLU (gate + up + down). All target models (Llama, Mixtral, DeepSeek, Qwen) use gated FFN. Non-gated models (4·d·d_ff) are rare; β₁/β₂ absorb the ~1.5× overestimate.

**R_pf — Routed expert FLOPs** (L_moe layers, all experts on every GPU, TP-sharded):

```
R_pf = L_moe · T_pf · k_eff · 6 · d · d_ff_expert / TP
```

**S_pf — Shared expert FLOPs** (L_moe layers, divided by TP):

Shared experts are always activated for every token. They are TP-sharded (confirmed by NVIDIA B200 analysis), same architecture as routed experts (using d_ff_expert, not d_ff).

```
S_pf = L_moe · T_pf · n_shared · 6 · d · d_ff_expert / TP
```

For pure dense models: R_pf = 0, S_pf = 0, L_moe = 0.

### 4.2 F_dc_compute — Decode GPU Compute Time

Same structure as F_pf_compute but for decode tokens:

```
F_dc_compute = [A_dc + D_dc + R_dc + S_dc] / (TFlops_peak · 1e12)   [µs at peak]
```

Per decode request, T_new = 1 and context S = ProgressIndex. Attention scores scale with full context length (the dominant cost for decode, especially for long sequences).

### 4.3 F_pf_kv — Prefill KV Cache Bandwidth

KV reads (existing context for chunked prefill) + KV writes (new tokens). MoE does NOT affect this — KV is purely attention.

```
Bytes_pf_kv = L · d_kv_eff · (Σ_pf_req S_i + T_pf) · bytes

F_pf_kv = Bytes_pf_kv / BW_peak   [µs at peak]
```

- Σ S_i: total cached context tokens across prefill requests (S_i = ProgressIndex for request i). During chunked prefill, each new token attends to all previously cached KV — this read dominates for long-context requests where S >> T_new.
- T_pf: total new prefill tokens written to cache.
- Standard attention: d_kv_eff = 2 · (kv_heads / TP) · d_h
- MLA: d_kv_eff = kv_lora_rank + qk_rope_head_dim (NOT TP-sharded — compressed rep is small enough to replicate). For DeepSeek-V3: 512 + 64 = 576 elements.

### 4.4 F_dc_kv — Decode KV Cache Bandwidth

KV reads (full context per decode request) + KV writes (new tokens):

```
Bytes_dc_kv = L · d_kv_eff · (Σ_decode_req S_i + T_dc) · bytes

F_dc_kv = Bytes_dc_kv / BW_peak   [µs at peak]
```

This is typically the dominant memory cost — every decode token reads the entire KV cache for its context. MLA dramatically reduces this (~57× fewer cached elements per token).

### 4.5 F_weight_static — Static Weight Loading

Weights loaded every step regardless of batch routing: attention + dense FFN + shared experts.

```
W_attn  = L · [attn weight bytes] / TP
W_dense = L_dense · 3 · d · d_ff · bpp / TP
W_shared = L_moe · n_shared · 3 · d · d_ff_expert · bpp / TP

F_weight_static = (W_attn + W_dense + W_shared) / BW_peak   [µs at peak]
```

**Attention weight bytes per layer:**

*Standard MHA/GQA:*
```
W_attn_layer = d · (d + 2·d_kv + d) · bpp = d · (2d + 2·d_kv) · bpp
```

*MLA (original weights, absorbed at runtime):*
vLLM stores the **original** weight matrices (W_DKV, W_Q, W_UK, W_UV, W_O separately). Absorption is a runtime computation reordering: instead of materializing full-dimensional K/V, vLLM computes `(Q · W_UK) · c_kv^T` and `attn · c_kv · W_UV`. Weight matrices loaded per layer:
```
W_attn_layer_mla = [
    d · (kv_lora_rank + qk_rope_head_dim)         # W_DKV (fused c_kv + RoPE key), replicated
  + d · d / TP                                    # Q projection (W_Q, TP-sharded). Approx: q_a+q_b ≈ d²
  + (H/TP) · qk_nope_head_dim · kv_lora_rank     # Q absorption (per-head W_UK^T). Uses qk_nope_head_dim, NOT d/H
  + (H/TP) · kv_lora_rank · v_head_dim            # V absorption (per-head W_UV)
  + H · v_head_dim · d / TP                       # Output projection (W_O, TP-sharded). Non-square: (H·v_head_dim)→d
] · bpp
```
For DeepSeek-V3 (d=7168, H=128, qk_nope_head_dim=128, v_head_dim=128, kv_lora_rank=512, TP=1): W_DKV=4.13M + W_Q≈51.4M + W_UK=8.39M + W_UV=8.39M + W_O=117.4M ≈ 190M params/layer.
(β₃ absorbs discrepancies from the q_a+q_b ≈ d² approximation.)

### 4.6 F_weight_moe — Routed Expert Weight Loading

Batch-dependent: how many of the N experts get their weights loaded from HBM.

```
N_eff = min(N, max(k, B · k))

W_moe = L_moe · N_eff · 3 · d · d_ff_expert · bpp / TP

F_weight_moe = W_moe / BW_peak   [µs at peak]
```

All N experts on every GPU, each TP-sharded. N_eff experts actually accessed per step (batch-dependent). For dense models: F_weight_moe = 0.

**N_eff limitations:** The linear formula `min(N, max(k, B·k))` assumes uniform expert activation. Two real-world effects cause overestimation:

1. **Group routing:** DeepSeek-V2/V3 uses hierarchical routing (`n_group=8, topk_group=4`) which limits co-activation patterns.
2. **Expert popularity skew:** MoE models exhibit Zipf-like expert popularity in production (documented in Switch Transformer and GShard). With B=32 and k=6, far fewer than min(N, 192) unique experts may be accessed.

β₄ absorbs both effects (fitting < 1.0). **Fallback:** If β₄ < 0.5 consistently for MoE models, replace with a group-aware formula: `N_eff = min(N, n_groups · min(topk_group, ceil(B·k/n_groups)))` or an empirical lookup from profiled expert activation counts.

### 4.7 F_tp — TP Communication

Two all-reduce operations per layer (post-attention, post-FFN/MoE). For MoE layers with shared experts, vLLM fuses shared+routed outputs into one AllReduce — still 2 per layer, not 3.

```
TP = 1:   F_tp = 0
TP ≥ 2:   F_tp = L · 2 · T_total · d · bytes / BW_peak   [µs at peak]
```

Normalized by HBM bandwidth (not NVLink). β₅ absorbs the NVLink/HBM ratio (~3-4×), launch latency, and algorithm selection (one-shot vs two-shot, NCCL fallback).

### 4.8 Overhead Terms (learned, no physics prior)

```
β₆ · batch_size + β₇   [µs]
```

- **β₆ · batch_size:** CPU scheduling overhead that scales per-request. In vLLM, `prepare_model_input()` builds per-request block tables, slot mappings, and sampling parameters — O(B) CPU work before each GPU step.
- **β₇ (intercept):** Fixed per-step cost — kernel launch overhead, CUDA graph replay overhead, driver dispatch. Subsumes the old `L · perLayerOverhead` term (per-layer overhead is already captured by β₁–β₅ which all scale with L).

No analytical model — learned directly from vLLM training data. Addresses #482 (zero inter-step overhead in existing backends).

---

## 5. Architecture Impact Matrix

| Feature | Dense (Llama) | MoE (Mixtral) | MoE+shared+MLA (DeepSeek-V3) |
|---------|---------------|---------------|-------------------------------|
| F_pf_compute | QKV+attn+O + dense_ffn | same + k·routed_ffn | Q+absorption+scores(noPE+RoPE)+O + k·routed + shared |
| F_pf_kv | 2·kv_heads·d_h / TP | same | kv_lora_rank + qk_rope_head_dim (replicated) |
| F_dc_kv | 2·kv_heads·d_h · Σ S / TP | same | (kv_lora_rank + qk_rope_head_dim) · Σ S |
| F_weight_static | attn + ffn | attn | W_DKV + W_Q + W_UK + W_UV + W_O + shared |
| F_weight_moe | 0 | N_eff · expert_weights / TP | N_eff · expert_weights / TP |
| F_tp | 2L allreduce | 2L allreduce | 2L allreduce |

---

## 6. New Config Fields Required

### ModelConfig additions (from config.json)

| Field | Type | Default | Source |
|-------|------|---------|--------|
| `MoEIntermediateDim` | int | 0 | `moe_intermediate_size` in config.json (0 = use `intermediate_size`). d_ff_expert in formulas |
| `NumSharedExperts` | int | 0 | `n_shared_experts` in config.json |
| `KVLoraRank` | int | 0 | `kv_lora_rank` in config.json (0 = standard attention) |
| `QKNopeHeadDim` | int | 0 | `qk_nope_head_dim` in config.json (0 = use d/H). Used in MLA Q absorption FLOPs |
| `VHeadDim` | int | 0 | `v_head_dim` in config.json (0 = use qk_nope_head_dim). Used in MLA V absorption + W_O |
| `QKRopeHeadDim` | int | 0 | `qk_rope_head_dim` in config.json (0 = standard) |
| `FirstKDenseReplace` | int | 0 | `first_k_dense_replace` in config.json (0 = all MoE if experts > 0) |

No new HardwareCalib fields required. Step time features use only `TFlopsPeak` and `BwPeakTBs`; KV capacity auto-calculation uses `MemoryGiB`. All three are datasheet values — no empirical calibration needed for unseen hardware. MFU and BW_eff are absorbed by β coefficients.

**KV capacity impact:** For MLA models, per-token KV cache size is `(kv_lora_rank + qk_rope_head_dim) · bytes` per layer, replacing the standard `2 · kv_heads · d_h · bytes`. The KV capacity auto-calculation must use `d_kv_cache` (as defined in §3) when `KVLoraRank > 0`. Without this update, MLA models (DeepSeek-V2/V3) would massively over-allocate KV blocks (~57× too many).

---

## 7. Degenerate Cases

| Condition | Effect |
|-----------|--------|
| Pure dense, TP=1 | Only F_pf_compute, F_pf_kv, F_dc_compute, F_dc_kv, F_weight_static, β₆·B, β₇ nonzero |
| MoE, TP>1 | All experts on every GPU (TP-sharded). 2L AllReduce |
| MLA, TP=1 | F_pf_kv and F_dc_kv use compressed dim. Absorbed attention FLOPs |
| MLA, TP>1 | KV cache replicated on all TP ranks (MLA can't shard KV by head) |
| Empty batch | StepTime = β₇ (floored to 0) |
| All β = 1.0, β₆=β₇=0 | Pure analytical roofline (no corrections, no overhead) |

---

## 8. Comparison with Rev 6

| Aspect | Rev 6 | This design |
|--------|-------|-------------|
| Parameters | 7β + 3α = 10 | 7β + 3α = 10 |
| MoE FLOPs | k_eff on combined FFN | k_eff on routed + separate shared + dense layer split |
| MoE weights | Single T_weight with N_eff | Split: F_weight_static + F_weight_moe |
| MLA | Not modeled | Absorbed attention FLOPs + compressed KV (kv_lora_rank + qk_rope_head_dim) |
| Hybrid layers | Not modeled | L_moe / L_dense via first_k_dense_replace |
| TP communication | Piecewise (ring) | Custom IPC all-reduce (one-shot/two-shot by msg size) |
| Shared experts | Not modeled | Explicit FLOPs + weights, TP-sharded |

---

## 9. Fitting Notes

- Training data must include TP diversity (TP=1,2,4) for F_tp identifiability.
- Must include MoE models for F_weight_moe identifiability (vs F_weight_static).
- MLA models (DeepSeek) exercise the compressed KV features and the qk_nope_head_dim-based absorption formulas.
- Sequential anchoring (Rev 6 Section 5.2): fit β₆, β₇ first on single-request traces, then remaining params jointly.
- Falsification: if any β > 5.0, the corresponding analytical feature is a poor prior.
- If β₄ < 0.5 for MoE models, investigate group-aware N_eff formula.

**Formula structure choice:** The `β · max(F_compute, F_kv)` decomposition uses a single β per phase rather than separate βs for compute and bandwidth (e.g., `β₁ₐ·F_pf_compute + β₁ᵦ·F_pf_kv`). The single-β form has fewer parameters (7 vs 11) and better physical motivation — the `max()` selects the bottleneck, and β corrects for utilization. The separate-β form would allow the model to weight compute and bandwidth independently but at the cost of overfitting risk and loss of the roofline prior. If residual analysis shows systematic phase-dependent errors, splitting β₁ and β₂ into per-resource βs is the natural next step.

**Collinearity notes:**
- **F_weight_static vs β₇ (intercept):** In single-model fits, F_weight_static is constant (same architecture every step) → perfectly collinear with the intercept. Only identifiable in cross-model fitting where architecture varies across training samples. For single-model fits, β₃ and β₇ are jointly absorbed — this is acceptable since total overhead is still predicted correctly.
- **F_dc_compute vs F_dc_kv:** For memory-bound decode (most configurations), these are near-collinear (both scale linearly with Σ S_i). The `max()` operation breaks symmetry only at the compute-bound crossover. If β₂ confidence intervals are wide, consider constraining β₂ ∈ [0.5, 3.0] or fitting with ridge regularization.
- **β₆ (batch_size) vs decode terms:** In single-model fits, batch_size and decode token count are correlated (more requests = more decode tokens). Identifiable in cross-model fitting or with varied batch compositions (different prefill/decode ratios).

### Cross-hardware transfer

Since features use peak specs only, each β absorbs a hardware utilization factor:

| β | Absorbs | Typical range |
|---|---------|---------------|
| β₁ | 1/MFU_prefill | 1.4–2.0 |
| β₂–β₄ | 1/BW_eff | 1.2–1.6 |
| β₅ | BW_peak/BW_nvlink | 3–5 |

**Cross-model** (primary use case): β trained on models A, B predicts model C on the same hardware. Works directly — β captures hardware+software constants, physics features capture architecture differences.

**Cross-hardware**: β from GPU X does not directly transfer to GPU Y (utilization differs). Three modes:

1. **Zero-shot** (no target data): Apply source β with target datasheet specs. Expect ~1.5× error from utilization mismatch. Useful for rough capacity planning.
2. **Few-shot calibration** (recommended): Run 1 model (e.g., Llama-8B) at ~10 batch sizes on target hardware. Estimate 2 scalars — compute efficiency `η_c` and memory efficiency `η_m` — by least-squares on the calibration data. Scale: `β₁' = β₁ · (η_c_source / η_c_target)`, similarly for memory-bound β. All model-specific knowledge transfers; only 2 hardware-specific scalars re-estimated.
3. **Full re-fit**: Re-fit all 7 β on target hardware. Best accuracy, requires full benchmark sweep.

**Default behavior:** Cross-model (same hardware) is the primary use case — β coefficients ship in `defaults.yaml` per GPU family. Cross-hardware transfer defaults to zero-shot (apply source β with target datasheet specs). Few-shot calibration scalars (η_c, η_m) would be stored in `defaults.yaml` per GPU pair or passed via CLI flags — deferred to implementation.

---

## 10. Validation Against Published Literature

| Design choice | Published support | Reference |
|---------------|-------------------|-----------|
| Per-phase max(compute, bandwidth) | Standard roofline | Pope et al. 2022, DuetServe, JAX Scaling Book |
| Prefill/decode separation | Universal | Splitwise, DuetServe, Vidur, Wang et al. |
| MLA absorbed attention | vLLM production, ~0.56× HBM | DeepSeek-V2/V3 papers, vLLM v1 MLA backends |
| Scheduling as learned term | No analytical model exists | OOCO adds static overhead; BLIS Rev 6 |
| Separate MoE weight loading | Frontier uses separate ML predictor | AMD: "GPUs spend most time loading expert weights" |

### Known simplifications vs. DuetServe

DuetServe (the most precise published roofline) uses per-request attention summation: `t_attn = Σ_req max(F_req/Π, B_req/ℬ)`. Our design aggregates attention across all requests per phase. This is less accurate when one request has much longer context than others. Adopting per-request summation would improve accuracy but requires iterating requests inside the feature computation (higher implementation complexity, no additional β needed).

---

## 11. Known Simplifications

Acknowledged design choices where the model departs from physical reality. Each is documented with its expected impact and the compensation mechanism (which β absorbs the error).

| # | Simplification | Impact | Absorber | When to revisit |
|---|---------------|--------|----------|-----------------|
| S1 | **Weight loading additive, not inside max()** | Overpredicts when weights and KV access share HBM bus bandwidth. ~10-30% overestimate for weight-heavy steps. **Alternative considered:** `max(F_compute, F_kv + F_weight)` per phase — merges weight and KV bandwidth into a single bottleneck. Rejected because it couples weight and KV features, making β₃/β₄ unidentifiable in cross-model fitting. The additive form lets β₃ < 1.0 compensate naturally. **Fallback:** If β₃ < 0.3, switch to merged-bandwidth formula | β₃, β₄ fit < 1.0 | If fitted β₃ or β₄ consistently < 0.3 |
| S2 | **MFU/BW_eff absorbed into β, not explicit** | Features use peak specs; β₁ absorbs compute MFU (~1/0.65≈1.5), β₂–β₄ absorb bandwidth efficiency (~1/0.72≈1.4). max() bottleneck selection uses peak values — robust for strongly compute-bound prefill and memory-bound decode, less accurate near crossover. **Crossover cases:** (a) very long context prefill (128K+) where KV write bandwidth dominates, (b) small prefill chunks in chunked-prefill mode where existing KV reads dominate, (c) small models (1-3B) on H100 where even prefill can be memory-bound. Near crossover, the max() may select the wrong branch, and β correction becomes batch-dependent | β₁–β₅ | If β varies > 5× across GPUs (would indicate MFU ratio is not stable) |
| S3 | **N_eff linear approximation** ignores group routing | Overestimates loaded experts for DeepSeek-V3's hierarchical routing (8 groups, top-4 groups). Real N_eff may be 40-60% of linear estimate at moderate B | β₄ absorbs; already noted in §4.6 | If β₄ < 0.5 consistently for DeepSeek models |
| S4 | **Weight loading as separate sequential phase** | In reality, weight loading is pipelined — GPU prefetches layer N+1 weights while computing layer N. The additive β₃·F_weight term treats weight loading as a separate phase after compute, overestimating when compute is the bottleneck. Note: F_weight_static appears once in the formula (not per-phase), so there is no double-counting — just sequential vs overlapped timing | β₃ fits < 1.0 (absorbs overlap) | If mixed-batch error systematically > pure-phase error |
| S5 | **No paged attention scatter pattern** | Paged KV cache has non-contiguous memory access; actual bandwidth < peak | β₂–β₄ absorb | If decode-heavy workloads consistently under-predict |
| S6 | **CUDA graph overhead in intercept** | CUDA graph replay cost varies by graph size (more ops = more replay time). Lumped into β₇ | β₇ | If β₇ varies > 5× across models (graph size correlates with model size) |
| S7 | **Batch-aggregated attention** (not per-request max) | Less accurate when one request has 100K context alongside 128-token requests. **Alternative considered:** per-request `max(F_compute, F_bandwidth)` then sum (DuetServe approach). More accurate for heterogeneous batches but requires iterating requests inside feature computation — O(B) per feature instead of O(1) from batch aggregates. The error from batch aggregation is nonlinear (`max(Σa, Σb) ≤ Σmax(a,b)`), meaning a linear β cannot fully correct it for extreme variance. However, typical vLLM continuous-batching produces moderate context variance; extreme 128-vs-100K batches are rare. **Fallback:** If high-variance workloads consistently mis-predict, implement per-request feature summation (no additional β needed, only code change) | β₁, β₂ absorb average error | If high-variance context length workloads consistently mis-predict |
| S8 | **Q projection approximated as d²** for MLA | Actual is two-step: d→q_lora_rank→H·(noPE+RoPE). For DeepSeek-V3: 48.8M vs 51.4M (5% error) | β₁ absorbs | Never — error is < 5% |

---

## 12. Training Data

### Infrastructure

- **Data collection**: [tektonc-data-collection](https://github.com/inference-sim/tektonc-data-collection) — Tekton pipelines on K8s, results to S3
- **Workload generation**: [inference-perf](https://github.com/inference-sim/tektonc-data-collection) — controlled batch composition sweeps
- **Instrumented vLLM**: [inference-sim/vllm](https://github.com/inference-sim/vllm) — fork with step-level and journey-level OpenTelemetry tracing

### Per-step data (for β fitting)

From `step.BATCH_SUMMARY` events emitted by instrumented vLLM:

| Field | Maps to |
|-------|---------|
| `step_duration_us` | **Target variable** (StepTime) |
| `num_prefill_requests`, `prefill_tokens` | T_pf, prefill batch composition |
| `num_decode_requests`, `decode_tokens` | T_dc, decode batch composition |
| `total_tokens_scheduled` | T_total (= T_pf + T_dc) |
| `kv_cache_usage` | Validation (memory pressure) |

With `REQUEST_SNAPSHOT` enabled: per-request `prompt_tokens`, `computed_tokens` → context lengths S_i needed for attention score FLOPs and KV bandwidth features.

### Per-request data (for α fitting)

From journey tracing events:

| Event pair | Maps to |
|------------|---------|
| ARRIVED → SCHEDULED | α queueing time |
| SCHEDULED → FIRST_TOKEN | α preprocessing |
| FIRST_TOKEN → FINISHED | α output processing |

### Model coverage required

| Architecture class | Example models | Exercises features |
|-------------------|----------------|-------------------|
| Dense | Llama-3.1-8B/70B | F_pf/dc_compute, F_weight_static, F_tp |
| MoE | Mixtral-8x7B/22B | F_weight_moe, N_eff, d_ff_expert |
| MoE+shared+MLA | DeepSeek-V2/V3 | Absorbed attention, compressed KV, shared experts, hybrid layers |

Each model at TP=1,2,4 (minimum).

### Workload design for β identifiability

Each β requires specific workload variation to be learnable. Without it, β becomes confounded with another coefficient.

| β | What it captures | What workload variation separates it | Confounded with (if missing) |
|---|-----------------|--------------------------------------|------------------------------|
| β₁ | Prefill phase | Vary prefill token count (T_pf) while holding decode fixed. Pure-prefill batches + mixed batches. Vary input length (128→8K) | β₂ if all batches are mixed with fixed ratio |
| β₂ | Decode phase | Vary decode context length (S_i) while holding prefill fixed. Pure-decode batches. Vary context (128→8K+) | β₁ if all batches are mixed with fixed ratio |
| β₃ | Static weights | **Cross-model only.** F_weight_static is constant within a single model → identical to intercept. Need ≥3 models with different weight sizes (e.g., 8B, 22B, 70B) | β₇ (intercept) in single-model fits |
| β₄ | MoE weights | Need MoE models. Vary batch size (B=1→128) so N_eff changes (at B=1, N_eff=k; at B=64, N_eff→N). Dense models contribute nothing | β₃ if only dense models used |
| β₅ | TP comms | Need TP>1 configs. Vary T_total (different batch sizes) at each TP to see msg size vary. TP=1 contributes nothing (F_tp=0) | Not identifiable from TP=1 data alone |
| β₆ | CPU overhead | Vary batch size B independently of token counts. Key: prefill-heavy batches where B is small but T_pf is large (long prompts), vs decode-heavy batches where B is large but T_dc ≈ B | β₂ if B always scales with decode tokens |
| β₇ | Intercept | Small batches (B=1,2) anchor the constant. Also anchored by cross-model data where other features vary but overhead is stable | β₃ in single-model fits |

**Recommended workload sweep per model/TP config:**

1. **Pure prefill**: B=1, input lengths {128, 512, 1K, 2K, 4K, 8K} — anchors β₁, β₇
2. **Pure decode**: B=1, short prompt + long generation at varied context — anchors β₂, β₇
3. **Decode batch scaling**: fixed context, B ∈ {1, 4, 8, 16, 32, 64, 128} — separates β₂ from β₆
4. **Mixed batches**: fixed B, vary prefill:decode ratio {100:0, 75:25, 50:50, 25:75, 0:100} — separates β₁ from β₂
5. **MoE batch scaling** (MoE models only): B ∈ {1, 2, 4, 8, 16, 32, 64, 128} — N_eff variation identifies β₄

### Minimum dataset

- **Cross-model fitting** (primary): ≥3 model families × ≥3 TP configs × ~50 batch compositions ≈ 450+ step measurements
- **Per-model fitting** (validation): ~50 batch compositions per model/TP combo
- **Few-shot hardware calibration** (§9): 1 model × ~10 batch sizes on target GPU

Detailed workload profiles with inference-perf specs: see [`2026-03-06-corrected-roofline-training-workloads.md`](2026-03-06-corrected-roofline-training-workloads.md).

---

## 13. Future Work: DP/EP

When `--data-parallel-size > 1`, vLLM activates EP=TP×DP with sparse all-to-all dispatch/combine for routed experts. Modeling this requires two additional features (F_ep for all-to-all communication, F_dp for attention↔MoE redistribution), inter-node bandwidth parameters, and training data from DP+EP configurations. Deferred until sufficient DP+EP measurement data is available.
