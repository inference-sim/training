"""Analytical basis functions for the crossmodel latency model.

Computes roofline-derived basis function values (in microseconds) for each
scheduler step.  These are the features X in the regression step_time ≈ β·X,
where β₁–β₇ are the learned coefficients.

Formula reference: inference-sim/inference-sim#489 (issuecomment-4013680061)

    StepTime = β₁·max(T_pf_compute, T_pf_kv)   [prefill bottleneck]
             + β₂·max(T_dc_compute, T_dc_kv)   [decode bottleneck]
             + β₃·T_weight                      [weight loading]
             + β₄·T_tp                          [TP communication]
             + β₅·L                             [per-layer overhead]
             + β₆·batch_size + β₇              [scheduling overhead]

Extensibility: each basis function is a standalone pure function.  To add a
new basis function: (1) write a new function, (2) add a field to
StepBasisValues, (3) add one line to compute_step_basis().  No other files
change.

Public API
----------
compute_step_basis(step, arch, hw, tp) -> StepBasisValues
    Compute all basis functions for one step.

compute_experiment_basis(reconstruction, arch, hw, tp) -> list[StepBasisValues]
    Compute basis for every step in an experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from reconstruct_steps import ExperimentReconstruction, ReconstructedStep

# All byte counts use FP16 (2 bytes per element).
_BYTES_PER_ELEMENT = 2


# =============================================================================
# Data types (frozen, with invariants)
# =============================================================================

@dataclass(frozen=True)
class ModelArch:
    """Model architecture parameters derived from config.json.

    Invariants:
        d_h == d // H
        d_kv == kv_heads * d_h
        k_eff == max(1, k)
        is_moe == (N > 0)
    """
    L: int          # num_hidden_layers
    d: int          # hidden_size
    H: int          # num_attention_heads
    kv_heads: int   # num_key_value_heads
    d_h: int        # head dimension = d // H
    d_kv: int       # KV dimension = kv_heads * d_h
    d_ff: int       # intermediate_size (per expert for MoE)
    N: int          # num_local_experts (0 for dense)
    k: int          # num_experts_per_tok (0 for dense)
    k_eff: int      # effective expert multiplier = max(1, k)
    is_moe: bool    # True if N > 0


@dataclass(frozen=True)
class HardwareSpec:
    """GPU hardware specification derived from vendor datasheet.

    Invariants:
        flops_peak > 0, bw_hbm > 0, bw_nvlink > 0
    """
    flops_peak: float   # FP16 tensor core peak (TFLOPS)
    bw_hbm: float       # HBM bandwidth (TB/s)
    bw_nvlink: float    # NVLink bandwidth (GB/s)

    def __post_init__(self):
        for name in ("flops_peak", "bw_hbm", "bw_nvlink"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(f"HardwareSpec.{name} must be positive, got {val!r}")


@dataclass(frozen=True)
class StepBasisValues:
    """Analytical basis function values for one scheduler step, in microseconds.

    These are the feature vector X for the regression step_time ≈ β·X.

    Invariants:
        All time values >= 0.
        t_pf_compute == t_pf_kv == 0 when step has no prefill requests.
        t_dc_compute == t_dc_kv == 0 when step has no decode requests.
        t_tp == 0 when tp == 1.
        num_layers > 0.
    """
    step_id: int
    t_pf_compute: float     # prefill compute bottleneck (µs)
    t_pf_kv: float          # prefill KV bandwidth bottleneck (µs)
    t_dc_compute: float     # decode compute bottleneck (µs)
    t_dc_kv: float          # decode KV bandwidth bottleneck (µs)
    t_weight: float         # weight loading time (µs)
    t_tp: float             # TP communication time (µs)
    num_layers: int         # L (per-layer overhead feature)
    batch_size: int         # scheduling overhead feature


# =============================================================================
# Loading helpers
# =============================================================================

def load_model_arch(config_json_path: str | Path) -> ModelArch:
    """Load model architecture from a HuggingFace config.json.

    Requires: config_json_path points to a valid HF config.json.
    Guarantees: all derived values (d_h, d_kv, k_eff, is_moe) are computed.
    """
    with open(config_json_path) as f:
        cfg = json.load(f)

    H = cfg["num_attention_heads"]
    d = cfg["hidden_size"]
    kv_heads = cfg.get("num_key_value_heads", H)  # defaults to H for MHA
    N = cfg.get("num_local_experts", 0)
    k = cfg.get("num_experts_per_tok", 0)

    return ModelArch(
        L=cfg["num_hidden_layers"],
        d=d,
        H=H,
        kv_heads=kv_heads,
        d_h=d // H,
        d_kv=kv_heads * (d // H),
        d_ff=cfg["intermediate_size"],
        N=N,
        k=k,
        k_eff=max(1, k),
        is_moe=N > 0,
    )


def load_hardware_spec(datasheet_path: str | Path) -> HardwareSpec:
    """Load GPU hardware spec from a vendor datasheet JSON.

    Requires: datasheet_path points to a valid datasheet JSON with
    compute.fp16_tensor_core, memory.bandwidth_tb_s, interconnect.nvlink_bandwidth_gb_s.

    Guarantees: flops_peak is fp16_tensor_core / 2 (dense FP16, not sparse).
    """
    with open(datasheet_path) as f:
        ds = json.load(f)

    return HardwareSpec(
        flops_peak=ds["compute"]["fp16_tensor_core"] / 2,
        bw_hbm=ds["memory"]["bandwidth_tb_s"],
        bw_nvlink=ds["interconnect"]["nvlink_bandwidth_gb_s"],
    )


# =============================================================================
# Individual basis functions (pure, stateless)
#
# Each function:
#   Requires: step is a valid ReconstructedStep, arch/hw are loaded, tp >= 1.
#   Guarantees: returns a non-negative float in microseconds.
# =============================================================================

def t_pf_compute(
    step: ReconstructedStep, arch: ModelArch, hw: HardwareSpec, tp: int,
) -> float:
    """Prefill compute time (µs).

    Formula:
        FLOPs_proj = L * 2 * T_pf * d * (2*d + 2*d_kv) / TP
        FLOPs_attn = L * Σᵢ 4 * (H/TP) * T_pf_i * (S_pf_i + T_pf_i/2) * d_h
        FLOPs_ffn  = L * T_pf * k_eff * 6 * d * d_ff / TP
        result     = (FLOPs_proj + FLOPs_attn + FLOPs_ffn) / (FLOPS_peak * 1e6)
    """
    if not step.prefill_reqs:
        return 0.0

    L = arch.L
    T_pf = step.total_prefill_tokens

    # QKV + output projection
    flops_proj = L * 2 * T_pf * arch.d * (2 * arch.d + 2 * arch.d_kv) / tp

    # Self-attention scores (per request, accounts for causal masking)
    H_per_gpu = arch.H // tp
    flops_attn = 0.0
    for entry in step.prefill_reqs:
        t_i = entry.tokens_this_step
        s_i = entry.prompt_tokens
        flops_attn += 4 * H_per_gpu * t_i * (s_i + t_i / 2) * arch.d_h

    flops_attn *= L

    # FFN (gate + up + down projections)
    flops_ffn = L * T_pf * arch.k_eff * 6 * arch.d * arch.d_ff / tp

    return (flops_proj + flops_attn + flops_ffn) / (hw.flops_peak * 1e6)


def t_pf_kv(
    step: ReconstructedStep, arch: ModelArch, hw: HardwareSpec, tp: int,
) -> float:
    """Prefill KV cache write bandwidth time (µs).

    Formula:
        Bytes = L * 2 * (kv_heads/TP) * d_h * T_pf * 2
        result = Bytes / (BW_hbm * 1e6)
    """
    if not step.prefill_reqs:
        return 0.0

    kv_heads_per_gpu = arch.kv_heads // tp
    bytes_kv = arch.L * 2 * kv_heads_per_gpu * arch.d_h * step.total_prefill_tokens * _BYTES_PER_ELEMENT
    return bytes_kv / (hw.bw_hbm * 1e6)


def t_dc_compute(
    step: ReconstructedStep, arch: ModelArch, hw: HardwareSpec, tp: int,
) -> float:
    """Decode compute time (µs).

    Formula:
        FLOPs_proj = L * 2 * T_dc * d * (2*d + 2*d_kv) / TP
        FLOPs_attn = L * Σⱼ 4 * (H/TP) * 1 * S_dc_j * d_h
        FLOPs_ffn  = L * T_dc * k_eff * 6 * d * d_ff / TP
        result     = (FLOPs_proj + FLOPs_attn + FLOPs_ffn) / (FLOPS_peak * 1e6)
    """
    if not step.decode_reqs:
        return 0.0

    L = arch.L
    T_dc = step.total_decode_tokens

    flops_proj = L * 2 * T_dc * arch.d * (2 * arch.d + 2 * arch.d_kv) / tp

    H_per_gpu = arch.H // tp
    sum_ctx = sum(entry.context_length for entry in step.decode_reqs)
    flops_attn = L * 4 * H_per_gpu * sum_ctx * arch.d_h

    flops_ffn = L * T_dc * arch.k_eff * 6 * arch.d * arch.d_ff / tp

    return (flops_proj + flops_attn + flops_ffn) / (hw.flops_peak * 1e6)


def t_dc_kv(
    step: ReconstructedStep, arch: ModelArch, hw: HardwareSpec, tp: int,
) -> float:
    """Decode KV cache read+write bandwidth time (µs).

    Formula:
        Bytes = L * 2 * (kv_heads/TP) * d_h * 2 * (Σⱼ S_dc_j + T_dc)
        result = Bytes / (BW_hbm * 1e6)
    """
    if not step.decode_reqs:
        return 0.0

    kv_heads_per_gpu = arch.kv_heads // tp
    sum_ctx = sum(entry.context_length for entry in step.decode_reqs)
    T_dc = step.total_decode_tokens
    bytes_kv = arch.L * 2 * kv_heads_per_gpu * arch.d_h * _BYTES_PER_ELEMENT * (sum_ctx + T_dc)
    return bytes_kv / (hw.bw_hbm * 1e6)


def t_weight(
    step: ReconstructedStep, arch: ModelArch, hw: HardwareSpec, tp: int,
) -> float:
    """Weight loading time (µs).

    Formula:
        N_eff = 1 for dense; min(N, max(k, B*k)) for MoE
        Bytes_attn = L * d * (2*d + 2*d_kv) * 2 / TP
        Bytes_ffn  = L * N_eff * 3 * d * d_ff * 2 / TP
        result     = (Bytes_attn + Bytes_ffn) / (BW_hbm * 1e6)

    Note: weight loading is constant per model for dense models. For MoE,
    it depends on the number of activated experts which scales with batch size.
    """
    if arch.is_moe:
        B = step.total_prefill_tokens + step.total_decode_tokens
        n_eff = min(arch.N, max(arch.k, B * arch.k))
    else:
        n_eff = 1

    bytes_attn = arch.L * arch.d * (2 * arch.d + 2 * arch.d_kv) * _BYTES_PER_ELEMENT / tp
    bytes_ffn = arch.L * n_eff * 3 * arch.d * arch.d_ff * _BYTES_PER_ELEMENT / tp
    return (bytes_attn + bytes_ffn) / (hw.bw_hbm * 1e6)


def t_tp(
    step: ReconstructedStep, arch: ModelArch, hw: HardwareSpec, tp: int,
) -> float:
    """Tensor-parallel communication bandwidth time (µs).

    NCCL launch latency (per-layer constant) is absorbed into β₅·L.
    This captures only the bandwidth-dependent component.

    Formula:
        msg_bytes = T * d * 2   [T = T_pf + T_dc]
        TP == 1: 0
        TP == 2: L * 2 * msg_bytes / (BW_nvlink * 1e3)      [point-to-point]
        TP >= 4: L * 2 * 2 * msg_bytes / (BW_nvlink * 1e3)  [ring all-reduce]
    """
    if tp <= 1:
        return 0.0

    T = step.total_prefill_tokens + step.total_decode_tokens
    if T == 0:
        return 0.0

    msg_bytes = T * arch.d * _BYTES_PER_ELEMENT

    if tp == 2:
        return arch.L * 2 * msg_bytes / (hw.bw_nvlink * 1e3)
    else:  # tp >= 4
        return arch.L * 2 * 2 * msg_bytes / (hw.bw_nvlink * 1e3)


# =============================================================================
# Composition (public API)
# =============================================================================

def compute_step_basis(
    step: ReconstructedStep,
    arch: ModelArch,
    hw: HardwareSpec,
    tp: int,
) -> StepBasisValues:
    """Compute all basis function values for one scheduler step.

    Requires: step is a valid ReconstructedStep, arch and hw are loaded,
              tp >= 1.
    Guarantees: returns a StepBasisValues with all fields >= 0,
                num_layers == arch.L, batch_size == step.batch_size.
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")

    return StepBasisValues(
        step_id=step.step_id,
        t_pf_compute=t_pf_compute(step, arch, hw, tp),
        t_pf_kv=t_pf_kv(step, arch, hw, tp),
        t_dc_compute=t_dc_compute(step, arch, hw, tp),
        t_dc_kv=t_dc_kv(step, arch, hw, tp),
        t_weight=t_weight(step, arch, hw, tp),
        t_tp=t_tp(step, arch, hw, tp),
        num_layers=arch.L,
        batch_size=step.batch_size,
    )


def compute_experiment_basis(
    reconstruction: ExperimentReconstruction,
    arch: ModelArch,
    hw: HardwareSpec,
    tp: int,
) -> list[StepBasisValues]:
    """Compute basis function values for every step in an experiment.

    Requires: reconstruction is the output of reconstruct_experiment(exp),
              arch and hw are loaded, tp >= 1.
    Guarantees: returns one StepBasisValues per step in reconstruction.steps,
                in the same order.
    """
    return [compute_step_basis(step, arch, hw, tp) for step in reconstruction.steps]
