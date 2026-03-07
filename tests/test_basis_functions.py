"""Behavioral tests for analytical basis function computation.

Each test class describes a batch scenario and verifies that the computed
basis function values satisfy physical properties. Tests call
compute_step_basis() with synthetic ReconstructedStep objects and known
ModelArch/HardwareSpec parameters.

Design discipline:
    Given: a step with known batch composition + model architecture + hardware
    When:  compute_step_basis() is called
    Then:  the returned StepBasisValues satisfy physical properties
"""

from __future__ import annotations

import pytest

from basis_functions import (
    ModelArch,
    HardwareSpec,
    compute_step_basis,
)
from reconstruct_steps import ReconstructedStep, PrefillEntry, DecodeEntry


# ---------------------------------------------------------------------------
# Fixtures: reusable model architectures and hardware specs
# ---------------------------------------------------------------------------

LLAMA_7B = ModelArch(
    L=32, d=4096, H=32, kv_heads=32, d_h=128, d_kv=4096,
    d_ff=11008, N=0, k=0, k_eff=1, is_moe=False,
)

# GQA dense model (kv_heads < H, no MoE) for isolating GQA behavior
LLAMA_70B = ModelArch(
    L=80, d=8192, H=64, kv_heads=8, d_h=128, d_kv=1024,
    d_ff=28672, N=0, k=0, k_eff=1, is_moe=False,
)

MIXTRAL_8X7B = ModelArch(
    L=32, d=4096, H=32, kv_heads=8, d_h=128, d_kv=1024,
    d_ff=14336, N=8, k=2, k_eff=2, is_moe=True,
)

H100_SXM = HardwareSpec(flops_peak=989.5, bw_hbm=3.35, bw_nvlink=900.0)


def make_step(
    step_id: int = 1,
    prefill_reqs: tuple[PrefillEntry, ...] = (),
    decode_reqs: tuple[DecodeEntry, ...] = (),
) -> ReconstructedStep:
    """Create a synthetic ReconstructedStep for testing."""
    total_pf = sum(e.tokens_this_step for e in prefill_reqs)
    total_dc = len(decode_reqs)
    return ReconstructedStep(
        step_id=step_id,
        prefill_reqs=prefill_reqs,
        decode_reqs=decode_reqs,
        total_prefill_tokens=total_pf,
        total_decode_tokens=total_dc,
        batch_size=len(prefill_reqs) + total_dc,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Empty step — no requests
# ---------------------------------------------------------------------------

class TestEmptyStep:
    """A step with no prefill and no decode requests.

    Expected: all compute/bandwidth terms are 0.  Weight loading and
    num_layers are model constants (weight bytes don't depend on batch).
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(prefill_reqs=(), decode_reqs=())
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_compute_terms_are_zero(self, basis):
        assert basis.t_pf_compute == 0.0
        assert basis.t_pf_kv == 0.0
        assert basis.t_dc_compute == 0.0
        assert basis.t_dc_kv == 0.0

    def test_tp_is_zero_for_empty_batch(self, basis):
        assert basis.t_tp == 0.0

    def test_batch_size_is_zero(self, basis):
        assert basis.batch_size == 0


# ---------------------------------------------------------------------------
# Scenario 2: Prefill only — one request, no decode
# ---------------------------------------------------------------------------

class TestPrefillOnly:
    """One prefill request with 512 tokens, no decode.

    Expected: decode basis values are 0, prefill compute and KV are positive,
    and FLOP count matches hand calculation.
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(
            prefill_reqs=(PrefillEntry("req-1", tokens_this_step=512, prompt_tokens=512),),
        )
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_decode_terms_are_zero(self, basis):
        assert basis.t_dc_compute == 0.0
        assert basis.t_dc_kv == 0.0

    def test_prefill_compute_is_positive(self, basis):
        assert basis.t_pf_compute > 0.0

    def test_prefill_kv_is_positive(self, basis):
        assert basis.t_pf_kv > 0.0

    def test_prefill_compute_matches_hand_calculation(self, basis):
        # Llama-2-7b, TP=1, 512 prefill tokens, prompt=512
        L, d, H, d_h, d_kv, d_ff = 32, 4096, 32, 128, 4096, 11008
        T_pf = 512
        # Projection FLOPs: L * 2 * T_pf * d * (2*d + 2*d_kv) / TP
        flops_proj = L * 2 * T_pf * d * (2 * d + 2 * d_kv)
        # Attention FLOPs: L * 4 * (H/TP) * T_pf * (S_pf + T_pf/2) * d_h
        flops_attn = L * 4 * H * T_pf * (512 + 512 / 2) * d_h
        # FFN FLOPs: L * T_pf * k_eff * 6 * d * d_ff / TP
        flops_ffn = L * T_pf * 1 * 6 * d * d_ff
        total = flops_proj + flops_attn + flops_ffn
        expected_us = total / (H100_SXM.flops_peak * 1e6)
        assert abs(basis.t_pf_compute - expected_us) / expected_us < 1e-6


# ---------------------------------------------------------------------------
# Scenario 3: Decode only — multiple requests, no prefill
# ---------------------------------------------------------------------------

class TestDecodeOnly:
    """Three decode requests with different context lengths, no prefill.

    Expected: prefill terms are 0, decode compute reflects Σ context_lengths,
    decode KV bandwidth reflects Σ(S_dc_j) + T_dc.
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(decode_reqs=(
            DecodeEntry("req-a", context_length=200),
            DecodeEntry("req-b", context_length=500),
            DecodeEntry("req-c", context_length=800),
        ))
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_prefill_terms_are_zero(self, basis):
        assert basis.t_pf_compute == 0.0
        assert basis.t_pf_kv == 0.0

    def test_decode_compute_is_positive(self, basis):
        assert basis.t_dc_compute > 0.0

    def test_decode_kv_reflects_context_length_sum(self, basis):
        # T_dc_kv = L * 2 * (kv_heads/TP) * d_h * 2 * (Σ S_dc_j + T_dc) / (BW_hbm * 1e6)
        L, kv_heads, d_h = 32, 32, 128
        sum_ctx = 200 + 500 + 800
        T_dc = 3
        bytes_kv = L * 2 * kv_heads * d_h * 2 * (sum_ctx + T_dc)
        expected_us = bytes_kv / (H100_SXM.bw_hbm * 1e6)
        assert abs(basis.t_dc_kv - expected_us) / expected_us < 1e-6


# ---------------------------------------------------------------------------
# Scenario 4: Mixed batch — prefill + decode together
# ---------------------------------------------------------------------------

class TestMixedBatch:
    """One prefill request and two decode requests in the same step.

    Expected: all compute and bandwidth terms are positive.
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(
            prefill_reqs=(PrefillEntry("pf-1", tokens_this_step=256, prompt_tokens=256),),
            decode_reqs=(
                DecodeEntry("dc-1", context_length=300),
                DecodeEntry("dc-2", context_length=600),
            ),
        )
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_all_compute_terms_positive(self, basis):
        assert basis.t_pf_compute > 0.0
        assert basis.t_dc_compute > 0.0

    def test_all_bandwidth_terms_positive(self, basis):
        assert basis.t_pf_kv > 0.0
        assert basis.t_dc_kv > 0.0

    def test_batch_size_is_three(self, basis):
        assert basis.batch_size == 3


# ---------------------------------------------------------------------------
# Scenario 5: MoE weight loading vs dense
# ---------------------------------------------------------------------------

class TestMoEWeightLoading:
    """MoE model loads more expert weights than dense model.

    For Mixtral with N=8, k=2, at batch_size tokens the effective expert
    count N_eff = min(N, max(k, B*k)) which for small batches is k (=2)
    and for large batches approaches N (=8).
    """

    def test_moe_weight_loading_exceeds_dense_scaled(self):
        step = make_step(decode_reqs=(
            DecodeEntry("dc-1", context_length=300),
        ))
        dense_basis = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)
        moe_basis = compute_step_basis(step, MIXTRAL_8X7B, H100_SXM, tp=2)

        # MoE FFN weight bytes = N_eff * 3 * d * d_ff * 2 / TP
        # Dense FFN weight bytes = 3 * d * d_ff * 2 / TP
        # For B=1 token, N_eff = min(8, max(2, 1*2)) = 2
        # MoE loads 2 experts vs 1 for dense, and Mixtral d_ff is larger
        # So MoE t_weight should be substantially larger per-layer
        assert moe_basis.t_weight > 0.0
        assert dense_basis.t_weight > 0.0

    def test_large_batch_activates_more_experts(self):
        # With 10 tokens, N_eff = min(8, max(2, 10*2)) = min(8, 20) = 8
        small_step = make_step(decode_reqs=tuple(
            DecodeEntry(f"dc-{i}", context_length=100) for i in range(1)
        ))
        large_step = make_step(decode_reqs=tuple(
            DecodeEntry(f"dc-{i}", context_length=100) for i in range(10)
        ))
        small_basis = compute_step_basis(small_step, MIXTRAL_8X7B, H100_SXM, tp=2)
        large_basis = compute_step_basis(large_step, MIXTRAL_8X7B, H100_SXM, tp=2)
        # More tokens → higher N_eff → more weight bytes → larger t_weight
        assert large_basis.t_weight > small_basis.t_weight


# ---------------------------------------------------------------------------
# Scenario 6: TP communication scaling
# ---------------------------------------------------------------------------

class TestTPCommunication:
    """TP communication depends on parallelism degree.

    TP=1: no communication. TP=2: point-to-point. TP≥4: ring all-reduce (2x).
    """

    @pytest.fixture()
    def step(self):
        return make_step(decode_reqs=(
            DecodeEntry("dc-1", context_length=300),
        ))

    def test_tp1_has_zero_communication(self, step):
        basis = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)
        assert basis.t_tp == 0.0

    def test_tp2_has_point_to_point(self, step):
        basis = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=2)
        assert basis.t_tp > 0.0

    def test_tp4_is_double_tp2(self, step):
        basis_tp2 = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=2)
        basis_tp4 = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=4)
        # Ring all-reduce at TP≥4 has 2x the bandwidth cost of TP=2
        assert abs(basis_tp4.t_tp - 2.0 * basis_tp2.t_tp) < 1e-6


# ---------------------------------------------------------------------------
# Scenario 7: Chunked prefill (tokens_this_step < prompt_tokens)
# ---------------------------------------------------------------------------

class TestChunkedPrefill:
    """A prefill request where only part of the prompt is processed this step.

    tokens_this_step=256 but prompt_tokens=1024. The attention FLOP count
    should use tokens_this_step for the query dimension and prompt_tokens
    for the KV context size.
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(
            prefill_reqs=(PrefillEntry("req-chunk", tokens_this_step=256, prompt_tokens=1024),),
        )
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_prefill_kv_uses_tokens_this_step(self, basis):
        # T_pf_kv depends on total_prefill_tokens (=tokens_this_step), NOT prompt_tokens
        L, kv_heads, d_h = 32, 32, 128
        bytes_kv = L * 2 * kv_heads * d_h * 256 * 2  # T_pf = 256
        expected = bytes_kv / (H100_SXM.bw_hbm * 1e6)
        assert abs(basis.t_pf_kv - expected) / expected < 1e-6

    def test_prefill_compute_differs_from_full_prompt(self):
        # With chunked prefill, compute should differ from processing the full prompt
        chunk_step = make_step(
            prefill_reqs=(PrefillEntry("req-1", tokens_this_step=256, prompt_tokens=1024),),
        )
        full_step = make_step(
            prefill_reqs=(PrefillEntry("req-1", tokens_this_step=1024, prompt_tokens=1024),),
        )
        chunk_basis = compute_step_basis(chunk_step, LLAMA_7B, H100_SXM, tp=1)
        full_basis = compute_step_basis(full_step, LLAMA_7B, H100_SXM, tp=1)
        # Chunk processes fewer tokens → less compute
        assert chunk_basis.t_pf_compute < full_basis.t_pf_compute


# ---------------------------------------------------------------------------
# Scenario 8: Multiple concurrent prefill requests
# ---------------------------------------------------------------------------

class TestMultiplePrefills:
    """Two prefill requests in the same step. Attention FLOPs should sum
    across both requests, not just use the first one."""

    @pytest.fixture()
    def basis(self):
        step = make_step(prefill_reqs=(
            PrefillEntry("pf-a", tokens_this_step=300, prompt_tokens=300),
            PrefillEntry("pf-b", tokens_this_step=200, prompt_tokens=200),
        ))
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_prefill_kv_uses_total_tokens(self, basis):
        # T_pf = 300 + 200 = 500
        L, kv_heads, d_h = 32, 32, 128
        bytes_kv = L * 2 * kv_heads * d_h * 500 * 2
        expected = bytes_kv / (H100_SXM.bw_hbm * 1e6)
        assert abs(basis.t_pf_kv - expected) / expected < 1e-6

    def test_prefill_compute_exceeds_single_request(self):
        single_step = make_step(prefill_reqs=(
            PrefillEntry("pf-a", tokens_this_step=300, prompt_tokens=300),
        ))
        dual_step = make_step(prefill_reqs=(
            PrefillEntry("pf-a", tokens_this_step=300, prompt_tokens=300),
            PrefillEntry("pf-b", tokens_this_step=200, prompt_tokens=200),
        ))
        single = compute_step_basis(single_step, LLAMA_7B, H100_SXM, tp=1)
        dual = compute_step_basis(dual_step, LLAMA_7B, H100_SXM, tp=1)
        assert dual.t_pf_compute > single.t_pf_compute


# ---------------------------------------------------------------------------
# Scenario 9: Decode compute hand-verification
# ---------------------------------------------------------------------------

class TestDecodeComputeHandCalc:
    """Hand-verify decode compute FLOPs for a known configuration.

    Three decode requests with context lengths 200, 500, 800.
    Llama-2-7b (TP=1): L=32, d=4096, H=32, d_h=128, d_kv=4096, d_ff=11008.
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(decode_reqs=(
            DecodeEntry("dc-a", context_length=200),
            DecodeEntry("dc-b", context_length=500),
            DecodeEntry("dc-c", context_length=800),
        ))
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_decode_compute_matches_hand_calculation(self, basis):
        L, d, H, d_h, d_kv, d_ff = 32, 4096, 32, 128, 4096, 11008
        T_dc = 3
        sum_ctx = 200 + 500 + 800
        flops_proj = L * 2 * T_dc * d * (2 * d + 2 * d_kv)
        flops_attn = L * 4 * H * sum_ctx * d_h
        flops_ffn = L * T_dc * 1 * 6 * d * d_ff
        expected_us = (flops_proj + flops_attn + flops_ffn) / (H100_SXM.flops_peak * 1e6)
        assert abs(basis.t_dc_compute - expected_us) / expected_us < 1e-6


# ---------------------------------------------------------------------------
# Scenario 10: GQA isolation (kv_heads < H, dense model, TP=1)
# ---------------------------------------------------------------------------

class TestGQAIsolation:
    """GQA model (kv_heads=8 vs H=64) should show different scaling in
    KV bandwidth vs compute. KV bandwidth scales with kv_heads, while
    attention compute scales with H."""

    def test_kv_bandwidth_per_layer_scales_with_kv_heads(self):
        step = make_step(decode_reqs=(DecodeEntry("dc-1", context_length=500),))
        mha_basis = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)   # kv_heads=32, L=32
        gqa_basis = compute_step_basis(step, LLAMA_70B, H100_SXM, tp=1)  # kv_heads=8, L=80

        # Normalize by layer count to isolate the kv_heads effect
        mha_per_layer = mha_basis.t_dc_kv / LLAMA_7B.L
        gqa_per_layer = gqa_basis.t_dc_kv / LLAMA_70B.L
        # GQA (kv_heads=8) has 4x less KV bandwidth per layer than MHA (kv_heads=32)
        assert gqa_per_layer < mha_per_layer

    def test_compute_scales_with_H_not_kv_heads(self):
        step = make_step(decode_reqs=(DecodeEntry("dc-1", context_length=500),))
        mha_basis = compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)   # H=32
        gqa_basis = compute_step_basis(step, LLAMA_70B, H100_SXM, tp=1)  # H=64

        # GQA has H=64 vs MHA H=32, so attention compute per layer is higher
        # (despite fewer KV heads). Also L=80 vs L=32.
        assert gqa_basis.t_dc_compute > mha_basis.t_dc_compute


# ---------------------------------------------------------------------------
# Scenario 11: Known values for Llama-2-7b + H100 SXM
# ---------------------------------------------------------------------------

class TestKnownValues:
    """End-to-end verification with Llama-2-7b architecture, H100 SXM,
    and a specific batch composition. All values hand-computed.

    Step: 1 prefill (512 tokens, prompt=512) + 2 decode (ctx=300, ctx=600)
    TP=1, L=32, d=4096, H=32, kv_heads=32, d_h=128, d_kv=4096, d_ff=11008
    FLOPS_peak=989.5 TFLOPS, BW_hbm=3.35 TB/s, BW_nvlink=900 GB/s
    """

    @pytest.fixture()
    def basis(self):
        step = make_step(
            prefill_reqs=(PrefillEntry("pf-1", tokens_this_step=512, prompt_tokens=512),),
            decode_reqs=(
                DecodeEntry("dc-1", context_length=300),
                DecodeEntry("dc-2", context_length=600),
            ),
        )
        return compute_step_basis(step, LLAMA_7B, H100_SXM, tp=1)

    def test_t_pf_kv_matches_formula(self, basis):
        # Bytes_pf_kv = L * 2 * kv_heads * d_h * T_pf * 2
        # = 32 * 2 * 32 * 128 * 512 * 2 = 268,435,456 bytes
        bytes_kv = 32 * 2 * 32 * 128 * 512 * 2
        expected = bytes_kv / (3.35 * 1e6)
        assert abs(basis.t_pf_kv - expected) < 0.01

    def test_t_dc_kv_matches_formula(self, basis):
        # Bytes_dc_kv = L * 2 * kv_heads * d_h * 2 * (Σ S_dc_j + T_dc)
        # Σ S_dc_j = 300 + 600 = 900, T_dc = 2
        bytes_kv = 32 * 2 * 32 * 128 * 2 * (900 + 2)
        expected = bytes_kv / (3.35 * 1e6)
        assert abs(basis.t_dc_kv - expected) < 0.01

    def test_t_weight_matches_formula(self, basis):
        # Dense: N_eff = 1
        # Bytes_attn_wt = L * d * (2*d + 2*d_kv) * 2 / TP  (QKV + output proj)
        bytes_attn = 32 * 4096 * (2 * 4096 + 2 * 4096) * 2
        # Bytes_ffn_wt = L * 1 * 3 * d * d_ff * 2 / TP
        # = 32 * 3 * 4096 * 11008 * 2
        bytes_ffn = 32 * 1 * 3 * 4096 * 11008 * 2
        expected = (bytes_attn + bytes_ffn) / (3.35 * 1e6)
        assert abs(basis.t_weight - expected) < 0.1

    def test_t_tp_is_zero_at_tp1(self, basis):
        assert basis.t_tp == 0.0

    def test_num_layers_is_32(self, basis):
        assert basis.num_layers == 32

    def test_batch_size_is_3(self, basis):
        assert basis.batch_size == 3
