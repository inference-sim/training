"""Three-phase NNLS fitting of crossmodel latency model parameters.

Fits 10 parameters (α₀–α₂, β₁–β₇) from vLLM journey trace data using
non-negative least squares regression.

Phase 1: α₀ — API processing overhead (mean of QUEUED.ts − ARRIVED.ts)
Phase 2: α₁, α₂ — Post-decode overhead (NNLS on DEPARTED.ts − FINISHED.ts)
Phase 3: β₁–β₇ — GPU step-time model (regularized NNLS on processing_us)

Public API
----------
fit_coefficients(hw) -> FittedCoefficients
    Fit all 10 parameters from training data, tune λ on validation.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from basis_functions import (
    HardwareSpec,
    ModelArch,
    StepBasisValues,
    compute_experiment_basis,
    load_hardware_spec,
    load_model_arch,
)
from reconstruct_steps import (
    ExperimentReconstruction,
    ReconstructedStep,
    RequestLabel,
    reconstruct_experiment,
)
from split import ExperimentMeta, get_train, get_validate
from trace_parser import (
    attr_map,
    parse_api_events,
    parse_journey_events,
    traces_path_for,
)


# =============================================================================
# Output type (frozen)
# =============================================================================

@dataclass(frozen=True)
class FittedCoefficients:
    """Fitted crossmodel latency model parameters.

    Invariants:
        alpha_0 >= 0, alpha_1 >= 0, alpha_2 >= 0.
        All betas >= 0.
        len(betas) == 7.
        lambda_val >= 0.
        train_mse >= 0, val_mse >= 0.
    """
    alpha_0: float
    alpha_1: float
    alpha_2: float
    betas: tuple[float, ...]
    lambda_val: float
    train_mse: float
    val_mse: float


# =============================================================================
# Phase 1: Estimate α₀
# =============================================================================

def estimate_alpha_0(arrived_queued_pairs: list[tuple[float, float]]) -> float:
    """Estimate API processing overhead as mean(QUEUED.ts − ARRIVED.ts).

    Requires: arrived_queued_pairs is a list of (arrived_ts, queued_ts) in seconds.
              Each queued_ts > arrived_ts.
    Guarantees: returns α₀ in microseconds, α₀ > 0.
    """
    if not arrived_queued_pairs:
        raise ValueError("No arrived/queued pairs provided")

    diffs = [(q - a) * 1e6 for a, q in arrived_queued_pairs]
    alpha_0 = float(np.mean(diffs))

    if alpha_0 <= 0:
        warnings.warn(f"α₀ = {alpha_0:.1f} µs is non-positive, expected > 0")

    return alpha_0


# =============================================================================
# Phase 2: Fit α₁, α₂ (post-decode overhead)
# =============================================================================

def fit_alpha_12(
    output_tokens: np.ndarray,
    post_decode_us: np.ndarray,
) -> tuple[float, float]:
    """Fit post-decode overhead: α₁ + α₂ · output_tokens via NNLS.

    Requires: output_tokens and post_decode_us are 1D arrays of equal length.
              post_decode_us values are in microseconds.
    Guarantees: α₁ >= 0, α₂ >= 0.
    """
    X = np.column_stack([np.ones(len(output_tokens)), output_tokens])
    coeffs, _ = nnls(X, post_decode_us)
    return float(coeffs[0]), float(coeffs[1])


# =============================================================================
# Per-request feature matrix
# =============================================================================

def build_feature_matrix(
    steps: list[ReconstructedStep] | tuple[ReconstructedStep, ...],
    basis_values: list[StepBasisValues],
    labels: list[RequestLabel] | tuple[RequestLabel, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build per-request feature matrix for Phase 3 regression.

    For each non-failed request, sums per-step basis values along the
    request's active steps.

    Requires:
        steps and basis_values are aligned (same length, same step order).
        labels contains one entry per request.
    Guarantees:
        X has shape (n_requests, 7), y has shape (n_requests,).
        All X values >= 0. y values are processing_us from labels.
        Failed requests are excluded.

    Returns: (X, y, request_ids) where request_ids[i] corresponds to row i.
    """
    label_map = {lb.request_id: lb for lb in labels if not lb.failed}

    features: dict[str, list[float]] = {rid: [0.0] * 7 for rid in label_map}
    step_counts: dict[str, int] = {rid: 0 for rid in label_map}

    for step, bv in zip(steps, basis_values):
        for entry in step.prefill_reqs:
            rid = entry.request_id
            if rid not in features:
                continue
            features[rid][0] += max(bv.t_pf_compute, bv.t_pf_kv)
            features[rid][2] += bv.t_weight
            features[rid][3] += bv.t_tp
            features[rid][4] += bv.num_layers
            features[rid][5] += bv.batch_size
            step_counts[rid] += 1

        for entry in step.decode_reqs:
            rid = entry.request_id
            if rid not in features:
                continue
            features[rid][1] += max(bv.t_dc_compute, bv.t_dc_kv)
            features[rid][2] += bv.t_weight
            features[rid][3] += bv.t_tp
            features[rid][4] += bv.num_layers
            features[rid][5] += bv.batch_size
            step_counts[rid] += 1

    req_ids: list[str] = []
    X_rows: list[list[float]] = []
    y_list: list[float] = []

    for rid, feat in features.items():
        if step_counts[rid] == 0:
            continue
        feat[6] = float(step_counts[rid])
        req_ids.append(rid)
        X_rows.append(feat)
        y_list.append(label_map[rid].processing_us)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    return X, y, req_ids


# =============================================================================
# Phase 3: Fit β₁–β₇ (GPU step-time model)
# =============================================================================

def fit_betas(
    X: np.ndarray,
    y: np.ndarray,
    lambda_val: float,
) -> tuple[np.ndarray, float]:
    """Fit GPU step-time coefficients via regularized NNLS.

    minimize ||Xβ - y||² + λ · Σᵢ₌₁⁴ (βᵢ - 1)²  subject to β ≥ 0

    Implemented by augmenting X and y with regularization rows:
        X_aug = vstack([X, √λ · I₄ₓ₇])
        y_aug = hstack([y, √λ · ones(4)])

    Requires: X has shape (n, 7), y has shape (n,), lambda_val >= 0.
    Guarantees: all β >= 0. Returns (betas, residual_norm).
    """
    n_features = X.shape[1]
    assert n_features == 7, f"Expected 7 features, got {n_features}"

    if lambda_val > 0:
        sqrt_lam = np.sqrt(lambda_val)
        reg_X = np.zeros((4, 7))
        reg_X[:4, :4] = sqrt_lam * np.eye(4)
        reg_y = sqrt_lam * np.ones(4)

        X_aug = np.vstack([X, reg_X])
        y_aug = np.concatenate([y, reg_y])
    else:
        X_aug = X
        y_aug = y

    betas, residual = nnls(X_aug, y_aug)
    return betas, float(residual)
