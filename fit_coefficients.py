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


# =============================================================================
# Data collection helpers
# =============================================================================

def _extract_journey_timestamps(
    journey_events: dict[str, list[dict]],
) -> dict[str, dict]:
    """Extract QUEUED and FINISHED timestamps from journey events.

    Returns: {request_id: {"queued_ts": float, "finished_ts": float, "output_tokens": int}}
    Only includes requests with both QUEUED and FINISHED events.
    """
    result: dict[str, dict] = {}
    for req_id, events in journey_events.items():
        queued_ts = None
        finished_ts = None
        output_tokens = 0
        for ev in events:
            attrs = attr_map(ev.get("attributes", []))
            name = ev["name"].replace("journey.", "")
            if name == "QUEUED" and queued_ts is None:
                queued_ts = attrs.get("ts.monotonic", 0.0)
            elif name == "FINISHED":
                finished_ts = attrs.get("ts.monotonic", 0.0)
                output_tokens = int(attrs.get("decode.done_tokens", 0))
        if queued_ts is not None and finished_ts is not None and queued_ts > 0 and finished_ts > 0:
            result[req_id] = {
                "queued_ts": queued_ts,
                "finished_ts": finished_ts,
                "output_tokens": output_tokens,
            }
    return result


def _journey_id_to_base(journey_id: str) -> str:
    """Strip the -0-xxx sequence suffix from a journey request ID.

    Journey IDs: cmpl-xxx-0-yyy -> base: cmpl-xxx
    API IDs: cmpl-xxx (no suffix)
    """
    if "-0-" in journey_id:
        return journey_id.rsplit("-0-", 1)[0]
    return journey_id


def collect_alpha_data(
    experiments: list[ExperimentMeta],
) -> tuple[list[tuple[float, float]], list[tuple[float, float, int]]]:
    """Collect timestamp pairs for alpha_0 and alpha_1/alpha_2 estimation.

    Requires: experiments is a list of ExperimentMeta from split.py.
    Guarantees:
        pairs_0: list of (arrived_ts, queued_ts) in seconds, queued > arrived.
        triples_12: list of (departed_ts, finished_ts, output_tokens),
                     departed > finished, output_tokens > 0.

    Returns: (pairs_0, triples_12)
    """
    pairs_0: list[tuple[float, float]] = []
    triples_12: list[tuple[float, float, int]] = []

    for exp in experiments:
        traces_path = traces_path_for(exp)
        api_events = parse_api_events(traces_path)
        journey_events = parse_journey_events(traces_path)
        journey_ts = _extract_journey_timestamps(journey_events)

        for journey_id, j_data in journey_ts.items():
            base_id = _journey_id_to_base(journey_id)
            if base_id not in api_events:
                continue
            api_data = api_events[base_id]

            # alpha_0 pairs: (arrived_ts, queued_ts)
            arrived = api_data["arrived_ts"]
            queued = j_data["queued_ts"]
            if queued > arrived > 0:
                pairs_0.append((arrived, queued))

            # alpha_1/alpha_2 triples: (departed_ts, finished_ts, output_tokens)
            departed = api_data["departed_ts"]
            finished = j_data["finished_ts"]
            n_tokens = j_data["output_tokens"]
            if departed > finished > 0 and n_tokens > 0:
                triples_12.append((departed, finished, n_tokens))

    return pairs_0, triples_12


# =============================================================================
# Lambda tuning
# =============================================================================

LAMBDA_GRID = [0, 0.01, 0.1, 1.0, 10.0, 100.0]


def tune_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambda_grid: list[float] | None = None,
) -> tuple[float, np.ndarray, float, float]:
    """Tune regularization strength lambda by grid search on validation MSE.

    Requires: X_train, X_val have shape (n, 7). y arrays match row count.
    Guarantees: Returns (best_lambda, best_betas, train_mse, val_mse).
                best_betas has 7 non-negative entries.
    """
    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    best_lambda = 0.0
    best_betas = np.zeros(7)
    best_val_mse = float("inf")
    best_train_mse = float("inf")

    for lam in lambda_grid:
        betas, _ = fit_betas(X_train, y_train, lambda_val=lam)
        train_pred = X_train @ betas
        val_pred = X_val @ betas
        train_mse = float(np.mean((y_train - train_pred) ** 2))
        val_mse = float(np.mean((y_val - val_pred) ** 2))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_train_mse = train_mse
            best_lambda = lam
            best_betas = betas

    return best_lambda, best_betas, best_train_mse, best_val_mse


# =============================================================================
# Per-experiment data collection for Phase 3
# =============================================================================

def _collect_beta_data(
    experiments: list[ExperimentMeta],
    hw: HardwareSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect feature matrix and targets across experiments for Phase 3.

    Returns: (X, y) concatenated across all experiments.
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for exp in experiments:
        result = reconstruct_experiment(exp)
        arch = load_model_arch(f"model_configs/{exp.config_json_dir}/config.json")
        basis = compute_experiment_basis(result, arch, hw, exp.tensor_parallelism)

        X, y, _ = build_feature_matrix(result.steps, basis, result.labels)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    return np.vstack(all_X), np.concatenate(all_y)


# =============================================================================
# Public API
# =============================================================================

BETA_EXPECTED_RANGES = {
    1: (1.5, 3.0, "prefill roofline correction"),
    2: (5.0, 15.0, "decode roofline correction"),
    3: (1.0, 3.0, "weight loading correction"),
    4: (0.5, 2.0, "TP communication correction"),
    5: (10.0, 50.0, "per-layer overhead (us/layer)"),
    6: (50.0, 200.0, "per-request CPU scheduling (us/req)"),
    7: (100.0, 2000.0, "per-step overhead (us)"),
}


def fit_coefficients(hw: HardwareSpec) -> FittedCoefficients:
    """Fit all 10 crossmodel latency parameters from training data.

    Requires: traces.json and exp-config.yaml for all 16 experiments
              exist under default_args/.
    Guarantees: all alpha >= 0, all beta >= 0, lambda chosen by validation MSE.
    """
    train_exps = get_train()
    val_exps = get_validate()

    # Phase 1: alpha_0
    pairs_0, triples_12 = collect_alpha_data(train_exps)
    alpha_0 = estimate_alpha_0(pairs_0)

    # Phase 2: alpha_1, alpha_2
    output_tokens = np.array([t[2] for t in triples_12], dtype=np.float64)
    post_decode_us = np.array([(d - f) * 1e6 for d, f, _ in triples_12], dtype=np.float64)
    alpha_1, alpha_2 = fit_alpha_12(output_tokens, post_decode_us)

    # Phase 3: beta_1-beta_7
    X_train, y_train = _collect_beta_data(train_exps, hw)
    X_val, y_val = _collect_beta_data(val_exps, hw)

    best_lambda, best_betas, train_mse, val_mse = tune_lambda(
        X_train, y_train, X_val, y_val,
    )

    # Warn on out-of-range betas
    for i, b in enumerate(best_betas, 1):
        if i in BETA_EXPECTED_RANGES:
            lo, hi, desc = BETA_EXPECTED_RANGES[i]
            if b < lo or b > hi:
                warnings.warn(
                    f"beta{i} = {b:.4f} outside expected range [{lo}, {hi}] ({desc})"
                )

    return FittedCoefficients(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        betas=tuple(float(b) for b in best_betas),
        lambda_val=best_lambda,
        train_mse=train_mse,
        val_mse=val_mse,
    )
