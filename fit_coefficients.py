"""Three-phase NNLS fitting of crossmodel latency model parameters.

Fits 10 parameters (α₀–α₂, β₁–β₇) from vLLM journey trace data using
non-negative least squares regression.

Phase 1: α₀ — API processing overhead (mean of QUEUED.ts − ARRIVED.ts)
Phase 2: α₁, α₂ — Post-decode overhead (NNLS on DEPARTED.ts − FINISHED.ts)
Phase 3: β₁–β₇ — GPU step-time model (stacked prefill/decode regularized NNLS)

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
    StepBasisValues,
    compute_experiment_basis,
    load_hardware_spec,
    load_model_arch,
)
from reconstruct_steps import (
    ReconstructedStep,
    RequestLabel,
    reconstruct_experiment,
)
from split import ExperimentMeta, Split, get_active, request_split
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

    def __post_init__(self) -> None:
        if len(self.betas) != 7:
            raise ValueError(f"betas must have exactly 7 elements, got {len(self.betas)}")
        for name in ("alpha_0", "alpha_1", "alpha_2", "lambda_val", "train_mse", "val_mse"):
            if getattr(self, name) < 0:
                raise ValueError(f"FittedCoefficients.{name} must be >= 0, got {getattr(self, name)!r}")
        if any(b < 0 for b in self.betas):
            raise ValueError(f"All betas must be >= 0, got {self.betas}")


# =============================================================================
# Phase 1: Estimate α₀
# =============================================================================

def estimate_alpha_0(arrived_queued_pairs: list[tuple[float, float]]) -> float:
    """Estimate API processing overhead as mean(QUEUED.ts − ARRIVED.ts).

    Requires: arrived_queued_pairs is a non-empty list of (arrived_ts, queued_ts) in seconds.
              Each queued_ts > arrived_ts.
    Guarantees: returns α₀ in microseconds. Warns if α₀ <= 0 (unexpected data).
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
# Stacked prefill/decode feature matrix
# =============================================================================

def build_stacked_feature_matrix(
    steps: list[ReconstructedStep] | tuple[ReconstructedStep, ...],
    basis_values: list[StepBasisValues],
    labels: list[RequestLabel] | tuple[RequestLabel, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build stacked prefill+decode feature matrix for Phase 3 regression.

    For each non-failed request, produces two rows:
    - Prefill row: basis values summed over steps where request is prefilling
    - Decode row: basis values summed over steps where request is decoding

    Selective accumulation: β₁ (prefill roofline) only from prefill entries,
    β₂ (decode roofline) only from decode entries. Shared β₃-β₇ from both.

    Requires:
        steps and basis_values are aligned (same length, same step order).
        labels contains one entry per request with prefill_processing_us
        and decode_processing_us fields.
    Guarantees:
        X has shape (2 * n_requests, 7), y has shape (2 * n_requests,).
        First n_requests rows are prefill, last n_requests rows are decode.
        X_pf[i] + X_dc[i] == X_total[i] for all i (feature sum invariant).
        Failed requests are excluded.

    Returns: (X, y, request_ids) where request_ids has n_requests entries
             (each appearing once, covering both their prefill and decode rows).
    """
    label_map = {lb.request_id: lb for lb in labels if not lb.failed}

    pf_features: dict[str, list[float]] = {rid: [0.0] * 7 for rid in label_map}
    dc_features: dict[str, list[float]] = {rid: [0.0] * 7 for rid in label_map}
    pf_step_counts: dict[str, int] = {rid: 0 for rid in label_map}
    dc_step_counts: dict[str, int] = {rid: 0 for rid in label_map}

    for step, bv in zip(steps, basis_values):
        for entry in step.prefill_reqs:
            rid = entry.request_id
            if rid not in pf_features:
                continue
            # β₁ (prefill roofline) — only from prefill entries
            pf_features[rid][0] += max(bv.t_pf_compute, bv.t_pf_kv)
            # β₂ (decode roofline) — NOT accumulated for prefill entries
            # Shared β₃-β₇
            pf_features[rid][2] += bv.t_weight
            pf_features[rid][3] += bv.t_tp
            pf_features[rid][4] += bv.num_layers
            pf_features[rid][5] += bv.batch_size
            pf_step_counts[rid] += 1

        for entry in step.decode_reqs:
            rid = entry.request_id
            if rid not in dc_features:
                continue
            # β₁ (prefill roofline) — NOT accumulated for decode entries
            # β₂ (decode roofline) — only from decode entries
            dc_features[rid][1] += max(bv.t_dc_compute, bv.t_dc_kv)
            # Shared β₃-β₇
            dc_features[rid][2] += bv.t_weight
            dc_features[rid][3] += bv.t_tp
            dc_features[rid][4] += bv.num_layers
            dc_features[rid][5] += bv.batch_size
            dc_step_counts[rid] += 1

    req_ids: list[str] = []
    pf_rows: list[list[float]] = []
    dc_rows: list[list[float]] = []
    y_pf: list[float] = []
    y_dc: list[float] = []

    for rid in label_map:
        total_steps = pf_step_counts[rid] + dc_step_counts[rid]
        if total_steps == 0:
            continue
        pf_features[rid][6] = float(pf_step_counts[rid])
        dc_features[rid][6] = float(dc_step_counts[rid])
        req_ids.append(rid)
        pf_rows.append(pf_features[rid])
        dc_rows.append(dc_features[rid])
        y_pf.append(label_map[rid].prefill_processing_us)
        y_dc.append(label_map[rid].decode_processing_us)

    if not pf_rows:
        empty_X = np.empty((0, 7), dtype=np.float64)
        empty_y = np.empty(0, dtype=np.float64)
        return empty_X, empty_y, []

    X = np.vstack([np.array(pf_rows, dtype=np.float64),
                   np.array(dc_rows, dtype=np.float64)])
    y = np.concatenate([np.array(y_pf, dtype=np.float64),
                        np.array(y_dc, dtype=np.float64)])
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
    if n_features != 7:
        raise ValueError(f"Expected 7 features, got {n_features}")

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


def collect_alpha_data() -> tuple[list[tuple[float, float]], list[tuple[float, float, int]]]:
    """Collect timestamp pairs for α₀ and α₁/α₂ estimation from training requests.

    Iterates all active experiments, filters to training-split requests only.

    Guarantees:
        pairs_0: list of (arrived_ts, queued_ts) in seconds, queued > arrived.
        triples_12: list of (departed_ts, finished_ts, output_tokens),
                     departed > finished, output_tokens > 0.

    Returns: (pairs_0, triples_12)
    """
    pairs_0: list[tuple[float, float]] = []
    triples_12: list[tuple[float, float, int]] = []

    for exp in get_active():
        traces_path = traces_path_for(exp)
        api_events = parse_api_events(traces_path)
        journey_events = parse_journey_events(traces_path)
        journey_ts = _extract_journey_timestamps(journey_events)

        for journey_id, j_data in journey_ts.items():
            # Only use training-split requests
            if request_split(journey_id) != Split.TRAIN:
                continue

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
    hw: HardwareSpec,
    split_filter: Split,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect stacked feature matrix and targets for Phase 3.

    Iterates all active experiments, includes only requests matching split_filter.

    Returns: (X, y) — stacked prefill+decode system.
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for exp in get_active():
        result = reconstruct_experiment(exp)
        arch = load_model_arch(f"model_configs/{exp.config_json_dir}/config.json")
        basis = compute_experiment_basis(result, arch, hw, exp.tensor_parallelism)

        # Filter labels to matching split only
        filtered_labels = [
            lb for lb in result.labels
            if not lb.failed and request_split(lb.request_id) == split_filter
        ]

        X, y, _ = build_stacked_feature_matrix(result.steps, basis, filtered_labels)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        raise ValueError(
            f"No usable {split_filter.value} data from {len(get_active())} experiments."
        )
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

    Requires: traces.json and exp-config.yaml for all active experiments
              exist under default_args/.
    Guarantees: all alpha >= 0, all beta >= 0, lambda chosen by validation MSE.
    """
    # Phase 1: α₀
    pairs_0, triples_12 = collect_alpha_data()
    alpha_0 = estimate_alpha_0(pairs_0)

    # Phase 2: α₁, α₂
    output_tokens = np.array([t[2] for t in triples_12], dtype=np.float64)
    post_decode_us = np.array([(d - f) * 1e6 for d, f, _ in triples_12], dtype=np.float64)
    alpha_1, alpha_2 = fit_alpha_12(output_tokens, post_decode_us)

    # Phase 3: β₁-β₇ (stacked prefill/decode)
    X_train, y_train = _collect_beta_data(hw, Split.TRAIN)
    X_val, y_val = _collect_beta_data(hw, Split.VALIDATE)

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


# =============================================================================
# Diagnostics output
# =============================================================================

OUTPUT_DIR = Path("output/fit")


def write_diagnostics(
    coeffs: FittedCoefficients,
    hw: HardwareSpec,
) -> None:
    """Write fitting diagnostics to output/fit/.

    Requires: coeffs is a fitted FittedCoefficients, hw is the hardware spec used.
    Guarantees: writes coefficients.json, lambda_tuning.json, and residuals.json
                to output/fit/, and prints summary to stdout.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON output
    result = {
        "alpha_0_us": coeffs.alpha_0,
        "alpha_1_us": coeffs.alpha_1,
        "alpha_2_us_per_token": coeffs.alpha_2,
        "betas": list(coeffs.betas),
        "lambda": coeffs.lambda_val,
        "train_mse": coeffs.train_mse,
        "val_mse": coeffs.val_mse,
    }
    with open(OUTPUT_DIR / "coefficients.json", "w") as f:
        json.dump(result, f, indent=2)

    # Lambda tuning curve
    X_train, y_train = _collect_beta_data(hw, Split.TRAIN)
    X_val, y_val = _collect_beta_data(hw, Split.VALIDATE)

    tuning_curve: list[dict] = []
    for lam in LAMBDA_GRID:
        betas, _ = fit_betas(X_train, y_train, lambda_val=lam)
        t_mse = float(np.mean((y_train - X_train @ betas) ** 2))
        v_mse = float(np.mean((y_val - X_val @ betas) ** 2))
        tuning_curve.append({"lambda": lam, "train_mse": t_mse, "val_mse": v_mse})

    with open(OUTPUT_DIR / "lambda_tuning.json", "w") as f:
        json.dump(tuning_curve, f, indent=2)

    # Per-experiment residual summary
    residuals: list[dict] = []
    for exp in get_active():
        rec = reconstruct_experiment(exp)
        arch = load_model_arch(f"model_configs/{exp.config_json_dir}/config.json")
        basis = compute_experiment_basis(rec, arch, hw, exp.tensor_parallelism)
        X, y, _ = build_stacked_feature_matrix(rec.steps, basis, rec.labels)
        if len(X) == 0:
            continue
        n = len(y) // 2
        pred = X @ np.array(coeffs.betas)
        resid = y - pred
        pf_rmse = float(np.sqrt(np.mean(resid[:n] ** 2))) if n > 0 else 0.0
        dc_rmse = float(np.sqrt(np.mean(resid[n:] ** 2))) if n > 0 else 0.0
        residuals.append({
            "experiment": exp.dir_name,
            "n_requests": n,
            "prefill_rmse_us": pf_rmse,
            "decode_rmse_us": dc_rmse,
            "combined_rmse_us": float(np.sqrt(np.mean(resid ** 2))),
            "mean_y_us": float(np.mean(y)),
        })

    with open(OUTPUT_DIR / "residuals.json", "w") as f:
        json.dump(residuals, f, indent=2)

    # Print summary
    print("=" * 60)
    print("  Fitted Crossmodel Latency Parameters")
    print("=" * 60)
    print(f"\n  Phase 1 — API processing overhead:")
    print(f"    α₀ = {coeffs.alpha_0:,.1f} µs ({coeffs.alpha_0/1000:.1f} ms)")
    print(f"\n  Phase 2 — Post-decode overhead:")
    print(f"    α₁ = {coeffs.alpha_1:,.1f} µs (fixed per-request)")
    print(f"    α₂ = {coeffs.alpha_2:,.2f} µs/token (detokenization)")
    print(f"\n  Phase 3 — GPU step-time model (λ = {coeffs.lambda_val}):")
    beta_names = [
        "prefill roofline", "decode roofline", "weight loading",
        "TP communication", "per-layer overhead", "per-request scheduling",
        "per-step overhead",
    ]
    for i, (b, name) in enumerate(zip(coeffs.betas, beta_names), 1):
        lo, hi, _ = BETA_EXPECTED_RANGES[i]
        flag = " ⚠" if b < lo or b > hi else ""
        print(f"    β{i} = {b:>10.4f}  [{lo:>5.1f}, {hi:>6.1f}]  {name}{flag}")
    print(f"\n  MSE (stacked prefill+decode):")
    print(f"    Train:    {coeffs.train_mse:>14,.0f} µs²")
    print(f"    Validate: {coeffs.val_mse:>14,.0f} µs²")
    print(f"    RMSE:     {np.sqrt(coeffs.val_mse):>14,.0f} µs (validation)")
    print(f"\n  Output written to {OUTPUT_DIR}/")
    print("=" * 60)


def main() -> int:
    hw = load_hardware_spec("datasheets/h100-sxm.json")
    coeffs = fit_coefficients(hw)
    write_diagnostics(coeffs, hw)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
