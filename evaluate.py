"""Comprehensive evaluation of the crossmodel latency model.

Assesses prediction quality for each component independently across all three
data splits (train/validate/test).

Evaluation matrix: 5 measures × 3 metrics × 3 splits = 45 cells.

Measures:
    Pre-queueing:  α₀ (constant) vs (QUEUED.ts − ARRIVED.ts) × 1e6
    Post-decode:   α₁ + α₂·n vs (DEPARTED.ts − FINISHED.ts) × 1e6
    GPU prefill:   X_pf @ β vs prefill_processing_us
    GPU decode:    X_dc @ β vs decode_processing_us
    GPU combined:  (X_pf + X_dc) @ β vs processing_us

Metrics: MAPE (%), RMSE (µs), MAE (µs).

Public API
----------
compute_metrics(predicted, observed) -> MeasureMetrics
    Compute MAPE, RMSE, MAE from predicted vs observed arrays.

evaluate(coeffs, hw) -> EvaluationResult
    Evaluate all 5 measures across train/validate/test splits.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from basis_functions import (
    HardwareSpec,
    compute_experiment_basis,
    load_hardware_spec,
    load_model_arch,
)
from fit_coefficients import (
    FittedCoefficients,
    build_stacked_feature_matrix,
    collect_alpha_data,
    fit_coefficients,
)
from reconstruct_steps import reconstruct_experiment
from split import Split, get_active, request_split


# =============================================================================
# Output types (frozen)
# =============================================================================

@dataclass(frozen=True)
class MeasureMetrics:
    """Metrics for one measure on one split.

    Invariants:
        mape >= 0, rmse >= 0, mae >= 0, n >= 0.
        mape is in percentage points (0-100+).
        rmse and mae are in µs.
    """
    mape: float
    rmse: float
    mae: float
    n: int

    def __post_init__(self) -> None:
        import math
        for name in ("mape", "rmse", "mae"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise ValueError(f"MeasureMetrics.{name} must be finite, got {val!r}")
            if val < 0:
                raise ValueError(f"MeasureMetrics.{name} must be >= 0, got {val!r}")
        if self.n < 0:
            raise ValueError(f"MeasureMetrics.n must be >= 0, got {self.n}")


_EXPECTED_SPLITS = frozenset({"train", "validate", "test"})


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation across all measures and splits.

    Invariants:
        Each dict has exactly keys {"train", "validate", "test"}.
        Every MeasureMetrics is expected to have n > 0 for the current
        dataset (verified by tests).
    """
    pre_queueing: dict[str, MeasureMetrics]
    post_decode: dict[str, MeasureMetrics]
    gpu_prefill: dict[str, MeasureMetrics]
    gpu_decode: dict[str, MeasureMetrics]
    gpu_combined: dict[str, MeasureMetrics]

    def __post_init__(self) -> None:
        for name in ("pre_queueing", "post_decode", "gpu_prefill", "gpu_decode", "gpu_combined"):
            d = getattr(self, name)
            if set(d.keys()) != _EXPECTED_SPLITS:
                raise ValueError(
                    f"EvaluationResult.{name} must have keys {_EXPECTED_SPLITS}, "
                    f"got {set(d.keys())}"
                )


# =============================================================================
# Core metric computation
# =============================================================================

def compute_metrics(predicted: np.ndarray, observed: np.ndarray) -> MeasureMetrics:
    """Compute MAPE, RMSE, MAE from predicted vs observed arrays.

    Requires: predicted and observed are 1D arrays of equal length.
    Guarantees: MAPE excludes observations where |observed| < 1.0 µs
                (avoids division by near-zero). n = len(predicted).
                Returns MeasureMetrics with mape=0, rmse=0, mae=0 if
                input arrays are empty.
    """
    if predicted.shape != observed.shape:
        raise ValueError(
            f"compute_metrics: shape mismatch — predicted {predicted.shape} "
            f"!= observed {observed.shape}"
        )
    n = len(predicted)
    if n == 0:
        return MeasureMetrics(mape=0.0, rmse=0.0, mae=0.0, n=0)

    errors = predicted - observed
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # MAPE: skip near-zero observations to avoid division by near-zero
    mask = np.abs(observed) >= 1.0
    if np.any(mask):
        mape = float(np.mean(np.abs(errors[mask] / observed[mask])) * 100)
    else:
        mape = 0.0

    return MeasureMetrics(mape=mape, rmse=rmse, mae=mae, n=n)


# =============================================================================
# Data collection for alpha measures
# =============================================================================

def _collect_alpha_eval_data(
    coeffs: FittedCoefficients,
    split_filter: Split,
) -> tuple[MeasureMetrics, MeasureMetrics]:
    """Collect and evaluate alpha measures for one split.

    Requires: coeffs is a valid FittedCoefficients, split_filter is a Split enum value.
    Guarantees: returns (pre_queueing_metrics, post_decode_metrics) where each is
                a MeasureMetrics. Returns n=0 metrics if no data exists for the split.
    """
    pairs_0, triples_12 = collect_alpha_data(split_filter)

    # Pre-queueing: observed = (q - a) * 1e6, predicted = α₀ (broadcast)
    if pairs_0:
        observed_0 = np.array([(q - a) * 1e6 for a, q in pairs_0])
        predicted_0 = np.full(len(observed_0), coeffs.alpha_0)
        pre_q = compute_metrics(predicted_0, observed_0)
    else:
        pre_q = MeasureMetrics(mape=0.0, rmse=0.0, mae=0.0, n=0)

    # Post-decode: observed = (d - f) * 1e6, predicted = α₁ + α₂ * n
    if triples_12:
        observed_12 = np.array([(d - f) * 1e6 for d, f, _ in triples_12])
        predicted_12 = np.array(
            [coeffs.alpha_1 + coeffs.alpha_2 * n for _, _, n in triples_12]
        )
        post_d = compute_metrics(predicted_12, observed_12)
    else:
        post_d = MeasureMetrics(mape=0.0, rmse=0.0, mae=0.0, n=0)

    return pre_q, post_d


# =============================================================================
# Data collection for GPU measures
# =============================================================================

def _collect_gpu_eval_data(
    hw: HardwareSpec,
    betas: np.ndarray,
    split_filter: Split,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect GPU predictions and observations for one split.

    Requires: betas is a 1D array of length 7, hw is a valid HardwareSpec.
    Guarantees: returns (pf_pred, pf_obs, dc_pred, dc_obs, combined_pred, combined_obs).
                All arrays are 1D with one entry per request.
                Returns empty arrays (length 0) if no data matches the split filter.

    Collects per-experiment to correctly separate prefill/decode rows
    from the stacked feature matrix (first n rows = prefill, last n = decode).
    """
    pf_pred_acc: list[np.ndarray] = []
    pf_obs_acc: list[np.ndarray] = []
    dc_pred_acc: list[np.ndarray] = []
    dc_obs_acc: list[np.ndarray] = []

    for exp in get_active():
        result = reconstruct_experiment(exp)
        arch = load_model_arch(f"model_configs/{exp.config_json_dir}/config.json")
        basis = compute_experiment_basis(result, arch, hw, exp.tensor_parallelism)

        # Filter labels to matching split
        filtered_labels = [
            lb for lb in result.labels
            if not lb.failed and request_split(lb.request_id) == split_filter
        ]

        X, y, request_ids = build_stacked_feature_matrix(
            result.steps, basis, filtered_labels,
        )
        if len(request_ids) == 0:
            continue

        n = len(request_ids)
        pred = X @ betas

        # First n rows = prefill, last n rows = decode
        pf_pred_acc.append(pred[:n])
        pf_obs_acc.append(y[:n])
        dc_pred_acc.append(pred[n:])
        dc_obs_acc.append(y[n:])

    if not pf_pred_acc:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty

    pf_pred = np.concatenate(pf_pred_acc)
    pf_obs = np.concatenate(pf_obs_acc)
    dc_pred = np.concatenate(dc_pred_acc)
    dc_obs = np.concatenate(dc_obs_acc)
    combined_pred = pf_pred + dc_pred
    combined_obs = pf_obs + dc_obs

    return pf_pred, pf_obs, dc_pred, dc_obs, combined_pred, combined_obs


# =============================================================================
# Public API
# =============================================================================

SPLITS = [Split.TRAIN, Split.VALIDATE, Split.TEST]
OUTPUT_DIR = Path("output/evaluate")


def evaluate(coeffs: FittedCoefficients, hw: HardwareSpec) -> EvaluationResult:
    """Evaluate all 5 measures across train/validate/test splits.

    Requires: coeffs is a valid FittedCoefficients (7 betas, all non-negative),
              hw is a valid HardwareSpec. Active experiments with trace data exist.
    Guarantees: returns EvaluationResult with all 5 measures, each containing
                "train", "validate", "test" keys. All metrics are non-negative.

    Note: Alpha measures (pre_queueing, post_decode) and GPU measures may have
    different n values per split because they use different data joins:
    - Alpha requires both API spans (ARRIVED/DEPARTED) and journey spans (QUEUED/FINISHED)
    - GPU requires successful RequestLabels with step-level reconstruction
    """
    betas = np.array(coeffs.betas)

    pre_queueing: dict[str, MeasureMetrics] = {}
    post_decode: dict[str, MeasureMetrics] = {}
    gpu_prefill: dict[str, MeasureMetrics] = {}
    gpu_decode: dict[str, MeasureMetrics] = {}
    gpu_combined: dict[str, MeasureMetrics] = {}

    for split in SPLITS:
        name = split.value

        # Alpha measures
        pre_q, post_d = _collect_alpha_eval_data(coeffs, split)
        pre_queueing[name] = pre_q
        post_decode[name] = post_d

        # GPU measures
        pf_pred, pf_obs, dc_pred, dc_obs, comb_pred, comb_obs = (
            _collect_gpu_eval_data(hw, betas, split)
        )
        gpu_prefill[name] = compute_metrics(pf_pred, pf_obs)
        gpu_decode[name] = compute_metrics(dc_pred, dc_obs)
        gpu_combined[name] = compute_metrics(comb_pred, comb_obs)

    return EvaluationResult(
        pre_queueing=pre_queueing,
        post_decode=post_decode,
        gpu_prefill=gpu_prefill,
        gpu_decode=gpu_decode,
        gpu_combined=gpu_combined,
    )


# =============================================================================
# Output
# =============================================================================

def _format_metric(value: float, width: int = 10) -> str:
    """Format a metric value for table display."""
    if abs(value) >= 1e6:
        return f"{value:>{width},.0f}"
    elif abs(value) >= 100:
        return f"{value:>{width},.1f}"
    else:
        return f"{value:>{width}.2f}"


def write_evaluation(result: EvaluationResult) -> None:
    """Write evaluation results to output/evaluate/ and print table.

    Requires: result is a valid EvaluationResult.
    Guarantees: writes metrics.json, metrics.csv, and prints formatted table to stdout.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validate", "test"]
    measures = [
        ("pre_queueing", result.pre_queueing),
        ("post_decode", result.post_decode),
        ("gpu_prefill", result.gpu_prefill),
        ("gpu_decode", result.gpu_decode),
        ("gpu_combined", result.gpu_combined),
    ]

    # JSON output
    output = {}
    for measure_name, measure_dict in measures:
        output[measure_name] = {
            split_name: asdict(metrics)
            for split_name, metrics in measure_dict.items()
        }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    # CSV output — one row per (measure, split) for easy visualization
    with open(OUTPUT_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["measure", "split", "mape_pct", "rmse_us", "mae_us", "n"])
        for measure_name, measure_dict in measures:
            for split_name in splits:
                m = measure_dict[split_name]
                writer.writerow([measure_name, split_name, f"{m.mape:.4f}", f"{m.rmse:.2f}", f"{m.mae:.2f}", m.n])

    # Display labels for the printed table
    display_labels = {
        "pre_queueing": "Pre-queue",
        "post_decode": "Post-decode",
        "gpu_prefill": "GPU prefill",
        "gpu_decode": "GPU decode",
        "gpu_combined": "GPU combined",
    }

    print("=" * 100)
    print("  Evaluation Results — 5 measures × 3 metrics × 3 splits")
    print("=" * 100)

    # Header
    header = f"{'':>14}"
    for split_name in splits:
        header += f"  {'MAPE%':>8} {'RMSE':>10} {'MAE':>10} {'n':>6}"
    print(f"\n{header}")

    subheader = f"{'':>14}"
    for _ in splits:
        subheader += f"  {'':>8} {'(µs)':>10} {'(µs)':>10} {'':>6}"
    print(subheader)
    print("-" * 100)

    for measure_name, measure_dict in measures:
        label = display_labels[measure_name]
        row = f"{label:>14}"
        for split_name in splits:
            m = measure_dict[split_name]
            row += f"  {m.mape:>8.2f} {_format_metric(m.rmse, 10)} {_format_metric(m.mae, 10)} {m.n:>6,}"
        print(row)

    print("-" * 100)
    print(f"\n  Output written to {OUTPUT_DIR}/")
    print("=" * 100)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    hw = load_hardware_spec("datasheets/h100-sxm.json")
    coeffs = fit_coefficients(hw)
    result = evaluate(coeffs, hw)
    write_evaluation(result)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
