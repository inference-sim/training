"""Tests for the evaluation module (evaluate.py).

Follows BDD discipline: tests verify observable behavior of public API.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import EvaluationResult, compute_metrics


# =============================================================================
# TestComputeMetrics — synthetic data with known MAPE/RMSE/MAE
# =============================================================================


class TestComputeMetrics:
    """Verify MAPE/RMSE/MAE computation from predicted vs observed arrays.

    Expected behavior:
        - Known prediction errors produce hand-computable metric values.
        - Perfect predictions yield all-zero metrics.
        - MAPE guard excludes observations with |observed| < 1.0 µs.
        - Empty inputs produce zero metrics with n=0.
    """

    def test_known_values(self):
        """Verify MAPE/RMSE/MAE against hand-computed values."""
        predicted = np.array([110.0, 210.0, 310.0])
        observed = np.array([100.0, 200.0, 300.0])
        # Errors: [10, 10, 10]
        # RMSE: sqrt(mean([100, 100, 100])) = 10
        # MAE: mean([10, 10, 10]) = 10
        # MAPE: mean([10/100, 10/200, 10/300]) * 100 = mean([0.1, 0.05, 0.0333]) * 100
        expected_mape = (10 / 100 + 10 / 200 + 10 / 300) / 3 * 100
        m = compute_metrics(predicted, observed)
        assert m.n == 3
        assert m.rmse == pytest.approx(10.0)
        assert m.mae == pytest.approx(10.0)
        assert m.mape == pytest.approx(expected_mape)

    def test_perfect_prediction(self):
        """All metrics are zero when predicted == observed."""
        values = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(values, values.copy())
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.mape == 0.0
        assert m.n == 3

    def test_mape_skips_near_zero_observed(self):
        """MAPE excludes observations where |observed| < 1.0 µs."""
        predicted = np.array([5.0, 100.0])
        observed = np.array([0.5, 100.0])  # first is near-zero
        m = compute_metrics(predicted, observed)
        # Only the second observation should be used for MAPE
        # |100 - 100| / 100 = 0
        assert m.mape == pytest.approx(0.0)
        # RMSE/MAE still use all observations
        assert m.n == 2

    def test_mape_all_near_zero(self):
        """MAPE is 0 when all observations are near-zero."""
        predicted = np.array([0.1, 0.2])
        observed = np.array([0.01, 0.02])
        m = compute_metrics(predicted, observed)
        assert m.mape == 0.0  # All below threshold, no MAPE computed

    def test_empty_input(self):
        """Empty arrays produce zero metrics with n=0."""
        m = compute_metrics(np.array([]), np.array([]))
        assert m.n == 0
        assert m.rmse == 0.0
        assert m.mae == 0.0
        assert m.mape == 0.0


# =============================================================================
# TestEvaluateEndToEnd — full pipeline on real data
# =============================================================================


class TestEvaluateEndToEnd:
    """End-to-end assessment on real trace data.

    Expected behavior:
        - All 5 measures x 3 splits are present with positive observation counts.
        - GPU prefill, decode, and combined share the same request count per split.
        - Combined MAE respects the triangle inequality (bounded by prefill + decode).
        - Train GPU RMSE is consistent with the fitting module's reported MSE.
    """

    @pytest.fixture(scope="class")
    def fitting_and_result(self):
        from basis_functions import load_hardware_spec
        from evaluate import evaluate as run_evaluate
        from fit_coefficients import fit_coefficients

        hw = load_hardware_spec("datasheets/h100-sxm.json")
        coeffs = fit_coefficients(hw)
        result = run_evaluate(coeffs, hw)
        return coeffs, result

    @pytest.fixture(scope="class")
    def result(self, fitting_and_result) -> EvaluationResult:
        return fitting_and_result[1]

    @pytest.fixture(scope="class")
    def fitted_coeffs(self, fitting_and_result):
        return fitting_and_result[0]

    def test_all_splits_present(self, result: EvaluationResult):
        """Every measure has train/validate/test keys."""
        expected_splits = {"train", "validate", "test"}
        for measure_name in ("pre_queueing", "post_decode", "gpu_prefill", "gpu_decode", "gpu_combined"):
            measure_dict = getattr(result, measure_name)
            assert set(measure_dict.keys()) == expected_splits, (
                f"{measure_name} missing splits: {expected_splits - set(measure_dict.keys())}"
            )

    def test_all_measures_have_positive_n(self, result: EvaluationResult):
        """Every split has > 0 observations for every measure."""
        for measure_name in ("pre_queueing", "post_decode", "gpu_prefill", "gpu_decode", "gpu_combined"):
            measure_dict = getattr(result, measure_name)
            for split_name, metrics in measure_dict.items():
                assert metrics.n > 0, (
                    f"{measure_name}/{split_name} has n=0"
                )

    def test_gpu_measures_have_same_n_per_split(self, result: EvaluationResult):
        """Prefill, decode, and combined all cover the same requests per split."""
        for split_name in ("train", "validate", "test"):
            pf_n = result.gpu_prefill[split_name].n
            dc_n = result.gpu_decode[split_name].n
            comb_n = result.gpu_combined[split_name].n
            assert pf_n == dc_n == comb_n, (
                f"{split_name}: n mismatch — prefill={pf_n}, decode={dc_n}, combined={comb_n}"
            )

    def test_gpu_combined_mae_bounded_by_prefill_plus_decode(self, result: EvaluationResult):
        """Combined MAE respects the triangle inequality.

        Since combined_pred = pf_pred + dc_pred and combined_obs = pf_obs + dc_obs,
        by triangle inequality: combined MAE <= prefill MAE + decode MAE.
        This verifies the stacked matrix split/recombine is correct.
        """
        for split_name in ("train", "validate", "test"):
            pf_mae = result.gpu_prefill[split_name].mae
            dc_mae = result.gpu_decode[split_name].mae
            comb_mae = result.gpu_combined[split_name].mae
            assert comb_mae <= pf_mae + dc_mae + 1.0, (
                f"{split_name}: combined MAE ({comb_mae:.1f}) > "
                f"prefill MAE ({pf_mae:.1f}) + decode MAE ({dc_mae:.1f})"
            )

    def test_train_gpu_combined_rmse_near_fit_diagnostics(
        self, result: EvaluationResult, fitted_coeffs,
    ):
        """Train GPU combined RMSE should be in same ballpark as fit RMSE.

        Not exact because fit MSE is on the stacked (2n) system while this
        RMSE is on the combined (n) per-request predictions. But they should be
        in the same order of magnitude.
        """
        train_combined = result.gpu_combined["train"]
        assert train_combined.rmse > 0
        fit_rmse = fitted_coeffs.train_mse ** 0.5
        ratio = train_combined.rmse / fit_rmse if fit_rmse > 0 else float("inf")
        assert 0.1 < ratio < 10, (
            f"Train GPU combined RMSE ({train_combined.rmse:.0f}) differs from fit RMSE "
            f"({fit_rmse:.0f}) by ratio {ratio:.1f}"
        )

    def test_mape_is_percentage(self, result: EvaluationResult):
        """All MAPE values are non-negative (percentage points)."""
        for measure_name in ("pre_queueing", "post_decode", "gpu_prefill", "gpu_decode", "gpu_combined"):
            measure_dict = getattr(result, measure_name)
            for split_name, metrics in measure_dict.items():
                assert metrics.mape >= 0, (
                    f"{measure_name}/{split_name} has negative MAPE: {metrics.mape}"
                )

    def test_rmse_and_mae_non_negative(self, result: EvaluationResult):
        """All RMSE and MAE values are non-negative."""
        for measure_name in ("pre_queueing", "post_decode", "gpu_prefill", "gpu_decode", "gpu_combined"):
            measure_dict = getattr(result, measure_name)
            for split_name, metrics in measure_dict.items():
                assert metrics.rmse >= 0, f"{measure_name}/{split_name} RMSE < 0"
                assert metrics.mae >= 0, f"{measure_name}/{split_name} MAE < 0"
