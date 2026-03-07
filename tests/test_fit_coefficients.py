"""Behavioral tests for coefficient fitting.

Each test class describes a fitting scenario and verifies the output
FittedCoefficients or intermediate results satisfy expected properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from basis_functions import StepBasisValues
from reconstruct_steps import ReconstructedStep, PrefillEntry, DecodeEntry, RequestLabel
from split import get_active


class TestEstimateAlpha0:
    """Phase 1: α₀ = mean(QUEUED.ts − ARRIVED.ts).

    Given synthetic ARRIVED and QUEUED timestamps with known differences,
    verify α₀ equals the mean of those differences in microseconds.
    """

    def test_alpha0_is_mean_of_differences(self):
        from fit_coefficients import estimate_alpha_0

        arrived_queued_pairs = [
            (100.000, 100.005),
            (200.000, 200.006),
            (300.000, 300.007),
            (400.000, 400.008),
        ]
        alpha_0 = estimate_alpha_0(arrived_queued_pairs)
        assert abs(alpha_0 - 6500.0) < 0.1

    def test_alpha0_is_positive(self):
        from fit_coefficients import estimate_alpha_0

        pairs = [(100.0, 100.003), (200.0, 200.005)]
        alpha_0 = estimate_alpha_0(pairs)
        assert alpha_0 > 0

    def test_alpha0_single_request(self):
        from fit_coefficients import estimate_alpha_0

        pairs = [(100.0, 100.010)]
        alpha_0 = estimate_alpha_0(pairs)
        assert abs(alpha_0 - 10000.0) < 0.1

    def test_alpha0_raises_on_empty_input(self):
        from fit_coefficients import estimate_alpha_0

        with pytest.raises(ValueError, match="No arrived/queued pairs"):
            estimate_alpha_0([])

    def test_alpha0_warns_on_non_positive(self):
        from fit_coefficients import estimate_alpha_0

        # queued < arrived -> negative difference
        pairs = [(100.010, 100.000), (200.010, 200.000)]
        with pytest.warns(UserWarning, match="non-positive"):
            alpha_0 = estimate_alpha_0(pairs)
        assert alpha_0 < 0


class TestFitAlpha12:
    """Phase 2: NNLS fit of α₁ + α₂·output_tokens."""

    def test_recovers_known_coefficients(self):
        from fit_coefficients import fit_alpha_12

        np.random.seed(42)
        output_tokens = np.array([50, 100, 150, 200, 250, 300])
        y = 500.0 + 5.0 * output_tokens + np.random.normal(0, 10, len(output_tokens))
        alpha_1, alpha_2 = fit_alpha_12(output_tokens, y)
        assert abs(alpha_1 - 500.0) < 100
        assert abs(alpha_2 - 5.0) < 2.0

    def test_coefficients_are_non_negative(self):
        from fit_coefficients import fit_alpha_12

        output_tokens = np.array([10, 20, 30, 40, 50])
        y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        alpha_1, alpha_2 = fit_alpha_12(output_tokens, y)
        assert alpha_1 >= 0
        assert alpha_2 >= 0

    def test_pure_intercept(self):
        from fit_coefficients import fit_alpha_12

        output_tokens = np.array([10, 50, 100, 200])
        y = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        alpha_1, alpha_2 = fit_alpha_12(output_tokens, y)
        assert alpha_1 > 500
        assert alpha_2 < 5


class TestStackedFeatureMatrix:
    """Stacked prefill+decode feature matrix for Phase 3 regression.

    Each non-failed request produces two rows: one for prefill steps,
    one for decode steps. Shared β₃-β₇, selective β₁/β₂.
    """

    @pytest.fixture()
    def data(self):
        steps = [
            ReconstructedStep(
                step_id=1,
                prefill_reqs=(PrefillEntry("req-A", tokens_this_step=512, prompt_tokens=512),),
                decode_reqs=(DecodeEntry("req-B", context_length=200),),
                total_prefill_tokens=512, total_decode_tokens=1, batch_size=2,
            ),
            ReconstructedStep(
                step_id=2,
                prefill_reqs=(),
                decode_reqs=(
                    DecodeEntry("req-A", context_length=513),
                    DecodeEntry("req-B", context_length=201),
                ),
                total_prefill_tokens=0, total_decode_tokens=2, batch_size=2,
            ),
        ]
        basis_values = [
            StepBasisValues(
                step_id=1, t_pf_compute=100.0, t_pf_kv=80.0,
                t_dc_compute=50.0, t_dc_kv=60.0,
                t_weight=200.0, t_tp=30.0, num_layers=32, batch_size=2,
            ),
            StepBasisValues(
                step_id=2, t_pf_compute=0.0, t_pf_kv=0.0,
                t_dc_compute=70.0, t_dc_kv=90.0,
                t_weight=200.0, t_tp=35.0, num_layers=32, batch_size=2,
            ),
        ]
        labels = [
            RequestLabel("req-A", prompt_tokens=512, output_tokens=50,
                         queueing_us=1000.0, ttft_us=2000.0,
                         processing_us=5000.0,
                         prefill_processing_us=500.0,
                         decode_processing_us=4500.0,
                         e2e_us=6000.0,
                         num_preemptions=0, failed=False, first_step=1, last_step=2),
            RequestLabel("req-B", prompt_tokens=200, output_tokens=80,
                         queueing_us=500.0, ttft_us=1500.0,
                         processing_us=4000.0,
                         prefill_processing_us=0.0,
                         decode_processing_us=4000.0,
                         e2e_us=5000.0,
                         num_preemptions=0, failed=False, first_step=1, last_step=2),
        ]
        return steps, basis_values, labels

    def test_stacked_shape_is_2n_by_7(self, data):
        from fit_coefficients import build_stacked_feature_matrix
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        assert X.shape == (2 * n, 7)
        assert y.shape == (2 * n,)

    def test_prefill_rows_target_prefill_processing_us(self, data):
        from fit_coefficients import build_stacked_feature_matrix
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_a = req_ids.index("req-A")
        assert abs(y[idx_a] - 500.0) < 0.01  # prefill target
        assert abs(y[n + idx_a] - 4500.0) < 0.01  # decode target

    def test_feature_sum_invariant(self, data):
        """X_pf[i] + X_dc[i] equals what the old single-row formulation produced."""
        from fit_coefficients import build_stacked_feature_matrix
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        for i in range(n):
            X_total = X[i] + X[n + i]  # prefill row + decode row
            assert all(X_total[j] >= 0 for j in range(7))

    def test_beta1_only_in_prefill_rows(self, data):
        """β₁ (prefill roofline) should only appear in rows from prefill steps."""
        from fit_coefficients import build_stacked_feature_matrix
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_b = req_ids.index("req-B")
        # req-B has no prefill steps → β₁ column is 0 in prefill row
        assert X[idx_b, 0] == 0.0
        # req-B decode row also has β₁ = 0 (decode entries don't accumulate β₁)
        assert X[n + idx_b, 0] == 0.0

    def test_beta2_only_in_decode_rows(self, data):
        """β₂ (decode roofline) should only appear in rows from decode steps."""
        from fit_coefficients import build_stacked_feature_matrix
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_a = req_ids.index("req-A")
        # req-A prefill row: β₂ = 0 (prefill entries don't accumulate β₂)
        assert X[idx_a, 1] == 0.0

    def test_req_a_prefill_features(self, data):
        """req-A has prefill at step 1, decode at step 2."""
        from fit_coefficients import build_stacked_feature_matrix
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_a = req_ids.index("req-A")
        pf_row = X[idx_a]
        dc_row = X[n + idx_a]
        # Prefill row: β₁=100 (max(100,80) from step 1), β₂=0
        assert abs(pf_row[0] - 100.0) < 0.01
        assert abs(pf_row[1] - 0.0) < 0.01
        # Decode row: β₁=0, β₂=90 (max(70,90) from step 2)
        assert abs(dc_row[0] - 0.0) < 0.01
        assert abs(dc_row[1] - 90.0) < 0.01

    def test_failed_requests_excluded(self, data):
        from fit_coefficients import build_stacked_feature_matrix
        steps, basis_values, labels = data
        failed_label = RequestLabel(
            "req-C", prompt_tokens=100, output_tokens=0,
            queueing_us=0.0, ttft_us=0.0, processing_us=0.0,
            prefill_processing_us=0.0, decode_processing_us=0.0,
            e2e_us=0.0,
            num_preemptions=0, failed=True, first_step=0, last_step=0,
        )
        X, y, req_ids = build_stacked_feature_matrix(steps, basis_values, list(labels) + [failed_label])
        assert "req-C" not in req_ids
        assert X.shape[0] == 4  # 2 requests * 2 rows

    def test_all_failed_returns_correctly_shaped_empty(self):
        from fit_coefficients import build_stacked_feature_matrix

        steps = [
            ReconstructedStep(
                step_id=1, prefill_reqs=(), decode_reqs=(),
                total_prefill_tokens=0, total_decode_tokens=0, batch_size=0,
            ),
        ]
        basis_values = [
            StepBasisValues(
                step_id=1, t_pf_compute=0, t_pf_kv=0,
                t_dc_compute=0, t_dc_kv=0,
                t_weight=0, t_tp=0, num_layers=32, batch_size=0,
            ),
        ]
        labels = [
            RequestLabel("req-X", prompt_tokens=100, output_tokens=0,
                         queueing_us=0, ttft_us=0, processing_us=0,
                         prefill_processing_us=0, decode_processing_us=0,
                         e2e_us=0,
                         num_preemptions=0, failed=True, first_step=0, last_step=0),
        ]
        X, y, req_ids = build_stacked_feature_matrix(steps, basis_values, labels)
        assert X.shape == (0, 7)
        assert y.shape == (0,)
        assert req_ids == []


class TestFitBetas:
    """Phase 3: regularized NNLS fit of β₁–β₇."""

    def test_recovers_known_betas_no_regularization(self):
        from fit_coefficients import fit_betas

        np.random.seed(42)
        n = 200
        true_betas = np.array([2.0, 8.0, 1.5, 1.0, 20.0, 50.0, 500.0])
        X = np.random.rand(n, 7) * 1000
        y = X @ true_betas + np.random.normal(0, 100, n)
        y = np.maximum(y, 0)

        betas, _ = fit_betas(X, y, lambda_val=0.0)
        for i in range(7):
            assert abs(betas[i] - true_betas[i]) < true_betas[i] * 0.5, (
                f"β{i+1}: expected ~{true_betas[i]}, got {betas[i]}"
            )

    def test_all_betas_non_negative(self):
        from fit_coefficients import fit_betas

        np.random.seed(123)
        X = np.random.rand(100, 7) * 500
        y = np.random.rand(100) * 10000
        betas, _ = fit_betas(X, y, lambda_val=1.0)
        for i, b in enumerate(betas):
            assert b >= 0, f"β{i+1} = {b} is negative"

    def test_regularization_pulls_toward_one(self):
        from fit_coefficients import fit_betas

        np.random.seed(42)
        n = 500
        # Use orthogonal features to avoid coupling effects between betas.
        X = np.zeros((n, 7))
        for col in range(7):
            X[:, col] = np.random.rand(n) * 1000
        true_betas = np.array([10.0, 30.0, 15.0, 20.0, 50.0, 100.0, 500.0])
        y = X @ true_betas + np.random.normal(0, 500, n)
        y = np.maximum(y, 0)

        betas_unreg, _ = fit_betas(X, y, lambda_val=0.0)
        betas_reg, _ = fit_betas(X, y, lambda_val=1e6)

        # At least 3 of the first 4 betas should be pulled closer to 1.0
        closer_count = 0
        for i in range(4):
            dist_unreg = abs(betas_unreg[i] - 1.0)
            dist_reg = abs(betas_reg[i] - 1.0)
            if dist_reg < dist_unreg:
                closer_count += 1
        assert closer_count >= 3, (
            f"Only {closer_count}/4 betas pulled closer to 1.0 by regularization"
        )

    def test_beta567_unaffected_by_regularization(self):
        from fit_coefficients import fit_betas

        np.random.seed(42)
        n = 500
        true_betas = np.array([2.0, 8.0, 1.5, 1.0, 20.0, 50.0, 500.0])
        X = np.random.rand(n, 7) * 1000
        y = X @ true_betas + np.random.normal(0, 50, n)
        y = np.maximum(y, 0)

        betas_unreg, _ = fit_betas(X, y, lambda_val=0.0)
        betas_reg, _ = fit_betas(X, y, lambda_val=1.0)

        for i in [4, 5, 6]:
            if betas_unreg[i] > 1:
                ratio = betas_reg[i] / betas_unreg[i]
                assert 0.5 < ratio < 2.0, (
                    f"β{i+1}: unreg={betas_unreg[i]:.2f}, reg={betas_reg[i]:.2f}, "
                    f"ratio={ratio:.2f}"
                )

    def test_rejects_wrong_feature_count(self):
        from fit_coefficients import fit_betas

        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        with pytest.raises(ValueError, match="Expected 7 features"):
            fit_betas(X, y, lambda_val=0.0)


class TestFittedCoefficientsValidation:
    """FittedCoefficients enforces invariants at construction time."""

    def test_rejects_wrong_betas_length(self):
        from fit_coefficients import FittedCoefficients

        with pytest.raises(ValueError, match="exactly 7 elements"):
            FittedCoefficients(
                alpha_0=1.0, alpha_1=1.0, alpha_2=1.0,
                betas=(1.0, 2.0), lambda_val=0.0,
                train_mse=0.0, val_mse=0.0,
            )

    def test_rejects_negative_alpha(self):
        from fit_coefficients import FittedCoefficients

        with pytest.raises(ValueError, match="alpha_0 must be >= 0"):
            FittedCoefficients(
                alpha_0=-1.0, alpha_1=0.0, alpha_2=0.0,
                betas=(1.0,) * 7, lambda_val=0.0,
                train_mse=0.0, val_mse=0.0,
            )

    def test_rejects_negative_beta(self):
        from fit_coefficients import FittedCoefficients

        with pytest.raises(ValueError, match="All betas must be >= 0"):
            FittedCoefficients(
                alpha_0=0.0, alpha_1=0.0, alpha_2=0.0,
                betas=(1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0),
                lambda_val=0.0, train_mse=0.0, val_mse=0.0,
            )


class TestCollectAlphaData:
    """Collect API/journey timestamp pairs across experiments for α fitting."""

    def test_collect_alpha0_pairs_returns_valid_data(self):
        from fit_coefficients import collect_alpha_data

        pairs_0, triples_12 = collect_alpha_data()
        assert len(pairs_0) > 0
        for arrived, queued in pairs_0:
            assert queued > arrived > 0

    def test_collect_alpha12_triples_returns_valid_data(self):
        from fit_coefficients import collect_alpha_data

        pairs_0, triples_12 = collect_alpha_data()
        assert len(triples_12) > 0
        for departed, finished, n_tokens in triples_12:
            assert departed > finished > 0
            assert n_tokens > 0


class TestTuneLambda:
    """Grid search λ tuning on validation set."""

    def test_selects_lambda_with_lowest_val_mse(self):
        from fit_coefficients import tune_lambda

        np.random.seed(42)
        n_train, n_val = 200, 50
        true_betas = np.array([2.0, 8.0, 1.5, 1.0, 20.0, 50.0, 500.0])
        X_train = np.random.rand(n_train, 7) * 1000
        y_train = X_train @ true_betas + np.random.normal(0, 100, n_train)
        y_train = np.maximum(y_train, 0)
        X_val = np.random.rand(n_val, 7) * 1000
        y_val = X_val @ true_betas + np.random.normal(0, 100, n_val)
        y_val = np.maximum(y_val, 0)

        best_lambda, best_betas, train_mse, val_mse = tune_lambda(
            X_train, y_train, X_val, y_val
        )
        assert best_lambda >= 0
        assert len(best_betas) == 7
        assert all(b >= 0 for b in best_betas)
        assert train_mse >= 0
        assert val_mse >= 0


class TestFitCoefficientsEndToEnd:
    """End-to-end fitting on real data."""

    @pytest.fixture(scope="class")
    def result(self):
        from fit_coefficients import fit_coefficients
        from basis_functions import load_hardware_spec

        hw = load_hardware_spec("datasheets/h100-sxm.json")
        return fit_coefficients(hw)

    def test_alpha_0_in_expected_range(self, result):
        assert result.alpha_0 > 0
        assert 1000 < result.alpha_0 < 50000

    def test_alpha_12_non_negative(self, result):
        assert result.alpha_1 >= 0
        assert result.alpha_2 >= 0

    def test_alpha_2_in_expected_range(self, result):
        assert result.alpha_2 < 100

    def test_all_betas_non_negative(self, result):
        assert len(result.betas) == 7
        for i, b in enumerate(result.betas):
            assert b >= 0, f"β{i+1} = {b} is negative"

    def test_beta1_prefill_roofline_positive(self, result):
        """Primary success criterion: β₁ should now be > 0 with stacked formulation."""
        assert result.betas[0] > 0, (
            f"β₁ (prefill roofline) = {result.betas[0]}, expected > 0. "
            f"The stacked formulation should give prefill proper signal."
        )

    def test_lambda_non_negative(self, result):
        assert result.lambda_val >= 0

    def test_mse_values_non_negative(self, result):
        assert result.train_mse >= 0
        assert result.val_mse >= 0

    def test_returns_frozen_dataclass(self, result):
        from fit_coefficients import FittedCoefficients
        assert isinstance(result, FittedCoefficients)
        with pytest.raises(AttributeError):
            result.alpha_0 = 999  # type: ignore[misc]
