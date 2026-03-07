"""Behavioral tests for coefficient fitting.

Each test class describes a fitting scenario and verifies the output
FittedCoefficients or intermediate results satisfy expected properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from basis_functions import StepBasisValues
from reconstruct_steps import ReconstructedStep, PrefillEntry, DecodeEntry, RequestLabel
from split import get_train


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


class TestBuildFeatureMatrix:
    """Build per-request feature matrix from steps and basis values."""

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
                         processing_us=5000.0, e2e_us=6000.0,
                         num_preemptions=0, failed=False, first_step=1, last_step=2),
            RequestLabel("req-B", prompt_tokens=200, output_tokens=80,
                         queueing_us=500.0, ttft_us=1500.0,
                         processing_us=4000.0, e2e_us=5000.0,
                         num_preemptions=0, failed=False, first_step=1, last_step=2),
        ]
        return steps, basis_values, labels

    def test_feature_matrix_shape(self, data):
        from fit_coefficients import build_feature_matrix
        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        assert X.shape == (2, 7)
        assert y.shape == (2,)
        assert len(req_ids) == 2

    def test_req_a_features(self, data):
        from fit_coefficients import build_feature_matrix
        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        idx = req_ids.index("req-A")
        row = X[idx]
        assert abs(row[0] - 100.0) < 0.01  # max(100,80) from step 1
        assert abs(row[1] - 90.0) < 0.01   # max(70,90) from step 2
        assert abs(row[2] - 400.0) < 0.01  # 200+200
        assert abs(row[3] - 65.0) < 0.01   # 30+35
        assert abs(row[4] - 64.0) < 0.01   # 32+32
        assert abs(row[5] - 4.0) < 0.01    # 2+2
        assert abs(row[6] - 2.0) < 0.01    # 2 active steps

    def test_req_b_features(self, data):
        from fit_coefficients import build_feature_matrix
        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        idx = req_ids.index("req-B")
        row = X[idx]
        assert abs(row[0] - 0.0) < 0.01    # no prefill for req-B
        assert abs(row[1] - 150.0) < 0.01  # max(50,60)+max(70,90) = 60+90

    def test_target_is_processing_us(self, data):
        from fit_coefficients import build_feature_matrix
        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        idx_a = req_ids.index("req-A")
        idx_b = req_ids.index("req-B")
        assert abs(y[idx_a] - 5000.0) < 0.01
        assert abs(y[idx_b] - 4000.0) < 0.01

    def test_failed_requests_excluded(self, data):
        from fit_coefficients import build_feature_matrix
        steps, basis_values, labels = data
        failed_label = RequestLabel(
            "req-C", prompt_tokens=100, output_tokens=0,
            queueing_us=0.0, ttft_us=0.0, processing_us=0.0, e2e_us=0.0,
            num_preemptions=0, failed=True, first_step=0, last_step=0,
        )
        X, y, req_ids = build_feature_matrix(steps, basis_values, list(labels) + [failed_label])
        assert "req-C" not in req_ids
        assert X.shape[0] == 2


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
        # With correlated features, regularizing beta_i can push beta_j
        # further from its target.
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
