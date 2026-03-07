# fit_coefficients.py Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three-phase NNLS fitting of 10 crossmodel latency model parameters (α₀–α₂, β₁–β₇) from vLLM journey trace data.

**Architecture:** Extend trace_parser.py with API event extraction, then build fit_coefficients.py with three sequential fitting phases. Each phase is a pure function operating on numpy arrays. The public API composes all phases and writes diagnostics to output/fit/.

**Tech Stack:** Python 3.10+, scipy.optimize.nnls, numpy, existing modules (split.py, trace_parser.py, reconstruct_steps.py, basis_functions.py)

---

### Task 1: Add scipy dependency

**Files:**
- Modify: `pyproject.toml:6-9`

**Step 1: Add scipy and numpy to dependencies**

In `pyproject.toml`, change:
```toml
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
]
```
to:
```toml
dependencies = [
    "numpy>=1.24",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "scipy>=1.10",
]
```

**Step 2: Install**

Run: `pip install -e .`
Expected: Successfully installed with scipy and numpy

**Step 3: Verify import**

Run: `python3 -c "from scipy.optimize import nnls; print('ok')"`
Expected: `ok`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add scipy and numpy dependencies for coefficient fitting"
```

---

### Task 2: parse_api_events() in trace_parser.py

**Files:**
- Modify: `trace_parser.py` (add function after `parse_journey_events`)
- Test: `tests/test_trace_parser_api.py` (create)

**Context for implementer:**
- `llm_request` spans live under scope `vllm.api` (not `vllm.scheduler`)
- Request IDs on these spans have NO `-0-xxx` suffix (e.g. `cmpl-9dcf4ea42cd9ace2`)
- Journey event IDs DO have the suffix (e.g. `cmpl-9dcf4ea42cd9ace2-0-aa4f0692`)
- API event timestamps use attribute key `event.ts.monotonic` (not `ts.monotonic`)
- Each span has exactly 2 events: `api.ARRIVED` and `api.DEPARTED`
- Use existing `attr_map()` helper to parse attributes

**Step 1: Write the failing test**

Create `tests/test_trace_parser_api.py`:

```python
"""Behavioral tests for API event parsing from llm_request spans.

Verifies that parse_api_events correctly extracts api.ARRIVED and
api.DEPARTED timestamps from the vllm.api scope in traces.json.
"""

from __future__ import annotations

import pytest

from trace_parser import parse_api_events, parse_journey_events, traces_path_for
from split import get_train


class TestParseApiEventsOnRealData:
    """Parse a real experiment and verify API events match journey events.

    The llm_request spans (vllm.api scope) should cover at least as many
    requests as the llm_core spans (vllm.scheduler scope), since API
    events are emitted even for requests that fail before reaching the
    scheduler.
    """

    @pytest.fixture()
    def exp(self):
        return get_train()[0]  # llama-2-7b general

    @pytest.fixture()
    def api_events(self, exp):
        return parse_api_events(traces_path_for(exp))

    @pytest.fixture()
    def journey_events(self, exp):
        return parse_journey_events(traces_path_for(exp))

    def test_returns_dict_of_request_ids(self, api_events):
        assert isinstance(api_events, dict)
        assert len(api_events) > 0

    def test_each_entry_has_arrived_and_departed(self, api_events):
        for req_id, ts in api_events.items():
            assert "arrived_ts" in ts, f"{req_id} missing arrived_ts"
            assert "departed_ts" in ts, f"{req_id} missing departed_ts"

    def test_departed_after_arrived(self, api_events):
        for req_id, ts in api_events.items():
            assert ts["departed_ts"] > ts["arrived_ts"], (
                f"{req_id}: departed {ts['departed_ts']} <= arrived {ts['arrived_ts']}"
            )

    def test_timestamps_are_positive(self, api_events):
        for req_id, ts in api_events.items():
            assert ts["arrived_ts"] > 0, f"{req_id}: arrived_ts <= 0"
            assert ts["departed_ts"] > 0, f"{req_id}: departed_ts <= 0"

    def test_api_covers_all_journey_requests(self, api_events, journey_events):
        """Every journey request ID (with suffix stripped) should have an API entry."""
        for journey_id in journey_events:
            base_id = journey_id.rsplit("-0-", 1)[0] if "-0-" in journey_id else journey_id
            assert base_id in api_events, (
                f"Journey request {journey_id} (base={base_id}) has no API event"
            )

    def test_api_request_ids_have_no_suffix(self, api_events):
        """API request IDs should be base IDs without the -0-xxx suffix."""
        for req_id in api_events:
            assert "-0-" not in req_id or req_id.count("-") <= 4, (
                f"API request ID {req_id} looks like it has a journey suffix"
            )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trace_parser_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_api_events'`

**Step 3: Implement parse_api_events**

Add to `trace_parser.py` after `parse_journey_events()`:

```python
def parse_api_events(traces_path: Path | str) -> dict[str, dict]:
    """Parse traces.json and return API timestamps grouped by request ID.

    Reads ``llm_request`` spans from the ``vllm.api`` scope and collects
    ``api.ARRIVED`` and ``api.DEPARTED`` event timestamps keyed by
    ``gen_ai.request.id``.

    Requires: traces_path points to a valid traces.json file.
    Guarantees: For every returned entry, departed_ts > arrived_ts > 0.

    Note: API span request IDs do NOT have the sequence suffix (-0-xxx)
    that llm_core spans have.  The timestamp attribute key is
    ``event.ts.monotonic`` (not ``ts.monotonic`` like journey events).

    Returns:
        ``{base_request_id: {"arrived_ts": float, "departed_ts": float}}``
    """
    requests: dict[str, dict[str, float]] = {}

    with open(traces_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.warn(f"{traces_path}:{line_num}: skipping malformed JSON line: {e}")
                continue
            for rs in batch.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    if ss.get("scope", {}).get("name") != "vllm.api":
                        continue
                    for span in ss.get("spans", []):
                        if span["name"] != "llm_request":
                            continue
                        request_id = None
                        for a in span.get("attributes", []):
                            if a["key"] == "gen_ai.request.id":
                                request_id = a["value"].get("stringValue")
                                break
                        if request_id is None:
                            continue

                        ts_data: dict[str, float] = {}
                        for ev in span.get("events", []):
                            ev_attrs = attr_map(ev.get("attributes", []))
                            ts_val = ev_attrs.get("event.ts.monotonic", 0.0)
                            if ev["name"] == "api.ARRIVED":
                                ts_data["arrived_ts"] = ts_val
                            elif ev["name"] == "api.DEPARTED":
                                ts_data["departed_ts"] = ts_val

                        if "arrived_ts" in ts_data and "departed_ts" in ts_data:
                            if ts_data["departed_ts"] > ts_data["arrived_ts"] > 0:
                                requests[request_id] = ts_data
                            else:
                                warnings.warn(
                                    f"Request {request_id}: invalid API timestamps "
                                    f"arrived={ts_data['arrived_ts']}, "
                                    f"departed={ts_data['departed_ts']}"
                                )

    return requests
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trace_parser_api.py -v`
Expected: All 6 tests PASS

Run: `python3 -m pytest -q`
Expected: All tests pass (56 existing + 6 new = 62)

**Step 5: Commit**

```bash
git add trace_parser.py tests/test_trace_parser_api.py
git commit -m "feat: add parse_api_events() for API timestamp extraction"
```

---

### Task 3: Phase 1 — estimate α₀

**Files:**
- Create: `fit_coefficients.py`
- Test: `tests/test_fit_coefficients.py` (create)

**Context for implementer:**
- α₀ = mean(QUEUED.ts − ARRIVED.ts) across training requests, in µs
- Need to join API events (base ID) with journey events (suffixed ID) by stripping the `-0-xxx` suffix
- QUEUED.ts comes from `parse_journey_events` → attrs `ts.monotonic` on `journey.QUEUED` events
- ARRIVED.ts comes from `parse_api_events` → `arrived_ts`
- Use `attr_map()` to parse journey event attributes for the QUEUED timestamp

**Step 1: Write the failing test**

Create `tests/test_fit_coefficients.py`:

```python
"""Behavioral tests for coefficient fitting.

Each test class describes a fitting scenario and verifies the output
FittedCoefficients or intermediate results satisfy expected properties.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestEstimateAlpha0:
    """Phase 1: α₀ = mean(QUEUED.ts − ARRIVED.ts).

    Given synthetic ARRIVED and QUEUED timestamps with known differences,
    verify α₀ equals the mean of those differences in microseconds.
    """

    def test_alpha0_is_mean_of_differences(self):
        from fit_coefficients import estimate_alpha_0

        # Synthetic data: 4 requests with known gaps (in seconds)
        # Gaps: 0.005, 0.006, 0.007, 0.008 → mean = 0.0065s = 6500 µs
        arrived_queued_pairs = [
            (100.000, 100.005),
            (200.000, 200.006),
            (300.000, 300.007),
            (400.000, 400.008),
        ]
        alpha_0 = estimate_alpha_0(arrived_queued_pairs)
        assert abs(alpha_0 - 6500.0) < 0.1  # µs

    def test_alpha0_is_positive(self):
        from fit_coefficients import estimate_alpha_0

        pairs = [(100.0, 100.003), (200.0, 200.005)]
        alpha_0 = estimate_alpha_0(pairs)
        assert alpha_0 > 0

    def test_alpha0_single_request(self):
        from fit_coefficients import estimate_alpha_0

        pairs = [(100.0, 100.010)]  # 10ms gap
        alpha_0 = estimate_alpha_0(pairs)
        assert abs(alpha_0 - 10000.0) < 0.1  # 10ms = 10000 µs
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestEstimateAlpha0 -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fit_coefficients'`

**Step 3: Implement estimate_alpha_0**

Create `fit_coefficients.py`:

```python
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
    alpha_0: float          # API processing overhead (µs)
    alpha_1: float          # Post-decode fixed overhead (µs)
    alpha_2: float          # Post-decode per-token cost (µs/token)
    betas: tuple[float, ...]  # (β₁, β₂, β₃, β₄, β₅, β₆, β₇)
    lambda_val: float       # Regularization strength used
    train_mse: float        # Training set MSE (µs²)
    val_mse: float          # Validation set MSE (µs²)


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
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestEstimateAlpha0 -v`
Expected: All 3 tests PASS

Run: `python3 -m pytest -q`
Expected: All tests pass (62 existing + 3 new = 65)

**Step 5: Commit**

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: Phase 1 — estimate α₀ API processing overhead"
```

---

### Task 4: Phase 2 — fit α₁, α₂

**Files:**
- Modify: `fit_coefficients.py` (add function)
- Modify: `tests/test_fit_coefficients.py` (add test class)

**Context for implementer:**
- NNLS regression: `post_decode(r) = α₁ + α₂ · output_tokens(r)`
- Signal y = DEPARTED.ts − FINISHED.ts (µs)
- Feature matrix X = [[1, output_tokens_1], [1, output_tokens_2], ...]
- scipy.optimize.nnls(X, y) returns (coefficients, residual_norm)
- output_tokens comes from journey FINISHED event's decode.done_tokens

**Step 1: Write the failing test**

Add to `tests/test_fit_coefficients.py`:

```python
class TestFitAlpha12:
    """Phase 2: NNLS fit of α₁ + α₂·output_tokens.

    Given synthetic post-decode gaps with a known linear relationship
    to output_tokens, verify NNLS recovers α₁ and α₂.
    """

    def test_recovers_known_coefficients(self):
        from fit_coefficients import fit_alpha_12

        # y = 500 + 5 * output_tokens + noise
        np.random.seed(42)
        output_tokens = np.array([50, 100, 150, 200, 250, 300])
        y = 500.0 + 5.0 * output_tokens + np.random.normal(0, 10, len(output_tokens))
        alpha_1, alpha_2 = fit_alpha_12(output_tokens, y)
        assert abs(alpha_1 - 500.0) < 100  # within 100 µs
        assert abs(alpha_2 - 5.0) < 2.0    # within 2 µs/token

    def test_coefficients_are_non_negative(self):
        from fit_coefficients import fit_alpha_12

        output_tokens = np.array([10, 20, 30, 40, 50])
        y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        alpha_1, alpha_2 = fit_alpha_12(output_tokens, y)
        assert alpha_1 >= 0
        assert alpha_2 >= 0

    def test_pure_intercept(self):
        from fit_coefficients import fit_alpha_12

        # y ≈ 1000 (constant, no per-token cost)
        output_tokens = np.array([10, 50, 100, 200])
        y = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        alpha_1, alpha_2 = fit_alpha_12(output_tokens, y)
        assert alpha_1 > 500  # should capture most of the signal
        assert alpha_2 < 5    # per-token should be near zero
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestFitAlpha12 -v`
Expected: FAIL with `ImportError: cannot import name 'fit_alpha_12'`

**Step 3: Implement fit_alpha_12**

Add to `fit_coefficients.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestFitAlpha12 -v`
Expected: All 3 tests PASS

Run: `python3 -m pytest -q`
Expected: All tests pass (65 + 3 = 68)

**Step 5: Commit**

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: Phase 2 — fit α₁, α₂ post-decode overhead via NNLS"
```

---

### Task 5: Per-request feature matrix builder

**Files:**
- Modify: `fit_coefficients.py` (add function)
- Modify: `tests/test_fit_coefficients.py` (add test class)

**Context for implementer:**
- Walk through ReconstructedStep list. For each step, for each request_id in prefill_reqs and decode_reqs, accumulate that step's basis values into that request's feature vector.
- 7 features per request: Σ max(t_pf_compute, t_pf_kv), Σ max(t_dc_compute, t_dc_kv), Σ t_weight, Σ t_tp, Σ L (num_layers), Σ batch_size, num_active_steps
- Only include non-failed requests (matched by request_id from labels)
- Preempted requests are automatically handled — they don't appear in steps during the gap

**Step 1: Write the failing test**

Add to `tests/test_fit_coefficients.py`:

```python
from basis_functions import StepBasisValues
from reconstruct_steps import ReconstructedStep, PrefillEntry, DecodeEntry, RequestLabel


class TestBuildFeatureMatrix:
    """Build per-request feature matrix from steps and basis values.

    Given 2 steps with known basis values and request compositions,
    verify the feature matrix has correct sums per request.
    """

    @pytest.fixture()
    def data(self):
        """Two steps: step 1 has req-A prefill + req-B decode,
        step 2 has req-A decode + req-B decode."""
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
        assert X.shape == (2, 7)  # 2 requests, 7 features
        assert y.shape == (2,)
        assert len(req_ids) == 2

    def test_req_a_features(self, data):
        from fit_coefficients import build_feature_matrix

        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        idx = req_ids.index("req-A")
        row = X[idx]
        # req-A: step 1 (prefill) + step 2 (decode)
        # β₁: Σ max(t_pf_compute, t_pf_kv) = max(100,80) + max(0,0) = 100
        assert abs(row[0] - 100.0) < 0.01
        # β₂: Σ max(t_dc_compute, t_dc_kv) = 0 + max(70,90) = 90
        assert abs(row[1] - 90.0) < 0.01
        # β₃: Σ t_weight = 200 + 200 = 400
        assert abs(row[2] - 400.0) < 0.01
        # β₄: Σ t_tp = 30 + 35 = 65
        assert abs(row[3] - 65.0) < 0.01
        # β₅: Σ L = 32 + 32 = 64
        assert abs(row[4] - 64.0) < 0.01
        # β₆: Σ batch_size = 2 + 2 = 4
        assert abs(row[5] - 4.0) < 0.01
        # β₇: num_active_steps = 2
        assert abs(row[6] - 2.0) < 0.01

    def test_req_b_features(self, data):
        from fit_coefficients import build_feature_matrix

        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        idx = req_ids.index("req-B")
        row = X[idx]
        # req-B: step 1 (decode) + step 2 (decode)
        # β₁: Σ max(t_pf_compute, t_pf_kv) = 0 + 0 = 0 (no prefill for req-B)
        assert abs(row[0] - 0.0) < 0.01
        # β₂: Σ max(t_dc_compute, t_dc_kv) = max(50,60) + max(70,90) = 60 + 90 = 150
        assert abs(row[1] - 150.0) < 0.01

    def test_target_is_processing_us(self, data):
        from fit_coefficients import build_feature_matrix

        steps, basis_values, labels = data
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels)
        # y should be processing_us from labels
        idx_a = req_ids.index("req-A")
        idx_b = req_ids.index("req-B")
        assert abs(y[idx_a] - 5000.0) < 0.01
        assert abs(y[idx_b] - 4000.0) < 0.01

    def test_failed_requests_excluded(self, data):
        from fit_coefficients import build_feature_matrix

        steps, basis_values, labels = data
        # Add a failed request
        failed_label = RequestLabel(
            "req-C", prompt_tokens=100, output_tokens=0,
            queueing_us=0.0, ttft_us=0.0, processing_us=0.0, e2e_us=0.0,
            num_preemptions=0, failed=True, first_step=0, last_step=0,
        )
        X, y, req_ids = build_feature_matrix(steps, basis_values, labels + [failed_label])
        assert "req-C" not in req_ids
        assert X.shape[0] == 2  # still only 2 non-failed requests
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestBuildFeatureMatrix -v`
Expected: FAIL with `ImportError: cannot import name 'build_feature_matrix'`

**Step 3: Implement build_feature_matrix**

Add to `fit_coefficients.py`:

```python
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
    # Build label lookup for non-failed requests
    label_map = {lb.request_id: lb for lb in labels if not lb.failed}

    # Accumulate features per request
    features: dict[str, list[float]] = {rid: [0.0] * 7 for rid in label_map}
    step_counts: dict[str, int] = {rid: 0 for rid in label_map}

    for step, bv in zip(steps, basis_values):
        # Prefill requests in this step
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

        # Decode requests in this step
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

    # Build arrays, only include requests that appeared in at least one step
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
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestBuildFeatureMatrix -v`
Expected: All 5 tests PASS

Run: `python3 -m pytest -q`
Expected: All tests pass (68 + 5 = 73)

**Step 5: Commit**

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: build per-request feature matrix for Phase 3"
```

---

### Task 6: Phase 3 — fit β₁–β₇

**Files:**
- Modify: `fit_coefficients.py` (add function)
- Modify: `tests/test_fit_coefficients.py` (add test classes)

**Context for implementer:**
- Regularized NNLS: minimize ||Xβ - y||² + λ·Σᵢ₌₁⁴(βᵢ - 1)² subject to β ≥ 0
- Implemented via augmented matrix: append √λ·I₄ₓ₇ rows to X, √λ·1₄ to y
- I₄ₓ₇ has identity in first 4 cols, zeros in last 3 (only regularize β₁-β₄)
- λ tuned by grid search on validation set (lowest validation MSE wins)

**Step 1: Write the failing tests**

Add to `tests/test_fit_coefficients.py`:

```python
class TestFitBetas:
    """Phase 3: regularized NNLS fit of β₁–β₇.

    Tests verify NNLS recovery of known coefficients and regularization
    behavior.
    """

    def test_recovers_known_betas_no_regularization(self):
        from fit_coefficients import fit_betas

        np.random.seed(42)
        n = 200
        true_betas = np.array([2.0, 8.0, 1.5, 1.0, 20.0, 50.0, 500.0])
        X = np.random.rand(n, 7) * 1000
        y = X @ true_betas + np.random.normal(0, 100, n)
        y = np.maximum(y, 0)  # ensure non-negative targets

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
        n = 100
        # True betas far from 1.0
        true_betas = np.array([5.0, 20.0, 4.0, 3.0, 50.0, 100.0, 1000.0])
        X = np.random.rand(n, 7) * 500
        y = X @ true_betas

        betas_unreg, _ = fit_betas(X, y, lambda_val=0.0)
        betas_reg, _ = fit_betas(X, y, lambda_val=1000.0)

        # With heavy regularization, β₁-β₄ should be pulled toward 1.0
        for i in range(4):
            dist_unreg = abs(betas_unreg[i] - 1.0)
            dist_reg = abs(betas_reg[i] - 1.0)
            assert dist_reg < dist_unreg, (
                f"β{i+1}: regularized ({betas_reg[i]:.2f}) not closer to 1.0 "
                f"than unregularized ({betas_unreg[i]:.2f})"
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

        # β₅, β₆, β₇ should be similar regardless of λ
        for i in [4, 5, 6]:
            if betas_unreg[i] > 1:  # only check if meaningful
                ratio = betas_reg[i] / betas_unreg[i]
                assert 0.5 < ratio < 2.0, (
                    f"β{i+1}: unreg={betas_unreg[i]:.2f}, reg={betas_reg[i]:.2f}, "
                    f"ratio={ratio:.2f} — regularization shouldn't affect β₅-β₇"
                )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestFitBetas -v`
Expected: FAIL with `ImportError: cannot import name 'fit_betas'`

**Step 3: Implement fit_betas**

Add to `fit_coefficients.py`:

```python
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
        # Regularization rows: identity for β₁-β₄, zeros for β₅-β₇
        reg_X = np.zeros((4, 7))
        reg_X[:4, :4] = sqrt_lam * np.eye(4)
        reg_y = sqrt_lam * np.ones(4)  # prior of 1.0

        X_aug = np.vstack([X, reg_X])
        y_aug = np.concatenate([y, reg_y])
    else:
        X_aug = X
        y_aug = y

    betas, residual = nnls(X_aug, y_aug)
    return betas, float(residual)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestFitBetas -v`
Expected: All 4 tests PASS

Run: `python3 -m pytest -q`
Expected: All tests pass (73 + 4 = 77)

**Step 5: Commit**

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: Phase 3 — fit β₁–β₇ via regularized NNLS"
```

---

### Task 7: Data collection helpers

**Files:**
- Modify: `fit_coefficients.py` (add helper functions)
- Modify: `tests/test_fit_coefficients.py` (add test)

**Context for implementer:**
- Need helpers to collect (arrived_ts, queued_ts) pairs and (departed_ts, finished_ts, output_tokens) triples across experiments
- Journey IDs have `-0-xxx` suffix that needs stripping to match API IDs
- QUEUED.ts and FINISHED.ts come from journey events parsed with attr_map
- These helpers bridge the gap between per-experiment parsing and fitting functions

**Step 1: Write the failing test**

Add to `tests/test_fit_coefficients.py`:

```python
class TestCollectAlphaData:
    """Collect API/journey timestamp pairs across experiments for α fitting.

    Uses a single real experiment to verify data collection produces
    valid paired timestamps.
    """

    @pytest.fixture()
    def exp(self):
        return get_train()[0]  # llama-2-7b general

    def test_collect_alpha0_pairs_returns_valid_data(self, exp):
        from fit_coefficients import collect_alpha_data

        pairs_0, triples_12 = collect_alpha_data([exp])
        assert len(pairs_0) > 0
        for arrived, queued in pairs_0:
            assert queued > arrived > 0

    def test_collect_alpha12_triples_returns_valid_data(self, exp):
        from fit_coefficients import collect_alpha_data

        pairs_0, triples_12 = collect_alpha_data([exp])
        assert len(triples_12) > 0
        for departed, finished, n_tokens in triples_12:
            assert departed > finished > 0
            assert n_tokens > 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestCollectAlphaData -v`
Expected: FAIL with `ImportError: cannot import name 'collect_alpha_data'`

**Step 3: Implement collect_alpha_data**

Add to `fit_coefficients.py`:

```python
# =============================================================================
# Data collection helpers
# =============================================================================

def _extract_journey_timestamps(
    journey_events: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
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

    Journey IDs: cmpl-xxx-0-yyy → base: cmpl-xxx
    API IDs: cmpl-xxx (no suffix)
    """
    if "-0-" in journey_id:
        return journey_id.rsplit("-0-", 1)[0]
    return journey_id


def collect_alpha_data(
    experiments: list[ExperimentMeta],
) -> tuple[list[tuple[float, float]], list[tuple[float, float, int]]]:
    """Collect timestamp pairs for α₀ and α₁/α₂ estimation.

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

            # α₀ pairs: (arrived_ts, queued_ts)
            arrived = api_data["arrived_ts"]
            queued = j_data["queued_ts"]
            if queued > arrived > 0:
                pairs_0.append((arrived, queued))

            # α₁/α₂ triples: (departed_ts, finished_ts, output_tokens)
            departed = api_data["departed_ts"]
            finished = j_data["finished_ts"]
            n_tokens = j_data["output_tokens"]
            if departed > finished > 0 and n_tokens > 0:
                triples_12.append((departed, finished, n_tokens))

    return pairs_0, triples_12
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestCollectAlphaData -v`
Expected: All 2 tests PASS

Run: `python3 -m pytest -q`
Expected: All tests pass (77 + 2 = 79)

**Step 5: Commit**

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: data collection helpers for α estimation"
```

---

### Task 8: Lambda tuning and public API

**Files:**
- Modify: `fit_coefficients.py` (add tune_lambda and fit_coefficients)
- Modify: `tests/test_fit_coefficients.py` (add test classes)

**Context for implementer:**
- Grid search λ over [0, 0.01, 0.1, 1, 10, 100]
- For each λ: fit on training X/y, evaluate MSE on validation X/y
- Pick λ with lowest validation MSE
- fit_coefficients(hw) composes all phases: collect data → fit α₀ → fit α₁/α₂ → build feature matrices for train+val → tune λ → fit final β → return FittedCoefficients

**Step 1: Write the failing test**

Add to `tests/test_fit_coefficients.py`:

```python
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
    """End-to-end fitting on real data.

    Verifies that fitted coefficients are in expected ranges and
    that the fitting pipeline runs without errors.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from fit_coefficients import fit_coefficients
        from basis_functions import load_hardware_spec

        hw = load_hardware_spec("datasheets/h100-sxm.json")
        return fit_coefficients(hw)

    def test_alpha_0_in_expected_range(self, result):
        # Expected: ~5000-7000 µs
        assert result.alpha_0 > 0
        assert 1000 < result.alpha_0 < 50000

    def test_alpha_12_non_negative(self, result):
        assert result.alpha_1 >= 0
        assert result.alpha_2 >= 0

    def test_alpha_2_in_expected_range(self, result):
        # Expected: ~1-10 µs/token
        assert result.alpha_2 < 100  # sanity upper bound

    def test_all_betas_non_negative(self, result):
        assert len(result.betas) == 7
        for i, b in enumerate(result.betas):
            assert b >= 0, f"β{i+1} = {b} is negative"

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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestTuneLambda -v`
Expected: FAIL with `ImportError: cannot import name 'tune_lambda'`

**Step 3: Implement tune_lambda and fit_coefficients**

Add to `fit_coefficients.py`:

```python
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
    """Tune regularization strength λ by grid search on validation MSE.

    Requires: X_train, X_val have shape (n, 7). y arrays match row count.
    Guarantees: Returns (best_lambda, best_betas, train_mse, val_mse).
                best_betas has 7 non-negative entries.

    Returns the λ with lowest validation MSE.
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

# Expected ranges for diagnostics (from DESIGN.md)
BETA_EXPECTED_RANGES = {
    1: (1.5, 3.0, "prefill roofline correction"),
    2: (5.0, 15.0, "decode roofline correction"),
    3: (1.0, 3.0, "weight loading correction"),
    4: (0.5, 2.0, "TP communication correction"),
    5: (10.0, 50.0, "per-layer overhead (µs/layer)"),
    6: (50.0, 200.0, "per-request CPU scheduling (µs/req)"),  # wider range
    7: (100.0, 2000.0, "per-step overhead (µs)"),  # wider range
}


def fit_coefficients(hw: HardwareSpec) -> FittedCoefficients:
    """Fit all 10 crossmodel latency parameters from training data.

    Requires: traces.json and exp-config.yaml for all 16 experiments
              exist under default_args/.
    Guarantees: all α >= 0, all β >= 0, λ chosen by validation MSE.
    """
    train_exps = get_train()
    val_exps = get_validate()

    # Phase 1: α₀
    pairs_0, triples_12 = collect_alpha_data(train_exps)
    alpha_0 = estimate_alpha_0(pairs_0)

    # Phase 2: α₁, α₂
    output_tokens = np.array([t[2] for t in triples_12], dtype=np.float64)
    post_decode_us = np.array([(d - f) * 1e6 for d, f, _ in triples_12], dtype=np.float64)
    alpha_1, alpha_2 = fit_alpha_12(output_tokens, post_decode_us)

    # Phase 3: β₁-β₇
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
                    f"β{i} = {b:.4f} outside expected range [{lo}, {hi}] ({desc})"
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
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestTuneLambda -v`
Expected: PASS

Run: `python3 -m pytest tests/test_fit_coefficients.py::TestFitCoefficientsEndToEnd -v`
Expected: All 7 tests PASS (this will take a few minutes — reads all 16 experiments)

Run: `python3 -m pytest -q`
Expected: All tests pass (79 + 8 = 87)

**Step 5: Commit**

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: lambda tuning and fit_coefficients() public API"
```

---

### Task 9: Diagnostics output

**Files:**
- Modify: `fit_coefficients.py` (add main function + diagnostics)

**Context for implementer:**
- Write diagnostics to output/fit/ (gitignored)
- Print fitted values, λ tuning curve, per-experiment residuals
- Also output as JSON for downstream consumption
- When run as `python3 fit_coefficients.py`, run full pipeline and print results

**Step 1: Implement diagnostics and main**

Add to `fit_coefficients.py`:

```python
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
    Guarantees: writes coefficients.json and prints summary to stdout.
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
    train_exps = get_train()
    val_exps = get_validate()
    X_train, y_train = _collect_beta_data(train_exps, hw)
    X_val, y_val = _collect_beta_data(val_exps, hw)

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
    for exp in train_exps + val_exps:
        rec = reconstruct_experiment(exp)
        arch = load_model_arch(f"model_configs/{exp.config_json_dir}/config.json")
        basis = compute_experiment_basis(rec, arch, hw, exp.tensor_parallelism)
        X, y, _ = build_feature_matrix(rec.steps, basis, rec.labels)
        if len(X) == 0:
            continue
        pred = X @ np.array(coeffs.betas)
        resid = y - pred
        residuals.append({
            "experiment": exp.dir_name,
            "split": exp.split.value,
            "n_requests": len(y),
            "mean_residual_us": float(np.mean(resid)),
            "rmse_us": float(np.sqrt(np.mean(resid ** 2))),
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
    print(f"\n  MSE:")
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
```

**Step 2: Run the full pipeline**

Run: `python3 fit_coefficients.py`
Expected: Prints fitted parameters and writes to output/fit/

**Step 3: Verify output files exist**

Run: `ls output/fit/`
Expected: `coefficients.json`, `lambda_tuning.json`, `residuals.json`

**Step 4: Run all tests one final time**

Run: `python3 -m pytest -q`
Expected: All tests pass (87 total)

**Step 5: Commit**

```bash
git add fit_coefficients.py
git commit -m "feat: diagnostics output and CLI entry point for fit_coefficients"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add fit_coefficients command and architecture description**

In the Commands section of `CLAUDE.md`, add:
```
python3 fit_coefficients.py    # fit 10 parameters → output/fit/
```

In the Architecture section, add bullet for `fit_coefficients.py`:
```
- `fit_coefficients.py` is the coefficient fitting module. Three-phase NNLS: α₀ (mean), α₁/α₂ (NNLS), β₁-β₇ (regularized NNLS with λ tuned on validation). Public API: `fit_coefficients(hw)` → `FittedCoefficients`.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add fit_coefficients to CLAUDE.md commands and architecture"
```

---

Plan complete and saved to `docs/plans/2026-03-06-fit-coefficients.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?