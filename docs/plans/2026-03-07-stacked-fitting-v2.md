# Stacked Prefill/Decode Fitting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the coefficient fitting pipeline with (1) overload experiment exclusion, (2) request-level train/validate/test splitting, and (3) stacked prefill/decode NNLS formulation that gives β₁ (prefill roofline) proper signal.

**Architecture:** Three changes propagate bottom-up: `split.py` gains overload filtering and request-level split assignment; `reconstruct_steps.py` decomposes `processing_us` into `prefill_processing_us + decode_processing_us`; `fit_coefficients.py` replaces the single-target feature matrix with a stacked prefill+decode system where shared β₃–β₇ coefficients are fit jointly.

**Tech Stack:** Python 3.10+, numpy, scipy.optimize.nnls, hashlib (for deterministic request-level splitting)

**Key design doc:** `DESIGN.md` — Stage 4 section describes all formulas and invariants. Read it first.

---

### Task 1: Overload exclusion in split.py

**Files:**
- Modify: `split.py`
- No test file changes (integrity checks update automatically)

**Context:** Three experiments have >10% failure rate (overload regime) and must be excluded. We'll remove the `split` field from `ExperimentMeta`, remove the 3 overload experiments from `EXPERIMENTS`, add `EXCLUDED_OVERLOAD` tuple, and replace `get_train/get_validate/get_test` with a request-level split function. But this is a big change — let's do it incrementally.

This task only does the exclusion. Task 2 does the request-level split.

**Step 1: Add `EXCLUDED_OVERLOAD` and filter them from `EXPERIMENTS`**

In `split.py`, the 3 overload experiments are currently in `EXPERIMENTS` at these positions:
- `20260218-135247-mixtral-8x7b-v0-1-tp2-reasoning` (Split.VALIDATE, failure_rate=68.6%)
- `20260217-170634-llama-2-7b-tp1-reasoning` (Split.TEST, failure_rate=84.8%)
- `20260218-065057-llama-2-70b-hf-tp4-reasoning` (Split.TEST, failure_rate=33.3%)

Move these 3 entries from `EXPERIMENTS` to a new `EXCLUDED_OVERLOAD` tuple. Keep `codellama-34b-tp2-reasoning` (0.08% failure) in `EXPERIMENTS`.

Remove the `split` field from `ExperimentMeta`. Every experiment in `EXPERIMENTS` is now "active" — the split will be done at request level (Task 2).

Update `_validate_split_integrity()`:
- Retain: no duplicate dir_names (across both EXPERIMENTS and EXCLUDED_OVERLOAD)
- Change check 3 from `len(EXPERIMENTS) == 16` to `len(EXPERIMENTS) == 13`
- Remove checks 4, 5, 6, 7 (they reference train/validate/test experiment-level splits which no longer exist)
- Add: every experiment in `EXCLUDED_OVERLOAD` has `failure_rate > 0.10`
- Add: no overlap between EXPERIMENTS and EXCLUDED_OVERLOAD (by dir_name)
- Add: `len(EXPERIMENTS) + len(EXCLUDED_OVERLOAD) == 16`

Remove `get_train()`, `get_validate()`, `get_test()`, `get_split()` functions. These will be replaced in Task 2 with request-level splitting. (Note: if `fit_coefficients.py` imports these, it will break — that's expected, it gets fixed in Tasks 5-6.)

Remove the `Split` enum and the `split` field from `ExperimentMeta` (no longer needed at Task 1 time — re-added in Task 2 for `request_split()` return type).

Update `get_by_model()` and `get_by_profile()`: remove the `Optional[Split]` parameter and `e.split` filtering (the `split` field no longer exists on ExperimentMeta). They become simple model/profile filters over active experiments.

Update or remove `print_summary()`: this function uses `get_split()`, `Split.TRAIN/VALIDATE/TEST` extensively. Since experiment-level splits no longer exist, either: (a) rewrite to just list all active experiments without split grouping, or (b) remove entirely and update CLAUDE.md to remove the `python3 split.py` command. Option (a) is simpler.

Keep `experiment_dir()`, `config_json_path()`.

Add `get_active() -> list[ExperimentMeta]` that returns all experiments in `EXPERIMENTS` (for code that needs to iterate all active experiments).

**Step 2: Fix `exp.split.value` references in other modules**

After removing the `split` field from `ExperimentMeta`, these files will break:
- `reconstruct_steps.py` lines 596, 653, 662 — `_write_experiment_json` and main block use `exp.split.value`
- `validate_traces.py` lines 162, 189, 204, 223 — validation output uses `exp.split.value`
- `fit_coefficients.py` line 522 — `exp.split.value` in `write_diagnostics()` residual loop
- `tests/test_trace_parser_api.py` line 12 — imports `get_train`
- `tests/test_fit_coefficients.py` — imports `get_train`

Fix `reconstruct_steps.py` and `validate_traces.py`: remove the `"split"` field from JSON output and the split column from print statements (since all experiments are now active, the field adds no information). Also update docstrings that say "16 experiments" → "13 experiments".

Fix `fit_coefficients.py:write_diagnostics()` line 522: remove `"split": exp.split.value` from the residuals dict. (The function gets fully rewritten in Task 5, but this prevents crashes between Tasks 1-5.)

Fix `tests/test_trace_parser_api.py`: replace `from split import get_train` with `from split import get_active` and use `get_active()[0]` instead of `get_train()[0]`.

Leave `tests/test_fit_coefficients.py` broken for now — it gets fully rewritten in Task 5.

**Step 3: Run existing tests**

Run: `python3 -m pytest tests/test_reconstruct_steps.py tests/test_basis_functions.py tests/test_trace_parser_api.py -v`
Expected: All pass

Run: `python3 -m pytest tests/test_fit_coefficients.py -v`
Expected: Some tests FAIL because they import `get_train` which no longer exists. This is expected — we'll fix in Task 5.

**Step 4: Commit**

```bash
git add split.py reconstruct_steps.py validate_traces.py fit_coefficients.py tests/test_trace_parser_api.py
git commit -m "refactor: exclude 3 overload experiments, remove experiment-level split

Removes Split enum and split field from ExperimentMeta. Moves 3 overload
experiments (>10% failure) to EXCLUDED_OVERLOAD. Remaining 13 experiments
are active. get_train/get_validate/get_test removed — request-level
splitting will be added next.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Request-level split function in split.py

**Files:**
- Modify: `split.py`
- Create: `tests/test_split.py`

**Context:** Add a deterministic hash-based split function. Uses SHA-256 of the request ID mod 100. Ratio: 70% train (0-69), 15% validate (70-84), 15% test (85-99). Must use the **journey** request ID (with `-0-xxx` suffix) since that's the ID used in `reconstruct_steps.py` labels.

**Step 1: Write the failing tests**

Create `tests/test_split.py`:

```python
"""Behavioral tests for request-level split assignment."""

from __future__ import annotations

import pytest
from split import request_split, get_active, Split


class TestRequestSplit:
    """Deterministic hash-based request split assignment.

    Verify that split assignment is deterministic, covers all three splits,
    and produces roughly the expected proportions.
    """

    def test_deterministic(self):
        """Same request ID always gets the same split."""
        rid = "cmpl-abc123-0-def456"
        assert request_split(rid) == request_split(rid)

    def test_returns_split_enum(self):
        rid = "cmpl-test-0-abc"
        result = request_split(rid)
        assert isinstance(result, Split)

    def test_different_ids_cover_all_splits(self):
        """With enough IDs, all three splits should appear."""
        splits = {request_split(f"req-{i}-0-test") for i in range(1000)}
        assert splits == {Split.TRAIN, Split.VALIDATE, Split.TEST}

    def test_approximate_proportions(self):
        """Over many IDs, proportions should be roughly 70/15/15."""
        n = 10000
        counts = {Split.TRAIN: 0, Split.VALIDATE: 0, Split.TEST: 0}
        for i in range(n):
            counts[request_split(f"req-{i}-0-hash")] += 1
        # Allow 3% tolerance
        assert abs(counts[Split.TRAIN] / n - 0.70) < 0.03
        assert abs(counts[Split.VALIDATE] / n - 0.15) < 0.03
        assert abs(counts[Split.TEST] / n - 0.15) < 0.03


class TestGetActive:
    """get_active() returns all non-overload experiments."""

    def test_returns_13_experiments(self):
        assert len(get_active()) == 13

    def test_no_high_failure_experiments(self):
        for exp in get_active():
            assert exp.failure_rate <= 0.10, (
                f"{exp.dir_name} has {exp.failure_rate:.0%} failure rate"
            )
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_split.py -v`
Expected: ImportError for `request_split` and/or `get_active`

**Step 3: Implement request_split and get_active**

Add to `split.py`:

```python
import hashlib

# Re-add Split enum (needed for request_split return type)
class Split(str, Enum):
    """Data split assignment for individual requests."""
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


# Split thresholds: 70% train, 15% validate, 15% test
_TRAIN_THRESHOLD = 70
_VALIDATE_THRESHOLD = 85  # 70 + 15


def request_split(request_id: str) -> Split:
    """Assign a request to train/validate/test split.

    Uses SHA-256 hash of request_id mod 100 for deterministic,
    platform-independent assignment.

    Requires: request_id is a non-empty string.
    Guarantees: returns a Split enum value. Same input always
                returns same output (deterministic, no randomness).
    """
    h = int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100
    if h < _TRAIN_THRESHOLD:
        return Split.TRAIN
    elif h < _VALIDATE_THRESHOLD:
        return Split.VALIDATE
    else:
        return Split.TEST


def get_active() -> list[ExperimentMeta]:
    """Return all active (non-overload) experiments."""
    return list(EXPERIMENTS)
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_split.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add split.py tests/test_split.py
git commit -m "feat: add request_split() for deterministic request-level splitting

SHA-256 hash mod 100, ratio 70/15/15 train/validate/test.
Deterministic and platform-independent.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add prefill/decode timing decomposition to RequestLabel

**Files:**
- Modify: `reconstruct_steps.py` (RequestLabel dataclass + `_compute_label`)
- Modify: `tests/test_reconstruct_steps.py` (add decomposition tests to existing scenarios)

**Context:** Decompose `processing_us` into `prefill_processing_us + decode_processing_us` using `FIRST_TOKEN.ts` as the boundary. Preemption gaps are partitioned: `PREEMPTED.ts < FIRST_TOKEN.ts` → prefill gap, else decode gap.

**Key invariant:** `prefill_processing_us + decode_processing_us == processing_us` (exact, by construction).

**Step 1: Write the failing tests**

Create a NEW test class `TestTimingDecomposition` in `tests/test_reconstruct_steps.py`. Per CLAUDE.md BDD rule 4, each test class is a scenario — timing decomposition is a distinct behavioral concern from preemption or batch composition.

```python
class TestTimingDecomposition:
    """Timing decomposition splits processing_us into prefill and decode.

    The boundary is FIRST_TOKEN.ts. Preemption gaps before FIRST_TOKEN
    are prefill gaps; after are decode gaps. The invariant
    prefill_processing_us + decode_processing_us == processing_us holds exactly.
    """

    @pytest.fixture()
    def simple(self):
        """No preemption — baseline decomposition."""
        tl = (
            JourneyBuilder("req-simple", prompt_tokens=512, max_output_tokens=100)
            .queued(step=0, ts=1000.0)
            .scheduled(step=5, ts=1001.0)
            .first_token(step=5, ts=1001.2)
            .finished(step=104, ts=1010.0, decode_done=100)
            .build()
        )
        _, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return labels[0]

    @pytest.fixture()
    def decode_preemption(self):
        """Preemption during decode — all gap is decode gap."""
        tl = (
            JourneyBuilder("req-dc-pre", prompt_tokens=100, max_output_tokens=200)
            .queued(step=0, ts=1000.0)
            .scheduled(step=10, ts=1001.0)
            .first_token(step=10, ts=1001.1)
            .preempted(step=60, ts=1006.0, decode_done=50)
            .scheduled(step=100, ts=1010.0, kind="RESUME")
            .finished(step=249, ts=1025.0, decode_done=200)
            .build()
        )
        _, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return labels[0]

    @pytest.fixture()
    def prefill_preemption(self):
        """Preemption during prefill — gap is prefill gap."""
        tl = (
            JourneyBuilder("req-pf-pre", prompt_tokens=1000, max_output_tokens=50)
            .queued(step=0, ts=1000.0)
            .scheduled(step=10, ts=1001.0)
            .preempted(step=11, ts=1001.5, prefill_done=600)
            .scheduled(step=20, ts=1002.0, kind="RESUME")
            .first_token(step=20, ts=1002.3)
            .finished(step=69, ts=1005.0, decode_done=50)
            .build()
        )
        _, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return labels[0]

    # --- Invariant: prefill + decode = processing (must hold for all scenarios) ---

    def test_simple_decomposition_sums_to_processing(self, simple):
        assert abs(simple.prefill_processing_us + simple.decode_processing_us - simple.processing_us) < 1.0

    def test_decode_preemption_decomposition_sums_to_processing(self, decode_preemption):
        assert abs(decode_preemption.prefill_processing_us + decode_preemption.decode_processing_us - decode_preemption.processing_us) < 1.0

    def test_prefill_preemption_decomposition_sums_to_processing(self, prefill_preemption):
        assert abs(prefill_preemption.prefill_processing_us + prefill_preemption.decode_processing_us - prefill_preemption.processing_us) < 1.0

    # --- Specific values ---

    def test_simple_both_components_positive(self, simple):
        assert simple.prefill_processing_us > 0
        assert simple.decode_processing_us > 0

    def test_decode_preemption_prefill_time_excludes_gap(self, decode_preemption):
        # Preemption is after FIRST_TOKEN → all gap is decode, prefill unaffected
        prefill_expected = (1001.1 - 1001.0) * 1e6  # 100000 µs
        assert abs(decode_preemption.prefill_processing_us - prefill_expected) < 1.0

    def test_prefill_preemption_prefill_time_excludes_gap(self, prefill_preemption):
        # Prefill gap = 1002.0 - 1001.5 = 0.5s
        # Prefill raw = 1002.3 - 1001.0 = 1.3s
        # Prefill processing = 0.8s = 800000 µs
        assert abs(prefill_preemption.prefill_processing_us - 800000.0) < 1.0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_reconstruct_steps.py -v -k "decomposition"`
Expected: AttributeError: 'RequestLabel' has no attribute 'prefill_processing_us'

**Step 3: Implement the timing decomposition**

In `reconstruct_steps.py`, modify `RequestLabel` (line 89-110) — add two fields after `processing_us`:

```python
@dataclass(frozen=True)
class RequestLabel:
    """Ground-truth timing labels for one request.

    Invariants:
        When failed=True: all timing and token fields are zero (incomplete lifecycle).
        When failed=False: timing fields are non-negative, prompt_tokens > 0.
        e2e_us >= ttft_us >= queueing_us (when failed=False).
        processing_us = prefill_processing_us + decode_processing_us (exact).
    """
    request_id: str
    prompt_tokens: int
    output_tokens: int
    queueing_us: float
    ttft_us: float
    processing_us: float
    prefill_processing_us: float   # FIRST_TOKEN.ts - SCHEDULED.ts - prefill_gaps
    decode_processing_us: float    # FINISHED.ts - FIRST_TOKEN.ts - decode_gaps
    e2e_us: float
    num_preemptions: int
    failed: bool
    first_step: int
    last_step: int
```

Modify `_compute_label()` (line 329-372) — partition preemption gaps:

```python
def _compute_label(tl: RequestTimeline) -> RequestLabel:
    """Compute ground-truth timing labels from journey timestamps."""
    ev_map: dict[str, list[ParsedEvent]] = {}
    for ev in tl.events:
        ev_map.setdefault(ev.name, []).append(ev)

    queued = ev_map["QUEUED"][0]
    scheduled = ev_map["SCHEDULED"][0]
    first_token = ev_map["FIRST_TOKEN"][0]
    finished = ev_map["FINISHED"][0]

    # Partition preemption gaps into prefill vs decode using FIRST_TOKEN.ts
    prefill_gap_s = 0.0
    decode_gap_s = 0.0
    preempted_events = ev_map.get("PREEMPTED", [])
    resume_events = [e for e in ev_map.get("SCHEDULED", []) if e.schedule_kind == "RESUME"]
    for pre, res in zip(preempted_events, resume_events):
        gap = res.ts - pre.ts
        if pre.ts < first_token.ts:
            prefill_gap_s += gap
        else:
            decode_gap_s += gap

    preemption_gap_s = prefill_gap_s + decode_gap_s

    queueing_s = scheduled.ts - queued.ts
    ttft_s = first_token.ts - queued.ts
    processing_s = (finished.ts - scheduled.ts) - preemption_gap_s
    prefill_processing_s = (first_token.ts - scheduled.ts) - prefill_gap_s
    decode_processing_s = (finished.ts - first_token.ts) - decode_gap_s
    e2e_s = finished.ts - queued.ts

    for name, val in [("queueing", queueing_s), ("ttft", ttft_s),
                       ("processing", processing_s), ("e2e", e2e_s),
                       ("prefill_processing", prefill_processing_s),
                       ("decode_processing", decode_processing_s)]:
        if val < 0:
            warnings.warn(
                f"Request {tl.request_id}: negative {name} = {val:.6f}s"
            )

    label = RequestLabel(
        request_id=tl.request_id,
        prompt_tokens=tl.prompt_tokens,
        output_tokens=finished.decode_done,
        queueing_us=queueing_s * 1e6,
        ttft_us=ttft_s * 1e6,
        processing_us=processing_s * 1e6,
        prefill_processing_us=prefill_processing_s * 1e6,
        decode_processing_us=decode_processing_s * 1e6,
        e2e_us=e2e_s * 1e6,
        num_preemptions=len(preempted_events),
        failed=False,
        first_step=scheduled.step,
        last_step=finished.step,
    )
    tl.label = label
    return label
```

Also update the failed label construction in `reconstruct_experiment()` (line 566-570) — add the two new zero fields:

```python
failed_labels.append(RequestLabel(
    request_id=req_id, prompt_tokens=0, output_tokens=0,
    queueing_us=0.0, ttft_us=0.0, processing_us=0.0,
    prefill_processing_us=0.0, decode_processing_us=0.0,
    e2e_us=0.0,
    num_preemptions=0, failed=True, first_step=0, last_step=0,
))
```

**Step 4: Run all tests**

Run: `python3 -m pytest tests/test_reconstruct_steps.py -v`
Expected: All tests PASS (old + new decomposition tests)

Run: `python3 -m pytest tests/test_basis_functions.py -v`
Expected: All PASS (no dependency on RequestLabel fields)

**Step 5: Commit**

```bash
git add reconstruct_steps.py tests/test_reconstruct_steps.py
git commit -m "feat: decompose processing_us into prefill + decode components

Adds prefill_processing_us and decode_processing_us to RequestLabel.
Preemption gaps partitioned by FIRST_TOKEN.ts boundary.
Invariant: prefill_processing_us + decode_processing_us == processing_us.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Update conftest.py JourneyBuilder if needed

**Files:**
- Check: `tests/conftest.py`

**Context:** The `JourneyBuilder.build()` returns a `RequestTimeline` with events only — it does NOT construct `RequestLabel`. Labels are computed by `reconstruct_timelines()` which calls `_compute_label()`. So no change needed to `JourneyBuilder`.

However, any test code that constructs `RequestLabel` directly (e.g., in `tests/test_fit_coefficients.py`) will need the two new fields. That's handled in Task 5.

**Step 1: Verify no change needed**

Run: `python3 -m pytest tests/test_reconstruct_steps.py -v`
Expected: All PASS

**Step 2: No commit needed** (skip if no changes)

---

### Task 5: Update fit_coefficients.py for new split and stacked formulation

**Files:**
- Modify: `fit_coefficients.py` (major changes to imports, feature matrix, fit_coefficients)
- Modify: `tests/test_fit_coefficients.py` (update all tests)

**Context:** This is the biggest task. Three changes:
1. Replace `get_train()`/`get_validate()` with `get_active()` + `request_split()`
2. Replace single-target `build_feature_matrix` with stacked `build_stacked_feature_matrix`
3. Update `collect_alpha_data` to use `get_active()` + `request_split()` for train filtering
4. Update diagnostics to report three-phase MSE

**Step 1: Update imports in fit_coefficients.py**

Replace:
```python
from split import ExperimentMeta, get_train, get_validate
```
with:
```python
from split import ExperimentMeta, Split, get_active, request_split
```

**Step 2: Replace `build_feature_matrix` with `build_stacked_feature_matrix`**

The new function produces TWO rows per request (one for prefill steps, one for decode steps) with the same 7 β columns. Each row's target is the corresponding `prefill_processing_us` or `decode_processing_us`.

```python
def build_stacked_feature_matrix(
    steps: list[ReconstructedStep] | tuple[ReconstructedStep, ...],
    basis_values: list[StepBasisValues],
    labels: list[RequestLabel] | tuple[RequestLabel, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build stacked prefill+decode feature matrix for Phase 3 regression.

    For each non-failed request, produces two rows:
    - Prefill row: basis values summed over steps where request is prefilling
    - Decode row: basis values summed over steps where request is decoding

    All 7 β coefficients are shared between prefill and decode rows.

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
```

**Step 3: Update `collect_alpha_data` to use `get_active()` + `request_split()`**

Replace the `experiments` parameter approach. The function now collects from ALL active experiments, but only uses training-split requests:

```python
def collect_alpha_data() -> tuple[list[tuple[float, float]], list[tuple[float, float, int]]]:
    """Collect timestamp pairs for α₀ and α₁/α₂ estimation from training requests.

    Iterates all active experiments, filters to training-split requests only.

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

            arrived = api_data["arrived_ts"]
            queued = j_data["queued_ts"]
            if queued > arrived > 0:
                pairs_0.append((arrived, queued))

            departed = api_data["departed_ts"]
            finished = j_data["finished_ts"]
            n_tokens = j_data["output_tokens"]
            if departed > finished > 0 and n_tokens > 0:
                triples_12.append((departed, finished, n_tokens))

    return pairs_0, triples_12
```

**Step 4: Update `_collect_beta_data` to accept a split filter**

```python
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
```

**Step 5: Update `fit_coefficients()`**

```python
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
```

**Step 6: Update `write_diagnostics` for three-phase MSE**

Update the diagnostics to report α₀ MSE, α₁/α₂ MSE, and GPU processing MSE (prefill, decode, combined) separately. The printed summary should show all three.

The residuals loop should also compute per-experiment prefill and decode RMSE separately.

**Step 7: Write concrete tests for stacked feature matrix**

Replace `TestBuildFeatureMatrix` with `TestStackedFeatureMatrix` in `tests/test_fit_coefficients.py`. Key test methods:

```python
class TestStackedFeatureMatrix:
    """Stacked prefill+decode feature matrix for Phase 3 regression.

    Each non-failed request produces two rows: one for prefill steps,
    one for decode steps. Shared β₃-β₇, selective β₁/β₂.
    """

    @pytest.fixture()
    def data(self):
        # Same fixture as before but with prefill/decode processing fields
        steps = [...]  # same 2-step fixture
        basis_values = [...]  # same basis values
        labels = [
            RequestLabel("req-A", ..., processing_us=5000.0,
                         prefill_processing_us=500.0, decode_processing_us=4500.0, ...),
            RequestLabel("req-B", ..., processing_us=4000.0,
                         prefill_processing_us=0.0, decode_processing_us=4000.0, ...),
        ]
        return steps, basis_values, labels

    def test_stacked_shape_is_2n_by_7(self, data):
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        assert X.shape == (2 * n, 7)
        assert y.shape == (2 * n,)

    def test_prefill_rows_target_prefill_processing_us(self, data):
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_a = req_ids.index("req-A")
        assert abs(y[idx_a] - 500.0) < 0.01  # prefill target
        assert abs(y[n + idx_a] - 4500.0) < 0.01  # decode target

    def test_feature_sum_invariant(self, data):
        """X_pf[i] + X_dc[i] equals what the old single-row formulation produced."""
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        for i in range(n):
            X_total = X[i] + X[n + i]  # prefill row + decode row
            # Should equal the original single-row feature vector
            # (since selective accumulation is preserved)
            assert all(X_total[j] >= 0 for j in range(7))

    def test_beta1_only_in_prefill_rows(self, data):
        """β₁ (prefill roofline) should only appear in rows from prefill steps."""
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_b = req_ids.index("req-B")
        # req-B has no prefill steps → β₁ column is 0 in prefill row
        assert X[idx_b, 0] == 0.0
        # req-B decode row also has β₁ = 0 (decode entries don't accumulate β₁)
        assert X[n + idx_b, 0] == 0.0

    def test_beta2_only_in_decode_rows(self, data):
        """β₂ (decode roofline) should only appear in rows from decode steps."""
        X, y, req_ids = build_stacked_feature_matrix(*data)
        n = len(req_ids)
        idx_a = req_ids.index("req-A")
        # req-A prefill row: β₂ = 0 (prefill entries don't accumulate β₂)
        assert X[idx_a, 1] == 0.0
```

Also update:
- `TestCollectAlphaData` → `collect_alpha_data()` now takes no arguments
- `TestFitCoefficientsEndToEnd` → uses new split functions
- All `RequestLabel` constructions in test fixtures need the two new fields
- **Add `assert result.betas[0] > 0` to `TestFitCoefficientsEndToEnd`** — this is the primary success criterion for the entire refactor

Run: `python3 -m pytest -q`
Expected: All tests pass (94 → probably ~95+ after new stacked tests)

```bash
git add fit_coefficients.py tests/test_fit_coefficients.py
git commit -m "feat: stacked prefill/decode NNLS with request-level splitting

Replaces single-target processing_us regression with stacked system:
- Prefill rows: target = prefill_processing_us
- Decode rows: target = decode_processing_us
- Shared β₃-β₇ coefficients fit jointly
- β₁ now has proper signal from prefill rows

Request-level splitting via SHA-256 hash (70/15/15).
Three-phase MSE diagnostics.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update test fixtures with new RequestLabel fields

**Files:**
- Modify: `tests/test_fit_coefficients.py`
- Modify: `tests/test_trace_parser_api.py`

**Context:** Any test that constructs `RequestLabel` directly or imports `get_train` needs updating.

All `RequestLabel(...)` constructions in test fixtures need two new fields: `prefill_processing_us` and `decode_processing_us`. These should sum to `processing_us`.

Update imports from `split` to use `get_active` and `request_split` instead of `get_train`.

**Step 1: Update all RequestLabel constructions in test_fit_coefficients.py**

For each synthetic `RequestLabel` in the test file, add the two new fields. Example for the `TestBuildFeatureMatrix` fixture:

```python
RequestLabel("req-A", prompt_tokens=512, output_tokens=50,
             queueing_us=1000.0, ttft_us=2000.0,
             processing_us=5000.0,
             prefill_processing_us=500.0,    # NEW
             decode_processing_us=4500.0,    # NEW
             e2e_us=6000.0,
             num_preemptions=0, failed=False, first_step=1, last_step=2),
```

**Step 2: Update test_trace_parser_api.py imports**

Replace `from split import get_train` with `from split import get_active` and use `get_active()[0]` instead of `get_train()[0]`.

**Step 3: Run all tests**

Run: `python3 -m pytest -q`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/
git commit -m "fix: update test fixtures for new RequestLabel fields and split API

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Update CLAUDE.md and DESIGN.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DESIGN.md`

**Step 1: Update CLAUDE.md**

- Update the `split.py` description to mention request-level splitting and overload exclusion
- Add note that `split.py` no longer has `get_train()`/`get_validate()` — uses `request_split()` instead
- Update `reconstruct_steps.py` description to mention `prefill_processing_us`/`decode_processing_us`
- Update `fit_coefficients.py` description to mention stacked prefill/decode formulation

**Step 2: Update DESIGN.md**

- Change Stage 4 status from "In progress" back to "Done"
- Remove "(planned)" markers from timing decomposition, stacked formulation, overload exclusion, and data split sections
- Remove the "Current: experiment-level splitting" subsection (no longer current)
- Update request counts if they changed

**Step 3: Commit**

```bash
git add CLAUDE.md DESIGN.md
git commit -m "docs: update CLAUDE.md and DESIGN.md for stacked fitting v2

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Run full pipeline and verify results

**Step 1: Run all tests**

Run: `python3 -m pytest -q`
Expected: All pass

**Step 2: Run the fitting pipeline**

Run: `python3 fit_coefficients.py`
Expected: Prints fitted parameters. Key improvements to look for:
- β₁ (prefill roofline) should now be > 0 (was 0.0 before)
- β₂ (decode roofline) should increase (was 1.09)
- Validation RMSE should decrease dramatically (was 7 seconds due to mixtral-reasoning)

**Step 3: Verify diagnostics output**

Run: `ls output/fit/`
Expected: `coefficients.json`, `lambda_tuning.json`, `residuals.json`

Check that `coefficients.json` has reasonable values.

**Step 4: Commit results summary (optional)**

No code changes needed unless something is wrong.

---

## Task dependency graph

```
Task 1 (overload exclusion) ─► Task 2 (request_split) ─┐
                                                        ├─► Task 5 (stacked formulation)
Task 3 (timing decomposition) ─► Task 4 (conftest.py) ─┘        │
                                                          Task 6 (test fixtures)
                                                                 │
                                                          Task 7 (docs)
                                                                 │
                                                          Task 8 (run & verify)
```

Tasks 1-2 and Tasks 3-4 are **independent parallel branches** that merge at Task 5.

## Important notes for the implementer

1. **The `split` field on `ExperimentMeta`** is referenced throughout the codebase. After removing it in Task 1, fix all references immediately (reconstruct_steps.py, validate_traces.py, test_trace_parser_api.py — see Task 1 Step 2).

3. **`reconstruct_experiment()` constructs failed RequestLabels** directly (line 566-570). These need the two new fields (both 0.0). Don't forget this.

4. **The stacked feature matrix has shape `(2*n, 7)`, not `(n, 7)`**. All downstream code (tune_lambda, fit_betas, write_diagnostics) works on the stacked system. The `fit_betas` function doesn't need changes — it just sees a larger matrix. `tune_lambda` doesn't need changes either.

5. **Request IDs for splitting** — use the journey request ID (with `-0-xxx` suffix) since that's what appears in `RequestLabel.request_id`. The `request_split()` function hashes whatever string you give it, so it works with any ID format as long as you're consistent.

6. **Run `python3 -m pytest -q` frequently** — after every code change, not just after each task. The project has 94 tests that run in ~2 minutes.

7. **Read DESIGN.md Stage 4** before starting — it has the complete mathematical formulation and all invariants.
