# CLAUDE.md

## Project

Coefficient fitting pipeline for the inference-sim crossmodel latency model. Reconstructs per-step GPU batch composition from vLLM journey traces, then fits analytical model parameters.

## Commands

```bash
pytest                         # run tests (< 1s)
python3 validate_traces.py      # validate all 16 experiments → output/validate/
python3 reconstruct_steps.py    # reconstruct steps → output/reconstruct/
python3 fit_coefficients.py     # fit 10 parameters → output/fit/
python3 split.py                # print train/validate/test summary
```

## Architecture

- `split.py` is the **single source of truth** for experiment metadata. All modules import from it. Never hard-code experiment lists elsewhere. Validates its own integrity on import (7 assertions).
- `trace_parser.py` provides shared OTEL parsing. Owns all knowledge of the JSONL/span nesting format.
- `schemas.py` defines Pydantic schemas for all data file formats (traces, metrics, configs).
- `reconstruct_steps.py` is the core module. Public API: `reconstruct_experiment()` (end-to-end) and `reconstruct_timelines()` (testable core, no filesystem).
- `basis_functions.py` computes analytical roofline basis functions (µs) per step. Each basis function is a standalone pure function for extensibility. Public API: `compute_step_basis()` and `compute_experiment_basis()`.
- `fit_coefficients.py` is the coefficient fitting module. Three-phase NNLS: α₀ (mean), α₁/α₂ (NNLS), β₁-β₇ (regularized NNLS with λ tuned on validation). Public API: `fit_coefficients(hw)` → `FittedCoefficients`.
- `validate_traces.py` is independent verification. Does NOT import from `reconstruct_steps.py`.

Output structure: `output/validate/<exp>.json`, `output/reconstruct/<exp>.json` (each with a `summary.json`), and `output/fit/coefficients.json`.

## Key invariants

These invariants are documented inline where they are enforced. When adding code, state the invariants each function guarantees and requires.

- Prefill completes in 1 step for all current data (validated by check 5), but code handles multi-step chunked prefill.
- Decode produces exactly 1 token per step.
- `PREEMPTED@P` means the request is NOT active at step P (interval ends at P-1).
- `FINISHED@D` means the request IS active at step D (inclusive).
- Events are sorted by `(step, ts)` before state machine processing — this is a correctness requirement, not an optimization. Out-of-order OTEL exports break the state machine without it.
- `decode_tokens_before[iv_idx]` enables correct context_length across preemption gaps. Without it, context_length overcounts by the number of preempted steps.
- `prefill_remaining` is algorithm-local state in `_reconstruct_steps`, NOT stored on `RequestTimeline`. This prevents mixing scratch state with data.
- Output dataclasses (`ReconstructedStep`, `RequestLabel`, etc.) are frozen. Their invariants are documented in their docstrings.

## Invariant documentation discipline

Every function that transforms data should document:

1. **Requires** — what the caller must guarantee (preconditions).
2. **Guarantees** — what the function provides to its callers (postconditions).
3. **Invariants** — relationships that hold throughout the data structure's lifetime.

These go in docstrings or in comment blocks above the function. See `_reconstruct_steps` and `_build_intervals` for examples. When reviewing code, check that invariants are stated and that tests verify them behaviorally.

## Testing discipline (BDD)

Tests are behavioral — they verify WHAT the system does, not HOW.

### Rules

1. **Tests assert on output types only** — `ReconstructedStep`, `RequestLabel`, and their fields. Never inspect internal types like `Interval` or `RequestTimeline.intervals`.
2. **Tests call public API functions** — `reconstruct_timelines()` for pipeline tests, `parse_events()` for parsing boundary tests. Never call `_build_intervals` or `_compute_label` directly.
3. **`JourneyBuilder.build()` returns raw timelines** — events only, no intervals or labels computed. The test passes timelines to `reconstruct_timelines()` which runs the full pipeline. This ensures tests exercise the real orchestration.
4. **Each test class is a scenario** — describes a real scheduling situation (preemption, chunked prefill, concurrent batch). The class docstring explains what behavior is being verified.
5. **Test names describe expected behavior**, not implementation: `test_context_length_at_resume_accounts_for_gap` not `test_build_intervals_creates_two_decode_intervals`.

### Adding a new test

```python
class TestMyScenario:
    """Description of the scheduling situation and what behavior matters."""

    @pytest.fixture()
    def result(self):
        tl = JourneyBuilder("req", prompt_tokens=...).queued(...).scheduled(...)...build()
        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return steps, labels

    def test_the_observable_behavior(self, result):
        steps, labels = result
        # Assert on ReconstructedStep and RequestLabel fields ONLY
```

## Adding a new experiment

1. Add `ExperimentMeta` entry to `EXPERIMENTS` in `split.py`.
2. Update the count assertion in `_validate_split_integrity` (currently expects 16).
3. Place data in `default_args/<experiment>/`.
4. Run `python validate_traces.py` to verify integrity.

## Style

- Python 3.10+, type hints, dataclasses for data structures.
- Output dataclasses are `frozen=True`. Internal working types are mutable.
- `output/` is gitignored — scripts write JSON there.
- Warnings (not silent defaults) for unexpected data conditions.
