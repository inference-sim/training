# fit_coefficients.py — Design

Three-phase NNLS fitting of 10 crossmodel latency model parameters (α₀–α₂, β₁–β₇).

## Changes

### trace_parser.py — new function

```python
def parse_api_events(traces_path: Path | str) -> dict[str, dict]:
    """Extract api.ARRIVED and api.DEPARTED timestamps per request.

    Reads llm_request spans from the vllm.api scope.
    Returns: {base_request_id: {"arrived_ts": float, "departed_ts": float}}

    Key differences from parse_journey_events:
    - Scope: vllm.api (not vllm.scheduler)
    - Span name: llm_request (not llm_core)
    - Timestamp attribute: event.ts.monotonic (not ts.monotonic)
    - Request IDs: no -0-xxx suffix (base ID only)

    Invariant: For every request with both timestamps, departed_ts > arrived_ts > 0.
    """
```

### fit_coefficients.py — new module

**Phase 1 — α₀ (API processing overhead)**
```
α₀ = mean(QUEUED.ts − ARRIVED.ts) across training requests [µs]
```
Simple mean. Invariant: α₀ > 0.

**Phase 2 — α₁, α₂ (post-decode overhead)**
```
post_decode(r) = α₁ + α₂ · output_tokens(r)
minimize ||Xα - y||² subject to α ≥ 0
```
Signal: DEPARTED.ts − FINISHED.ts. Features: [1, output_tokens]. Solver: scipy.optimize.nnls.

**Phase 3 — β₁–β₇ (GPU step-time model)**

Per-request feature matrix: sum per-step basis values along request's active steps.
```
X_r = [Σ max(t_pf_compute, t_pf_kv),  Σ max(t_dc_compute, t_dc_kv),
       Σ t_weight,  Σ t_tp,  Σ L,  Σ batch_size,  num_active_steps]
```

Regularized NNLS:
```
minimize ||Xβ - y||² + λ · Σᵢ₌₁⁴ (βᵢ - 1)²  subject to β ≥ 0
```
Implemented via augmented matrix trick. λ tuned by grid search on validation set.

Target: RequestLabel.processing_us (excludes post-decode, ends at FINISHED).

**Output type:**
```python
@dataclass(frozen=True)
class FittedCoefficients:
    alpha_0: float
    alpha_1: float
    alpha_2: float
    betas: tuple[float, ...]  # (β₁..β₇)
    lambda_val: float
    train_mse: float
    val_mse: float
```

**Public API:**
```python
def fit_coefficients(hw: HardwareSpec) -> FittedCoefficients
```

**Diagnostics** written to `output/fit/`.

## Decisions

- Skip failed requests (label.failed=True) in feature matrix.
- Journey-to-API ID mapping: strip `-0-xxx` suffix from journey IDs.
- scipy added to pyproject.toml dependencies.

## Implementation order

1. `parse_api_events()` in trace_parser.py + test
2. Phase 1 (α₀) + test
3. Phase 2 (α₁, α₂) + test
4. Per-request feature matrix builder + test
5. Phase 3 (β₁–β₇) + test
6. Public API composition + end-to-end test
7. Diagnostics output
