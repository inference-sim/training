"""Reconstruct per-step batch composition from journey trace data.

For each experiment, parses journey events from traces.json and rebuilds
what the scheduler batch looked like at every step: which requests were
prefilling (and how many tokens), which were decoding (and their context
lengths).  This is the "teacher-forced" reconstruction described in
inference-sim/training#3 — using real batch compositions from the actual
system execution, not simulated ones.

Also computes per-request ground-truth timing labels (queueing, TTFT,
processing time) derived from journey event timestamps.

Handles chunked prefill: if a request's prompt spans multiple scheduler
steps (FIRST_TOKEN.step > SCHEDULED.step), tokens are distributed using
greedy fill with the experiment's max_num_batched_tokens budget, mirroring
vLLM's scheduler behavior.

Public API
----------
reconstruct_experiment(exp) -> ExperimentReconstruction
    End-to-end: traces.json → steps + labels.

reconstruct_timelines(timelines, budget) -> (steps, labels)
    Testable core: takes parsed timelines, returns steps + labels.
    This is the function behavioral tests should target.

Usage:
    python reconstruct_steps.py          # process all 16 experiments
    python -c "from reconstruct_steps import reconstruct_experiment; ..."
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from split import EXPERIMENTS, ExperimentMeta
from trace_parser import attr_map, parse_journey_events, load_exp_config, traces_path_for

OUTPUT_DIR = Path("output/reconstruct")


# =============================================================================
# Output types (frozen — these are the behavioral contract)
# =============================================================================
#
# Invariants on output types:
#   - All fields are set at construction time and never mutated.
#   - PrefillEntry: tokens_this_step > 0, tokens_this_step <= prompt_tokens.
#   - DecodeEntry: context_length >= prompt_tokens + 1.
#   - ReconstructedStep: total_prefill_tokens == sum(e.tokens_this_step),
#     total_decode_tokens == len(decode_reqs), batch_size == len(prefill) + len(decode).
#   - RequestLabel: when failed=True, all timing/token fields are zero.
#     When failed=False, all timing fields are non-negative (warns otherwise).

@dataclass(frozen=True)
class PrefillEntry:
    """One prefill request's contribution to a single step."""
    request_id: str
    tokens_this_step: int   # prefill tokens allocated this step (greedy fill)
    prompt_tokens: int      # total prompt length for this request

@dataclass(frozen=True)
class DecodeEntry:
    """One decode request's contribution to a single step."""
    request_id: str
    context_length: int     # prompt_tokens + tokens_decoded_so_far

@dataclass(frozen=True)
class ReconstructedStep:
    """Batch composition at one scheduler step.

    Invariants:
        total_prefill_tokens == sum(e.tokens_this_step for e in prefill_reqs)
        total_decode_tokens == len(decode_reqs)
        batch_size == len(prefill_reqs) + len(decode_reqs)
    """
    step_id: int
    prefill_reqs: tuple[PrefillEntry, ...]
    decode_reqs: tuple[DecodeEntry, ...]
    total_prefill_tokens: int
    total_decode_tokens: int
    batch_size: int

@dataclass(frozen=True)
class RequestLabel:
    """Ground-truth timing labels for one request.

    Invariants:
        When failed=True: all timing and token fields are zero (incomplete lifecycle).
        When failed=False: timing fields are non-negative, prompt_tokens > 0.
        e2e_us >= ttft_us >= queueing_us (when failed=False).
        processing_us = e2e_us - queueing_us - preemption_gaps (approximately).
    """
    request_id: str
    prompt_tokens: int
    output_tokens: int
    queueing_us: float      # SCHEDULED.ts - QUEUED.ts (microseconds)
    ttft_us: float           # FIRST_TOKEN.ts - QUEUED.ts
    processing_us: float     # FINISHED.ts - SCHEDULED.ts - preemption gaps
    e2e_us: float            # FINISHED.ts - QUEUED.ts
    num_preemptions: int
    failed: bool
    first_step: int
    last_step: int

@dataclass(frozen=True)
class ExperimentReconstruction:
    """Complete reconstruction result for one experiment."""
    steps: tuple[ReconstructedStep, ...]
    labels: tuple[RequestLabel, ...]
    max_num_batched_tokens: int


# =============================================================================
# Internal types for reconstruction
# =============================================================================

class Phase(Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"

@dataclass
class Interval:
    """An interval during which a request is active in the batch.

    Invariants:
        start_step <= end_step (inclusive range).
        If phase == PREFILL: prefill_tokens > 0 (derived from trace data).
        If phase == DECODE: prefill_tokens == 0.
    """
    start_step: int
    end_step: int       # inclusive
    phase: Phase
    prefill_tokens: int = 0

@dataclass
class ParsedEvent:
    """A journey event with parsed attributes."""
    name: str           # "QUEUED", "SCHEDULED", "FIRST_TOKEN", "PREEMPTED", "FINISHED"
    step: int
    ts: float           # ts.monotonic (seconds)
    phase: str
    prefill_done: int
    prefill_total: int
    decode_done: int
    decode_max: int
    schedule_kind: str  # "FIRST", "RESUME", or ""
    finish_status: str  # "length", "stopped", "aborted", etc. or ""

@dataclass
class RequestTimeline:
    """Parsed journey timeline for one request.

    Lifecycle: created by _parse_events with events only, then enriched:
      1. _build_intervals populates intervals and decode_tokens_before.
      2. _compute_label populates label.

    The prefill_remaining field is NOT stored here — it is local state
    in _reconstruct_steps (avoids mixing algorithm scratch with data).
    """
    request_id: str
    events: list[ParsedEvent]
    prompt_tokens: int
    first_token_step: int
    first_scheduled_step: int
    intervals: list[Interval] = field(default_factory=list)
    # decode_tokens_before[iv_idx] = tokens decoded before this decode interval starts
    # (including the first token from the FIRST_TOKEN step)
    decode_tokens_before: dict[int, int] = field(default_factory=dict)
    label: RequestLabel | None = None


# =============================================================================
# Phase 1: Parse raw events into typed timelines
# =============================================================================
#
# Guarantees:
#   - Returns None for requests missing any of {QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED}.
#   - Events are sorted by (step, ts) to handle out-of-order OTEL exports.
#   - prompt_tokens, first_token_step, first_scheduled_step derived from first occurrences.

REQUIRED = {"QUEUED", "SCHEDULED", "FIRST_TOKEN", "FINISHED"}

def parse_events(request_id: str, raw_events: list[dict]) -> RequestTimeline | None:
    """Parse raw OTEL events into a RequestTimeline.

    Returns None if the request is missing required lifecycle events
    (incomplete / failed request).
    """
    parsed: list[ParsedEvent] = []
    for ev in raw_events:
        attrs = attr_map(ev.get("attributes", []))
        name = ev["name"].replace("journey.", "")
        parsed.append(ParsedEvent(
            name=name,
            step=int(attrs.get("scheduler.step", 0)),
            ts=attrs.get("ts.monotonic", 0.0),
            phase=attrs.get("phase", ""),
            prefill_done=int(attrs.get("prefill.done_tokens", 0)),
            prefill_total=int(attrs.get("prefill.total_tokens", 0)),
            decode_done=int(attrs.get("decode.done_tokens", 0)),
            decode_max=int(attrs.get("decode.max_tokens", 0)),
            schedule_kind=attrs.get("schedule.kind", ""),
            finish_status=attrs.get("finish.status", ""),
        ))

    event_names = {e.name for e in parsed}
    if not REQUIRED.issubset(event_names):
        return None

    # Sort by step then timestamp — handles out-of-order OTEL exports.
    parsed.sort(key=lambda e: (e.step, e.ts))

    first_sched = next(e for e in parsed if e.name == "SCHEDULED")
    first_ft = next(e for e in parsed if e.name == "FIRST_TOKEN")

    return RequestTimeline(
        request_id=request_id,
        events=parsed,
        prompt_tokens=first_sched.prefill_total,
        first_token_step=first_ft.step,
        first_scheduled_step=first_sched.step,
    )


# =============================================================================
# Phase 2: Build active intervals from event state machine
# =============================================================================
#
# Guarantees:
#   - Populates tl.intervals with non-overlapping Interval objects.
#   - PREFILL intervals have prefill_tokens derived from prefill.done_tokens.
#   - Handles preemption during both prefill (re-enters PREFILL on resume)
#     and decode (re-enters DECODE on resume).
#   - Populates tl.decode_tokens_before for correct context_length across gaps.

def _build_intervals(tl: RequestTimeline) -> None:
    """Walk the event sequence and populate tl.intervals.

    State machine:
      SCHEDULED(FIRST)@S  → PREFILL interval starts at step S
      FIRST_TOKEN@F       → prefill interval ends at F (inclusive),
                             decode interval starts at F+1
      PREEMPTED@P         → current interval ends at P-1 (request NOT active at P)
      SCHEDULED(RESUME)@R → interval starts at R; phase depends on whether
                             prefill was completed (FIRST_TOKEN already seen)
      FINISHED@D          → current interval ends at D (inclusive, request IS active at D)
    """
    intervals: list[Interval] = []
    current_phase: Phase | None = None
    current_start: int | None = None
    prefill_complete = False
    prefill_done_at_start = 0

    for ev in tl.events:
        if ev.name == "SCHEDULED" and ev.schedule_kind in ("FIRST", ""):
            current_phase = Phase.PREFILL
            current_start = ev.step
            prefill_done_at_start = ev.prefill_done

        elif ev.name == "FIRST_TOKEN":
            if current_start is not None:
                tokens = ev.prefill_done - prefill_done_at_start
                intervals.append(Interval(current_start, ev.step, Phase.PREFILL,
                                          prefill_tokens=tokens))
            else:
                warnings.warn(
                    f"Request {tl.request_id}: FIRST_TOKEN at step {ev.step} "
                    f"without preceding SCHEDULED. Prefill interval lost."
                )
            prefill_complete = True
            current_phase = Phase.DECODE
            current_start = ev.step + 1

        elif ev.name == "PREEMPTED":
            if current_start is not None and current_phase is not None and ev.step - 1 >= current_start:
                if current_phase == Phase.PREFILL:
                    tokens = ev.prefill_done - prefill_done_at_start
                    intervals.append(Interval(current_start, ev.step - 1, Phase.PREFILL,
                                              prefill_tokens=tokens))
                else:
                    intervals.append(Interval(current_start, ev.step - 1, Phase.DECODE))
            current_phase = None
            current_start = None

        elif ev.name == "SCHEDULED" and ev.schedule_kind == "RESUME":
            if prefill_complete:
                current_phase = Phase.DECODE
            else:
                current_phase = Phase.PREFILL
                prefill_done_at_start = ev.prefill_done
            current_start = ev.step

        elif ev.name == "FINISHED":
            if current_start is not None and current_phase is not None and ev.step >= current_start:
                if current_phase == Phase.PREFILL:
                    tokens = ev.prefill_done - prefill_done_at_start
                    intervals.append(Interval(current_start, ev.step, Phase.PREFILL,
                                              prefill_tokens=tokens))
                else:
                    intervals.append(Interval(current_start, ev.step, Phase.DECODE))
            current_phase = None
            current_start = None

    tl.intervals = intervals

    # Precompute cumulative decode tokens before each decode interval.
    cumulative = 1  # first token produced at FIRST_TOKEN step (during prefill)
    for iv_idx, iv in enumerate(tl.intervals):
        if iv.phase == Phase.DECODE:
            tl.decode_tokens_before[iv_idx] = cumulative
            cumulative += iv.end_step - iv.start_step + 1


# =============================================================================
# Phase 3: Compute per-request ground-truth labels
# =============================================================================
#
# Guarantees:
#   - All timing values in microseconds.
#   - processing_us excludes preemption gaps (paired PREEMPTED → RESUME deltas).
#   - Warns on negative timing values (indicates data corruption).

def _compute_label(tl: RequestTimeline) -> RequestLabel:
    """Compute ground-truth timing labels from journey timestamps."""
    ev_map: dict[str, list[ParsedEvent]] = {}
    for ev in tl.events:
        ev_map.setdefault(ev.name, []).append(ev)

    queued = ev_map["QUEUED"][0]
    scheduled = ev_map["SCHEDULED"][0]
    first_token = ev_map["FIRST_TOKEN"][0]
    finished = ev_map["FINISHED"][0]

    preemption_gap_s = 0.0
    preempted_events = ev_map.get("PREEMPTED", [])
    resume_events = [e for e in ev_map.get("SCHEDULED", []) if e.schedule_kind == "RESUME"]
    for pre, res in zip(preempted_events, resume_events):
        preemption_gap_s += res.ts - pre.ts

    queueing_s = scheduled.ts - queued.ts
    ttft_s = first_token.ts - queued.ts
    processing_s = (finished.ts - scheduled.ts) - preemption_gap_s
    e2e_s = finished.ts - queued.ts

    for name, val in [("queueing", queueing_s), ("ttft", ttft_s),
                       ("processing", processing_s), ("e2e", e2e_s)]:
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
        e2e_us=e2e_s * 1e6,
        num_preemptions=len(preempted_events),
        failed=False,
        first_step=scheduled.step,
        last_step=finished.step,
    )
    tl.label = label
    return label


# =============================================================================
# Phase 4: Reconstruct steps with greedy-fill prefill allocation
# =============================================================================
#
# Guarantees:
#   - Steps are in ascending step_id order with no gaps (except empty steps).
#   - For single-step exact-prefill: token count matches trace data exactly.
#   - For multi-step: greedy fill with max_num_batched_tokens budget.
#   - Decode context_length correctly accounts for preemption gaps via
#     decode_tokens_before (cumulative tokens before each decode interval).
#   - prefill_remaining is local algorithm state, NOT stored on timelines.

_START = 0
_END = 1


def _reconstruct_steps(
    timelines: list[RequestTimeline],
    max_num_batched_tokens: int,
) -> list[ReconstructedStep]:
    """Build per-step batch composition using greedy-fill for prefill tokens.

    Requires: _build_intervals has been called on every timeline
    (tl.intervals and tl.decode_tokens_before are populated).
    """
    if not timelines:
        return []

    # Exact prefill: single-step intervals with known token count from trace data.
    exact_prefill: dict[tuple[int, int], int] = {}
    for tl_idx, tl in enumerate(timelines):
        for iv_idx, iv in enumerate(tl.intervals):
            if iv.phase == Phase.PREFILL and iv.start_step == iv.end_step and iv.prefill_tokens > 0:
                exact_prefill[(tl_idx, iv_idx)] = iv.prefill_tokens

    # Sweep-line events: (step, type, phase, tl_idx, iv_idx)
    sweep_events: list[tuple[int, int, Phase, int, int]] = []
    min_step = float("inf")
    max_step = float("-inf")

    for tl_idx, tl in enumerate(timelines):
        for iv_idx, iv in enumerate(tl.intervals):
            sweep_events.append((iv.start_step, _START, iv.phase, tl_idx, iv_idx))
            sweep_events.append((iv.end_step + 1, _END, iv.phase, tl_idx, iv_idx))
            if iv.start_step < min_step:
                min_step = iv.start_step
            if iv.end_step > max_step:
                max_step = iv.end_step

    if min_step > max_step:
        return []

    sweep_events.sort(key=lambda e: (e[0], e[1]))  # type: ignore[return-value]

    # Algorithm-local prefill budget tracker (NOT on RequestTimeline)
    prefill_remaining: dict[int, int] = {
        tl_idx: tl.prompt_tokens for tl_idx, tl in enumerate(timelines)
    }

    active_prefill: dict[int, int] = {}   # tl_idx -> iv_idx
    active_decode: dict[int, int] = {}    # tl_idx -> iv_idx
    sweep_idx = 0
    n_sweep = len(sweep_events)

    steps: list[ReconstructedStep] = []

    for step_id in range(int(min_step), int(max_step) + 1):
        while sweep_idx < n_sweep and sweep_events[sweep_idx][0] == step_id:
            _, etype, phase, tl_idx, iv_idx = sweep_events[sweep_idx]
            if etype == _START:
                if phase == Phase.PREFILL:
                    active_prefill[tl_idx] = iv_idx
                else:
                    active_decode[tl_idx] = iv_idx
            else:
                if phase == Phase.PREFILL:
                    active_prefill.pop(tl_idx, None)
                else:
                    active_decode.pop(tl_idx, None)
            sweep_idx += 1

        if not active_prefill and not active_decode:
            continue

        # Decode: 1 token each, context_length accounts for preemption gaps
        decode_entries: list[DecodeEntry] = []
        for tl_idx, iv_idx in active_decode.items():
            tl = timelines[tl_idx]
            iv = tl.intervals[iv_idx]
            tokens_before = tl.decode_tokens_before.get(iv_idx, 1)
            context = tl.prompt_tokens + tokens_before + (step_id - iv.start_step)
            decode_entries.append(DecodeEntry(
                request_id=tl.request_id,
                context_length=context,
            ))

        total_decode = len(decode_entries)

        # Prefill: exact for single-step, greedy for multi-step
        budget = max(0, max_num_batched_tokens - total_decode)

        prefill_sorted = sorted(
            active_prefill.keys(),
            key=lambda idx: timelines[idx].first_scheduled_step,
        )

        prefill_entries: list[PrefillEntry] = []
        total_prefill = 0
        for tl_idx in prefill_sorted:
            tl = timelines[tl_idx]
            iv_idx = active_prefill[tl_idx]
            remaining = prefill_remaining[tl_idx]

            if remaining <= 0:
                continue

            exact_key = (tl_idx, iv_idx)
            if exact_key in exact_prefill:
                chunk = min(remaining, exact_prefill[exact_key])
            else:
                if budget <= 0:
                    continue
                chunk = min(remaining, budget)

            budget -= chunk
            prefill_remaining[tl_idx] -= chunk
            total_prefill += chunk
            prefill_entries.append(PrefillEntry(
                request_id=tl.request_id,
                tokens_this_step=chunk,
                prompt_tokens=tl.prompt_tokens,
            ))

        batch_size = len(prefill_entries) + total_decode

        steps.append(ReconstructedStep(
            step_id=step_id,
            prefill_reqs=tuple(prefill_entries),
            decode_reqs=tuple(decode_entries),
            total_prefill_tokens=total_prefill,
            total_decode_tokens=total_decode,
            batch_size=batch_size,
        ))

    return steps


# =============================================================================
# Public API
# =============================================================================

def reconstruct_timelines(
    timelines: list[RequestTimeline],
    max_num_batched_tokens: int,
) -> tuple[list[ReconstructedStep], list[RequestLabel]]:
    """Run the full reconstruction pipeline on pre-parsed timelines.

    This is the testable core of the module. Behavioral tests should target
    this function rather than internal functions like _build_intervals.

    Guarantees:
        - Calls _build_intervals then _compute_label on each timeline.
        - Returns (steps, labels) where steps cover every active scheduler step
          and labels have one entry per timeline.
    """
    for tl in timelines:
        _build_intervals(tl)

    labels: list[RequestLabel] = []
    for tl in timelines:
        labels.append(_compute_label(tl))

    steps = _reconstruct_steps(timelines, max_num_batched_tokens)
    return steps, labels


def reconstruct_experiment(exp: ExperimentMeta) -> ExperimentReconstruction:
    """Reconstruct step batches and request labels for one experiment.

    End-to-end: reads traces.json and exp-config.yaml, returns frozen result.
    """
    raw_events = parse_journey_events(traces_path_for(exp))
    config = load_exp_config(exp)
    max_num_batched_tokens = config["max_num_batched_tokens"]

    timelines: list[RequestTimeline] = []
    failed_labels: list[RequestLabel] = []

    for req_id, events in raw_events.items():
        tl = parse_events(req_id, events)
        if tl is None:
            failed_labels.append(RequestLabel(
                request_id=req_id, prompt_tokens=0, output_tokens=0,
                queueing_us=0.0, ttft_us=0.0, processing_us=0.0, e2e_us=0.0,
                num_preemptions=0, failed=True, first_step=0, last_step=0,
            ))
            continue
        timelines.append(tl)

    steps, labels = reconstruct_timelines(timelines, max_num_batched_tokens)

    return ExperimentReconstruction(
        steps=tuple(steps), labels=tuple(labels + failed_labels),
        max_num_batched_tokens=max_num_batched_tokens,
    )


# =============================================================================
# JSON output
# =============================================================================

def _write_experiment_json(
    exp: ExperimentMeta,
    result: ExperimentReconstruction,
) -> None:
    """Write per-experiment reconstruction to JSON (streaming, one step at a time)."""
    out_path = OUTPUT_DIR / f"{exp.dir_name}.json"
    sep = (",", ":")
    with open(out_path, "w") as f:
        header = {
            "experiment": exp.dir_name,
            "split": exp.split.value,
            "model_short": exp.model_short,
            "tensor_parallelism": exp.tensor_parallelism,
            "max_num_batched_tokens": result.max_num_batched_tokens,
            "num_steps": len(result.steps),
            "num_requests": len(result.labels),
            "num_failed": sum(1 for l in result.labels if l.failed),
        }
        f.write("{")
        for k, v in header.items():
            f.write(f"{json.dumps(k)}:{json.dumps(v)},")

        f.write('"steps":[')
        for i, s in enumerate(result.steps):
            if i > 0:
                f.write(",")
            json.dump(asdict(s), f, separators=sep)
        f.write('],')

        f.write('"requests":[')
        for i, label in enumerate(result.labels):
            if i > 0:
                f.write(",")
            json.dump(asdict(label), f, separators=sep)
        f.write(']')

        f.write("}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'Experiment':<58} {'Split':<6} "
        f"{'Steps':>7} {'Reqs':>7} {'Fail':>5} "
        f"{'AvgBatch':>8} {'PF tok':>10} {'DC tok':>10}"
    )
    print(header)
    print("─" * len(header))

    summary_rows: list[dict] = []

    for exp in EXPERIMENTS:
        result = reconstruct_experiment(exp)

        n_steps = len(result.steps)
        n_reqs = len(result.labels)
        n_failed = sum(1 for l in result.labels if l.failed)
        avg_batch = sum(s.batch_size for s in result.steps) / n_steps if n_steps else 0
        total_pf = sum(s.total_prefill_tokens for s in result.steps)
        total_dc = sum(s.total_decode_tokens for s in result.steps)

        print(
            f"{exp.dir_name:<58} {exp.split.value:<6} "
            f"{n_steps:>7,} {n_reqs:>7,} {n_failed:>5} "
            f"{avg_batch:>8.1f} {total_pf:>10,} {total_dc:>10,}"
        )

        _write_experiment_json(exp, result)

        summary_rows.append({
            "experiment": exp.dir_name,
            "split": exp.split.value,
            "model_short": exp.model_short,
            "num_steps": n_steps,
            "num_requests": n_reqs,
            "num_failed": n_failed,
            "avg_batch_size": round(avg_batch, 2),
            "total_prefill_tokens": total_pf,
            "total_decode_tokens": total_dc,
        })

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({"experiments": summary_rows}, f, indent=2)

    print(f"\nOutput written to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
