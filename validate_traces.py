"""Validate journey trace data for all 16 experiments.

Parses each experiment's traces.json (JSONL, OTEL wire format) and validates
every request's lifecycle against 5 correctness checks:

    1. Lifecycle completeness  — all 4 required events present
    2. Timestamp ordering      — monotonic timestamps are non-decreasing
    3. Step index ordering     — scheduler steps follow expected progression
    4. Preemption pairing      — every PREEMPTED has a matching RESUME
    5. Single-step prefill     — prefill completes within 1 scheduler step

Exit code 1 if any experiment has >1% failure rate across all checks.
Writes per-experiment JSON to output/validate/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from split import EXPERIMENTS, ExperimentMeta
from trace_parser import attr_map, parse_journey_events, traces_path_for

OUTPUT_DIR = Path("output/validate")

REQUIRED_EVENTS = ("journey.QUEUED", "journey.SCHEDULED", "journey.FIRST_TOKEN", "journey.FINISHED")


# ---------------------------------------------------------------------------
# Per-request validation result
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: str
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def validate_request(request_id: str, events: list[dict]) -> RequestResult:
    """Run all 5 checks on a single request's journey events."""
    result = RequestResult(request_id=request_id)

    # Build list of (event_name, attrs) in trace order
    parsed = []
    for ev in events:
        name = ev["name"]
        attrs = attr_map(ev.get("attributes", []))
        parsed.append((name, attrs))

    event_names = [name for name, _ in parsed]

    # --- Check 1: Lifecycle completeness ---
    for req_event in REQUIRED_EVENTS:
        if req_event not in event_names:
            result.failures.append(f"missing {req_event}")

    if not result.ok:
        # Can't run remaining checks without all 4 events
        return result

    # Extract the FIRST occurrence of each required event
    required: dict[str, dict] = {}
    for name, attrs in parsed:
        short = name.replace("journey.", "")
        if name in REQUIRED_EVENTS and short not in required:
            required[short] = attrs

    ts_q = required["QUEUED"]["ts.monotonic"]
    ts_s = required["SCHEDULED"]["ts.monotonic"]
    ts_f = required["FIRST_TOKEN"]["ts.monotonic"]
    ts_d = required["FINISHED"]["ts.monotonic"]

    step_s = required["SCHEDULED"]["scheduler.step"]
    step_f = required["FIRST_TOKEN"]["scheduler.step"]
    step_d = required["FINISHED"]["scheduler.step"]

    # --- Check 2: Timestamp ordering ---
    if not (ts_q <= ts_s <= ts_f <= ts_d):
        result.failures.append(
            f"timestamp order: Q={ts_q:.6f} S={ts_s:.6f} F={ts_f:.6f} D={ts_d:.6f}"
        )

    # --- Check 3: Step index ordering ---
    # Use <= for FIRST_TOKEN vs FINISHED: a single-token decode (output_tokens=1)
    # can have FIRST_TOKEN and FINISHED in the same scheduler step.
    if not (step_s <= step_f <= step_d):
        result.failures.append(
            f"step order: S={step_s} F={step_f} D={step_d} (expected S<=F<=D)"
        )

    # --- Check 4: Preemption pairing ---
    # Walk events in order; every PREEMPTED must be followed by SCHEDULED(kind=RESUME)
    preempt_pending = False
    for name, attrs in parsed:
        if name == "journey.PREEMPTED":
            if preempt_pending:
                result.failures.append("consecutive PREEMPTED without RESUME")
            preempt_pending = True
        elif name == "journey.SCHEDULED":
            kind = attrs.get("schedule.kind", "")
            if preempt_pending:
                if kind != "RESUME":
                    result.failures.append(
                        f"PREEMPTED followed by SCHEDULED(kind={kind!r}), expected RESUME"
                    )
                preempt_pending = False
    if preempt_pending:
        result.failures.append("trailing PREEMPTED without RESUME")

    # --- Check 5: Single-step prefill ---
    # Use the last SCHEDULED before FIRST_TOKEN (handles preemption before prefill)
    last_sched_step = step_s
    for name, attrs in parsed:
        if name == "journey.SCHEDULED":
            s = attrs["scheduler.step"]
            if s <= step_f:
                last_sched_step = s
        elif name == "journey.FIRST_TOKEN":
            break
    prefill_steps = step_f - last_sched_step
    if prefill_steps > 1:
        result.failures.append(
            f"multi-step prefill: FIRST_TOKEN.step - SCHEDULED.step = {prefill_steps}"
        )

    return result


# ---------------------------------------------------------------------------
# Experiment-level parsing and validation
# ---------------------------------------------------------------------------

def validate_experiment(exp: ExperimentMeta) -> tuple[int, list[RequestResult]]:
    """Validate all requests in one experiment. Returns (total_requests, failures)."""
    requests = parse_journey_events(traces_path_for(exp))

    failures: list[RequestResult] = []
    for req_id, events in requests.items():
        result = validate_request(req_id, events)
        if not result.ok:
            failures.append(result)

    return len(requests), failures


def _write_experiment_json(
    exp: ExperimentMeta, total: int, failures: list[RequestResult],
) -> None:
    """Write per-experiment validation result as JSON."""
    record = {
        "experiment": exp.dir_name,
        "split": exp.split.value,
        "total_requests": total,
        "expected_requests": exp.num_total,
        "failed_requests": len(failures),
        "failure_rate": len(failures) / total if total > 0 else 0.0,
        "failures": [
            {"request_id": r.request_id, "checks": r.failures}
            for r in failures
        ],
    }
    out_path = OUTPUT_DIR / f"{exp.dir_name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


# ---------------------------------------------------------------------------
# Main: run across all 16 experiments
# ---------------------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    any_over_threshold = False
    total_requests_all = 0
    total_failures_all = 0
    summary_rows: list[dict] = []

    print(f"{'Experiment':<58} {'Split':<10} {'Requests':>8} {'Expected':>8} {'Fail':>6} {'Rate':>7}")
    print("─" * 103)

    for exp in EXPERIMENTS:
        total, failures = validate_experiment(exp)
        expected = exp.num_total
        n_fail = len(failures)
        rate = n_fail / total if total > 0 else 0.0

        total_requests_all += total
        total_failures_all += n_fail

        count_flag = "" if total == expected else f"  ⚠ expected {expected}"

        print(
            f"{exp.dir_name:<58} {exp.split.value:<10} {total:>8,} {expected:>8,} "
            f"{n_fail:>6} {rate:>6.2%}{count_flag}"
        )

        if rate > 0.01:
            any_over_threshold = True

        if failures:
            shown = failures[:10]
            for r in shown:
                for f_msg in r.failures:
                    print(f"    {r.request_id}: {f_msg}")
            if len(failures) > 10:
                print(f"    ... and {len(failures) - 10} more failed requests")

        _write_experiment_json(exp, total, failures)

        summary_rows.append({
            "experiment": exp.dir_name,
            "split": exp.split.value,
            "total_requests": total,
            "expected_requests": expected,
            "failed_requests": n_fail,
            "failure_rate": rate,
        })

    print("─" * 103)
    print(
        f"{'TOTAL':<58} {'':<10} {total_requests_all:>8,} "
        f"{sum(e.num_total for e in EXPERIMENTS):>8,} "
        f"{total_failures_all:>6} "
        f"{total_failures_all / total_requests_all if total_requests_all else 0:>6.2%}"
    )

    # Write summary JSON
    summary = {
        "total_requests": total_requests_all,
        "total_expected": sum(e.num_total for e in EXPERIMENTS),
        "total_failures": total_failures_all,
        "overall_failure_rate": total_failures_all / total_requests_all if total_requests_all else 0.0,
        "passed": not any_over_threshold,
        "experiments": summary_rows,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if any_over_threshold:
        print("\nFAILED: One or more experiments exceed 1% failure rate.")
        return 1
    else:
        print("\nPASSED: All experiments within tolerance.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
