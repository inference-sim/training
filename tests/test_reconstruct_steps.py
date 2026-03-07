"""Behavioral tests for step reconstruction from journey traces.

Each test class describes a real scheduling scenario and verifies the
observable outputs: ReconstructedStep batch composition and RequestLabel
timing data.  Tests call reconstruct_timelines() — the public testable
API — and never inspect internal Interval objects.

Design discipline:
    Given: a set of request journeys (built via JourneyBuilder)
    When:  reconstruct_timelines() is called
    Then:  the returned steps and labels satisfy behavioral properties
"""

from __future__ import annotations

import pytest

from reconstruct_steps import (
    ParsedEvent,
    reconstruct_timelines,
    parse_events,
)
from tests.conftest import JourneyBuilder


def _step(steps, step_id):
    """Find a step by ID, raising clear error if missing."""
    return next(s for s in steps if s.step_id == step_id)


# ---------------------------------------------------------------------------
# Scenario 1: A single request with no preemption
# ---------------------------------------------------------------------------

class TestSingleRequestLifecycle:
    """A request that prefills in one step and decodes to completion.

    Expected behavior:
        - Step 1 has exactly 512 prefill tokens and 0 decode tokens.
        - Steps 2-100 each have 0 prefill and 1 decode token.
        - Context length grows by 1 each step.
        - All timing labels are positive.
    """

    @pytest.fixture()
    def result(self):
        tl = (
            JourneyBuilder("req-1", prompt_tokens=512, max_output_tokens=100)
            .queued(step=0, ts=1000.0)
            .scheduled(step=1, ts=1000.1)
            .first_token(step=1, ts=1000.2)
            .finished(step=100, ts=1007.0)
            .build()
        )
        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return steps, labels

    def test_prefill_step_has_all_prompt_tokens(self, result):
        steps, _ = result
        s1 = _step(steps, 1)
        assert s1.total_prefill_tokens == 512
        assert s1.total_decode_tokens == 0
        assert s1.batch_size == 1

    def test_first_decode_step_has_correct_context_length(self, result):
        steps, _ = result
        s2 = _step(steps, 2)
        assert s2.total_prefill_tokens == 0
        assert s2.total_decode_tokens == 1
        # context = prompt(512) + 1 token from FIRST_TOKEN step
        assert s2.decode_reqs[0].context_length == 512 + 1 + 0

    def test_produces_correct_number_of_steps(self, result):
        steps, _ = result
        # Step 1 (prefill) + steps 2-100 (decode) = 100 steps total
        assert len(steps) == 100

    def test_ground_truth_labels_are_positive(self, result):
        _, labels = result
        label = labels[0]
        assert label.queueing_us > 0
        assert label.ttft_us > 0
        assert label.processing_us > 0
        assert label.e2e_us > 0
        assert label.num_preemptions == 0
        assert label.failed is False


# ---------------------------------------------------------------------------
# Scenario 2: Preemption during decode
# ---------------------------------------------------------------------------

class TestDecodePreemption:
    """A request preempted mid-decode and later resumed.

    Key behavioral property: context_length after resume must NOT include
    the preemption gap — only steps where the request was actively decoding
    contribute tokens.
    """

    @pytest.fixture()
    def result(self):
        tl = (
            JourneyBuilder("req-preempt", prompt_tokens=100, max_output_tokens=200)
            .queued(step=0, ts=1000.0)
            .scheduled(step=10, ts=1001.0)
            .first_token(step=10, ts=1001.1)
            .preempted(step=60, ts=1006.0, decode_done=50)
            .scheduled(step=100, ts=1010.0, kind="RESUME")
            .finished(step=249, ts=1025.0, decode_done=200)
            .build()
        )
        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return steps, labels

    def test_request_absent_during_preemption_gap(self, result):
        steps, _ = result
        # Steps 60-99 should have no decode entries (request is preempted)
        for s in steps:
            if 60 <= s.step_id <= 99:
                assert s.batch_size == 0 or all(
                    d.request_id != "req-preempt" for d in s.decode_reqs
                ), f"Request should not be active at step {s.step_id}"

    def test_context_length_at_resume_accounts_for_gap(self, result):
        steps, _ = result
        # Before preemption: step 59 has context = 100 + 1 + (59 - 11) = 149
        s59 = _step(steps, 59)
        assert s59.decode_reqs[0].context_length == 149

        # At resume: step 100 has context = 100 + 50 + 0 = 150
        # (50 tokens decoded before gap, NOT 100 + (100-10) = 190)
        s100 = _step(steps, 100)
        assert s100.decode_reqs[0].context_length == 150
        # Must be less than the naive formula would give
        assert s100.decode_reqs[0].context_length < 100 + (100 - 10)

    def test_processing_time_excludes_preemption_gap(self, result):
        _, labels = result
        label = labels[0]
        gap_us = (1010.0 - 1006.0) * 1e6
        raw_processing = (1025.0 - 1001.0) * 1e6
        assert abs(label.processing_us - (raw_processing - gap_us)) < 1.0
        assert label.num_preemptions == 1


# ---------------------------------------------------------------------------
# Scenario 3: Preemption during prefill
# ---------------------------------------------------------------------------

class TestPrefillPreemption:
    """A request preempted mid-prefill must re-prefill after resume.

    Key behavioral property: total prefill tokens across all steps must
    equal the full prompt length, with token counts derived from
    prefill.done_tokens at preemption boundaries.
    """

    @pytest.fixture()
    def result(self):
        tl = (
            JourneyBuilder("req-pf-preempt", prompt_tokens=1000, max_output_tokens=50)
            .queued(step=0, ts=1000.0)
            .scheduled(step=10, ts=1001.0)
            .preempted(step=11, ts=1001.5, prefill_done=600)
            .scheduled(step=20, ts=1002.0, kind="RESUME")
            .first_token(step=20, ts=1002.3)
            .finished(step=69, ts=1005.0, decode_done=50)
            .build()
        )
        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return steps, labels

    def test_total_prefill_tokens_equals_prompt(self, result):
        steps, _ = result
        total = sum(s.total_prefill_tokens for s in steps)
        assert total == 1000

    def test_first_prefill_step_has_partial_tokens(self, result):
        steps, _ = result
        s10 = _step(steps, 10)
        assert s10.total_prefill_tokens == 600

    def test_second_prefill_step_has_remaining_tokens(self, result):
        steps, _ = result
        s20 = _step(steps, 20)
        assert s20.total_prefill_tokens == 400

    def test_decode_starts_after_second_prefill(self, result):
        steps, _ = result
        s21 = _step(steps, 21)
        assert s21.total_decode_tokens == 1
        assert s21.total_prefill_tokens == 0


# ---------------------------------------------------------------------------
# Scenario 4: Double preemption (decode → preempt → resume → preempt → resume)
# ---------------------------------------------------------------------------

class TestDoublePreemption:
    """A request preempted twice during decode.

    Key behavioral property: context_length must account for BOTH gaps.
    Each resume picks up from where the last active decode left off.
    """

    @pytest.fixture()
    def result(self):
        tl = (
            JourneyBuilder("req-double", prompt_tokens=100, max_output_tokens=300)
            .queued(step=0, ts=1000.0)
            .scheduled(step=10, ts=1001.0)
            .first_token(step=10, ts=1001.1)
            # Decode 40 tokens (steps 11-50), then preempted
            .preempted(step=51, ts=1005.0, decode_done=41)
            # Gap: steps 51-99
            .scheduled(step=100, ts=1010.0, kind="RESUME")
            # Decode 20 more tokens (steps 100-119), then preempted again
            .preempted(step=120, ts=1012.0, decode_done=61)
            # Gap: steps 120-199
            .scheduled(step=200, ts=1020.0, kind="RESUME")
            .finished(step=439, ts=1044.0, decode_done=300)
            .build()
        )
        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return steps, labels

    def test_context_length_after_second_resume(self, result):
        steps, _ = result
        # At step 200 (second resume): tokens decoded = 1 + 40 + 20 = 61
        s200 = _step(steps, 200)
        assert s200.decode_reqs[0].context_length == 100 + 61

    def test_request_absent_during_both_gaps(self, result):
        steps, _ = result
        for s in steps:
            if (51 <= s.step_id <= 99) or (120 <= s.step_id <= 199):
                assert s.batch_size == 0, f"Request active at gap step {s.step_id}"

    def test_two_preemptions_recorded_in_label(self, result):
        _, labels = result
        assert labels[0].num_preemptions == 2

    def test_processing_time_excludes_both_gaps(self, result):
        _, labels = result
        gap1_us = (1010.0 - 1005.0) * 1e6
        gap2_us = (1020.0 - 1012.0) * 1e6
        raw_us = (1044.0 - 1001.0) * 1e6
        expected = raw_us - gap1_us - gap2_us
        assert abs(labels[0].processing_us - expected) < 1.0


# ---------------------------------------------------------------------------
# Scenario 5: Chunked prefill (multi-step)
# ---------------------------------------------------------------------------

class TestChunkedPrefill:
    """A large prompt spanning multiple steps due to budget constraints.

    With 20 concurrent decode requests and max_num_batched_tokens=2048,
    only 2028 tokens of budget remain. A 3000-token prompt needs 2 steps.
    """

    @pytest.fixture()
    def result(self):
        tls = []
        for i in range(20):
            tls.append(
                JourneyBuilder(f"decode-{i}", prompt_tokens=100, max_output_tokens=500)
                .queued(step=0).scheduled(step=1).first_token(step=1)
                .finished(step=500).build()
            )
        tls.append(
            JourneyBuilder("big-prefill", prompt_tokens=3000, max_output_tokens=10)
            .queued(step=49).scheduled(step=50).first_token(step=51)
            .finished(step=61, decode_done=10).build()
        )
        steps, labels = reconstruct_timelines(tls, max_num_batched_tokens=2048)
        return steps, labels

    def test_greedy_fill_respects_budget(self, result):
        steps, _ = result
        s50 = _step(steps, 50)
        assert s50.total_decode_tokens == 20
        pf = next(e for e in s50.prefill_reqs if e.request_id == "big-prefill")
        assert pf.tokens_this_step <= 2048 - 20

    def test_all_prefill_tokens_allocated_across_steps(self, result):
        steps, _ = result
        total = sum(
            e.tokens_this_step
            for s in steps for e in s.prefill_reqs
            if e.request_id == "big-prefill"
        )
        assert total == 3000


# ---------------------------------------------------------------------------
# Scenario 6: Single-token output (FIRST_TOKEN and FINISHED at same step)
# ---------------------------------------------------------------------------

class TestSingleTokenOutput:
    """A request producing exactly one output token.

    FIRST_TOKEN and FINISHED at the same step means no decode interval —
    the request appears only in the prefill step.
    """

    @pytest.fixture()
    def result(self):
        tl = (
            JourneyBuilder("req-single-tok", prompt_tokens=256, max_output_tokens=1)
            .queued(step=0, ts=1000.0)
            .scheduled(step=5, ts=1001.0)
            .first_token(step=5, ts=1001.1)
            .finished(step=5, ts=1001.2, decode_done=1)
            .build()
        )
        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)
        return steps, labels

    def test_produces_exactly_one_step(self, result):
        steps, _ = result
        assert len(steps) == 1
        assert steps[0].step_id == 5

    def test_step_has_prefill_only(self, result):
        steps, _ = result
        assert steps[0].total_prefill_tokens == 256
        assert steps[0].total_decode_tokens == 0

    def test_label_records_one_output_token(self, result):
        _, labels = result
        assert labels[0].output_tokens == 1
        assert labels[0].failed is False


# ---------------------------------------------------------------------------
# Scenario 7: Multiple concurrent requests at the same step
# ---------------------------------------------------------------------------

class TestConcurrentBatch:
    """Three requests overlapping at step 10: one prefilling, two decoding.

    Key behavioral property: batch composition aggregates correctly, and
    each decode request's context length reflects its own progress.
    """

    @pytest.fixture()
    def result(self):
        a = (
            JourneyBuilder("req-a", prompt_tokens=500, max_output_tokens=50)
            .queued(step=8).scheduled(step=10).first_token(step=10)
            .finished(step=59, decode_done=50).build()
        )
        b = (
            JourneyBuilder("req-b", prompt_tokens=200, max_output_tokens=100)
            .queued(step=0).scheduled(step=1).first_token(step=1)
            .finished(step=100, decode_done=100).build()
        )
        c = (
            JourneyBuilder("req-c", prompt_tokens=300, max_output_tokens=80)
            .queued(step=2).scheduled(step=3).first_token(step=3)
            .finished(step=82, decode_done=80).build()
        )
        steps, labels = reconstruct_timelines([a, b, c], max_num_batched_tokens=2048)
        return steps, labels

    def test_step_10_has_one_prefill_and_two_decode(self, result):
        steps, _ = result
        s10 = _step(steps, 10)
        assert len(s10.prefill_reqs) == 1
        assert s10.prefill_reqs[0].tokens_this_step == 500
        assert len(s10.decode_reqs) == 2
        assert s10.batch_size == 3

    def test_each_decode_context_reflects_own_progress(self, result):
        steps, _ = result
        s10 = _step(steps, 10)
        ctx = {d.request_id: d.context_length for d in s10.decode_reqs}
        # req-b: started decode at step 2, at step 10 has decoded 9 tokens
        # context = 200 + 1 + (10 - 2) = 209
        assert ctx["req-b"] == 200 + 1 + (10 - 2)
        # req-c: started decode at step 4, at step 10 has decoded 7 tokens
        # context = 300 + 1 + (10 - 4) = 307
        assert ctx["req-c"] == 300 + 1 + (10 - 4)


# ---------------------------------------------------------------------------
# Scenario 8: Out-of-order events are handled correctly
# ---------------------------------------------------------------------------

class TestOutOfOrderEvents:
    """Events arriving out of step order from batched OTEL exports.

    PREEMPTED@step 20 appears before FIRST_TOKEN@step 19 in the raw
    event list.  The pipeline must sort by step before processing,
    otherwise the prefill interval is lost.
    """

    def test_out_of_order_events_produce_correct_reconstruction(self):
        builder = JourneyBuilder("req-ooo", prompt_tokens=500, max_output_tokens=100)
        builder.queued(step=0, ts=1.0)
        builder.scheduled(step=10, ts=2.0)
        # Insert events in WRONG order (simulating batched OTEL export)
        builder.events.append(ParsedEvent(
            name="PREEMPTED", step=20, ts=3.5,
            phase="DECODE", prefill_done=500, prefill_total=500,
            decode_done=10, decode_max=100, schedule_kind="", finish_status="",
        ))
        builder.events.append(ParsedEvent(
            name="FIRST_TOKEN", step=19, ts=3.0,
            phase="DECODE", prefill_done=500, prefill_total=500,
            decode_done=1, decode_max=100, schedule_kind="", finish_status="",
        ))
        builder.scheduled(step=30, ts=5.0, kind="RESUME")
        builder._decode_done = 10
        builder.finished(step=119, ts=15.0, decode_done=100)
        tl = builder.build()

        steps, labels = reconstruct_timelines([tl], max_num_batched_tokens=2048)

        # Prefill tokens should be present (would be lost without sort)
        total_pf = sum(s.total_prefill_tokens for s in steps)
        assert total_pf == 500

        # Request should decode from step 30 onwards (after resume)
        s30 = _step(steps, 30)
        assert s30.total_decode_tokens == 1

        # Label should be valid
        assert labels[0].prompt_tokens == 500
        assert labels[0].failed is False


# ---------------------------------------------------------------------------
# Scenario 9: Incomplete request is excluded from reconstruction
# ---------------------------------------------------------------------------

class TestIncompleteRequest:
    """A request missing FINISHED is excluded from step reconstruction
    but recorded as a failed label by the pipeline."""

    def test_missing_finished_returns_none_from_parse(self):
        raw = [
            _make_raw_event("journey.QUEUED", step=0, ts=1.0,
                            prefill_total=100, decode_max=50),
            _make_raw_event("journey.SCHEDULED", step=1, ts=2.0, kind="FIRST",
                            prefill_total=100, decode_max=50),
            _make_raw_event("journey.FIRST_TOKEN", step=1, ts=2.5,
                            prefill_done=100, prefill_total=100,
                            decode_done=1, decode_max=50),
        ]
        result = parse_events("req-incomplete", raw)
        assert result is None

    def test_empty_timelines_produce_empty_steps(self):
        steps, labels = reconstruct_timelines([], max_num_batched_tokens=2048)
        assert steps == []
        assert labels == []


# ---------------------------------------------------------------------------
# Helpers for raw OTEL event construction
# ---------------------------------------------------------------------------

def _make_raw_event(
    name: str, step: int, ts: float,
    prefill_done: int = 0, prefill_total: int = 0,
    decode_done: int = 0, decode_max: int = 0,
    kind: str = "", status: str = "", phase: str = "",
) -> dict:
    """Build a raw OTEL event dict matching the traces.json wire format."""
    attrs = [
        {"key": "event.type", "value": {"stringValue": name.replace("journey.", "")}},
        {"key": "ts.monotonic", "value": {"doubleValue": ts}},
        {"key": "ts.monotonic_ns", "value": {"intValue": str(int(ts * 1e9))}},
        {"key": "scheduler.step", "value": {"intValue": str(step)}},
        {"key": "phase", "value": {"stringValue": phase or ("WAITING" if "QUEUED" in name else "DECODE")}},
        {"key": "prefill.done_tokens", "value": {"intValue": str(prefill_done)}},
        {"key": "prefill.total_tokens", "value": {"intValue": str(prefill_total)}},
        {"key": "decode.done_tokens", "value": {"intValue": str(decode_done)}},
        {"key": "decode.max_tokens", "value": {"intValue": str(decode_max)}},
        {"key": "num_preemptions", "value": {"intValue": "0"}},
    ]
    if kind:
        attrs.append({"key": "schedule.kind", "value": {"stringValue": kind}})
    if status:
        attrs.append({"key": "finish.status", "value": {"stringValue": status}})
    return {"name": name, "timeUnixNano": str(int(ts * 1e9)), "attributes": attrs}
