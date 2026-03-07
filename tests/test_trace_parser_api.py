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
