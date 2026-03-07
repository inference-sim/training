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
