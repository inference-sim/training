"""Shared test fixtures for building synthetic journey trace data.

Provides a fluent builder for constructing request journeys from simple
parameters, avoiding the deeply nested OTEL wire format in every test.

Design discipline:
    - JourneyBuilder.build() returns a raw RequestTimeline (events only).
    - Tests call reconstruct_timelines() to run the full pipeline.
    - Tests assert ONLY on ReconstructedStep and RequestLabel output.
    - Tests never inspect internal Interval objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reconstruct_steps import ParsedEvent, RequestTimeline


@dataclass
class JourneyBuilder:
    """Fluent builder for a single request's journey events.

    Usage::

        journey = (
            JourneyBuilder("req-1", prompt_tokens=512, max_output_tokens=100)
            .queued(step=0)
            .scheduled(step=5)
            .first_token(step=5)
            .finished(step=104)
            .build()
        )
    """

    request_id: str
    prompt_tokens: int = 512
    max_output_tokens: int = 100
    events: list[ParsedEvent] = field(default_factory=list)
    _ts: float = field(default=1000.0, repr=False)
    _prefill_done: int = field(default=0, repr=False)
    _decode_done: int = field(default=0, repr=False)

    def _next_ts(self, ts: float | None) -> float:
        if ts is not None:
            self._ts = ts
            return ts
        self._ts += 0.001
        return self._ts

    def queued(self, step: int = 0, ts: float | None = None) -> JourneyBuilder:
        self.events.append(ParsedEvent(
            name="QUEUED", step=step, ts=self._next_ts(ts),
            phase="WAITING",
            prefill_done=0, prefill_total=self.prompt_tokens,
            decode_done=0, decode_max=self.max_output_tokens,
            schedule_kind="", finish_status="",
        ))
        return self

    def scheduled(
        self, step: int, ts: float | None = None, kind: str = "FIRST",
    ) -> JourneyBuilder:
        phase = "PREFILL" if kind == "FIRST" else "DECODE"
        if kind == "RESUME" and self._prefill_done < self.prompt_tokens:
            phase = "PREFILL"
        self.events.append(ParsedEvent(
            name="SCHEDULED", step=step, ts=self._next_ts(ts),
            phase=phase,
            prefill_done=self._prefill_done, prefill_total=self.prompt_tokens,
            decode_done=self._decode_done, decode_max=self.max_output_tokens,
            schedule_kind=kind, finish_status="",
        ))
        return self

    def first_token(self, step: int, ts: float | None = None) -> JourneyBuilder:
        self._prefill_done = self.prompt_tokens
        self._decode_done = 1
        self.events.append(ParsedEvent(
            name="FIRST_TOKEN", step=step, ts=self._next_ts(ts),
            phase="DECODE",
            prefill_done=self.prompt_tokens, prefill_total=self.prompt_tokens,
            decode_done=1, decode_max=self.max_output_tokens,
            schedule_kind="", finish_status="",
        ))
        return self

    def preempted(
        self, step: int, ts: float | None = None,
        prefill_done: int | None = None, decode_done: int | None = None,
    ) -> JourneyBuilder:
        if prefill_done is not None:
            self._prefill_done = prefill_done
        if decode_done is not None:
            self._decode_done = decode_done
        phase = "DECODE" if self._prefill_done >= self.prompt_tokens else "PREFILL"
        self.events.append(ParsedEvent(
            name="PREEMPTED", step=step, ts=self._next_ts(ts),
            phase=phase,
            prefill_done=self._prefill_done, prefill_total=self.prompt_tokens,
            decode_done=self._decode_done, decode_max=self.max_output_tokens,
            schedule_kind="", finish_status="",
        ))
        return self

    def finished(
        self, step: int, ts: float | None = None,
        decode_done: int | None = None, status: str = "length",
    ) -> JourneyBuilder:
        if decode_done is not None:
            self._decode_done = decode_done
        else:
            self._decode_done = self.max_output_tokens
        self.events.append(ParsedEvent(
            name="FINISHED", step=step, ts=self._next_ts(ts),
            phase="DECODE",
            prefill_done=self.prompt_tokens, prefill_total=self.prompt_tokens,
            decode_done=self._decode_done, decode_max=self.max_output_tokens,
            schedule_kind="", finish_status=status,
        ))
        return self

    def build(self) -> RequestTimeline:
        """Build a raw RequestTimeline (events only, no intervals/labels).

        The returned timeline has NOT been processed by _build_intervals or
        _compute_label. Tests should pass it to reconstruct_timelines() to
        run the full pipeline and then assert on the output.
        """
        first_sched = next(e for e in self.events if e.name == "SCHEDULED")
        first_ft = next(e for e in self.events if e.name == "FIRST_TOKEN")
        return RequestTimeline(
            request_id=self.request_id,
            events=sorted(self.events, key=lambda e: (e.step, e.ts)),
            prompt_tokens=self.prompt_tokens,
            first_token_step=first_ft.step,
            first_scheduled_step=first_sched.step,
        )
