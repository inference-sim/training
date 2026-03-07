"""Shared OTEL trace parsing utilities.

Provides common functions for reading traces.json (JSONL, OTLP wire format)
and extracting journey events from llm_core spans.  Used by validate_traces.py
and reconstruct_steps.py.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from split import ExperimentMeta


def attr_map(attributes: list[dict]) -> dict[str, Any]:
    """Convert OTEL attribute list to ``{key: python_value}`` dict."""
    out: dict[str, Any] = {}
    for attr in attributes:
        key = attr["key"]
        val = attr["value"]
        if "doubleValue" in val:
            out[key] = val["doubleValue"]
        elif "intValue" in val:
            out[key] = int(val["intValue"])
        elif "stringValue" in val:
            out[key] = val["stringValue"]
        elif "boolValue" in val:
            out[key] = val["boolValue"]
        else:
            warnings.warn(
                f"attr_map: unrecognized value type for key {key!r}: "
                f"{list(val.keys())}. Attribute dropped.",
                stacklevel=2,
            )
    return out


def parse_journey_events(traces_path: Path | str) -> dict[str, list[dict]]:
    """Parse traces.json and return journey events grouped by request ID.

    Reads all ``llm_core`` spans from the ``vllm.scheduler`` scope and
    collects their events (journey.QUEUED, journey.SCHEDULED, etc.) keyed
    by ``gen_ai.request.id``.

    Events from multiple spans for the same request (e.g. after preemption)
    are concatenated in trace-file order.

    Returns:
        ``{request_id: [raw_event_dict, ...]}`` preserving trace-file order.
    """
    requests: dict[str, list[dict]] = defaultdict(list)

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
                    if ss.get("scope", {}).get("name") != "vllm.scheduler":
                        continue
                    for span in ss.get("spans", []):
                        if span["name"] != "llm_core":
                            continue
                        request_id = None
                        for a in span.get("attributes", []):
                            if a["key"] == "gen_ai.request.id":
                                request_id = a["value"].get("stringValue")
                                break
                        if request_id is None:
                            continue
                        requests[request_id].extend(span.get("events", []))

    return dict(requests)


def load_exp_config(exp: ExperimentMeta) -> dict[str, Any]:
    """Load an experiment's exp-config.yaml as a plain dict.

    Returns keys: model, tensor_parallelism, max_model_len,
    max_num_batched_tokens, max_num_seqs, app.
    """
    import yaml  # lazy import — only needed by reconstruct_steps

    config_path = Path("default_args") / exp.dir_name / "exp-config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def traces_path_for(exp: ExperimentMeta) -> Path:
    """Return the path to an experiment's traces.json."""
    return Path("default_args") / exp.dir_name / "traces.json"
