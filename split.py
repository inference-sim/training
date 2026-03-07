"""
Experiment metadata and request-level data splitting for the crossmodel
latency model fitting pipeline.

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR DATA PARTITIONING.
All training, validation, and evaluation code MUST import splits from here.
Never hard-code experiment lists elsewhere.

Split Design
============

Split unit: individual requests, not experiments. Since the fitting pipeline
uses teacher-forced reconstruction (real batch compositions, not simulated),
per-request features are independent — request N's feature vector does not
depend on request M's split assignment. This eliminates the regime bias of
experiment-level splitting, where an entire workload profile (reasoning) is
absent from training.

Assignment uses a deterministic hash of the request ID (SHA-256 mod 100):
    70% train (0-69), 15% validate (70-84), 15% test (85-99).

Overload Exclusion
==================

3 of the 16 experiments are in the overload regime (>10% failure rate) and
are excluded from the active dataset. The linear step-time model cannot
capture nonlinear preemption cascade dynamics.

Active dataset: 13 experiments, ~130K successful requests.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True)
class ExperimentMeta:
    """Immutable metadata for one experiment.

    All fields are derived from the experiment data at split-definition time
    and frozen here so downstream code never needs to re-parse metadata.
    """

    dir_name: str
    """Experiment directory name (e.g., '20260218-150304-codellama-34b-tp2-general')."""

    model_id: str
    """HuggingFace model identifier (e.g., 'codellama/CodeLlama-34b-Instruct-hf')."""

    model_short: str
    """Short model name for display (e.g., 'codellama-34b')."""

    profile: str
    """Workload profile: 'general', 'codegen', 'roleplay', or 'reasoning'."""

    tensor_parallelism: int
    """Tensor parallelism degree."""

    rates: tuple[float, ...]
    """Requested arrival rates per stage (e.g., (8.0, 20.0) for 2-stage)."""

    num_stages: int
    """Number of load stages (1 or 2)."""

    num_success: int
    """Number of successful requests."""

    num_failed: int
    """Number of failed requests (timeouts, errors)."""

    num_sampled_steps: int
    """Number of sampled step.BATCH_SUMMARY events in traces.json."""

    config_json_dir: str
    """Relative path to model config.json directory under training/model_configs/."""

    @property
    def num_total(self) -> int:
        """Total requests (success + failed)."""
        return self.num_success + self.num_failed

    @property
    def failure_rate(self) -> float:
        """Fraction of requests that failed."""
        return self.num_failed / self.num_total if self.num_total > 0 else 0.0


# =============================================================================
# EXPERIMENT DEFINITIONS — the single source of truth
# =============================================================================
#
# IMPORTANT: If you need to change the experiment set, change ONLY this section.
# All downstream code reads from EXPERIMENTS (active) or EXCLUDED_OVERLOAD.
# =============================================================================

EXPERIMENTS: tuple[ExperimentMeta, ...] = (
    # --- llama-2-7b (TP=1): general, codegen, roleplay ---
    ExperimentMeta(
        dir_name="20260217-231439-llama-2-7b-tp1-general",
        model_id="meta-llama/Llama-2-7b-hf",
        model_short="llama-2-7b",
        profile="general",
        tensor_parallelism=1,
        rates=(8.0, 20.0),
        num_stages=2,
        num_success=16800,
        num_failed=0,
        num_sampled_steps=11264,
        config_json_dir="Llama-2-7b-hf",
    ),
    ExperimentMeta(
        dir_name="20260217-155451-llama-2-7b-tp1-codegen",
        model_id="meta-llama/Llama-2-7b-hf",
        model_short="llama-2-7b",
        profile="codegen",
        tensor_parallelism=1,
        rates=(5.0, 10.0),
        num_stages=2,
        num_success=9000,
        num_failed=0,
        num_sampled_steps=15588,
        config_json_dir="Llama-2-7b-hf",
    ),
    ExperimentMeta(
        dir_name="20260217-162547-llama-2-7b-tp1-roleplay",
        model_id="meta-llama/Llama-2-7b-hf",
        model_short="llama-2-7b",
        profile="roleplay",
        tensor_parallelism=1,
        rates=(6.0,),
        num_stages=1,
        num_success=7200,
        num_failed=0,
        num_sampled_steps=15216,
        config_json_dir="Llama-2-7b-hf",
    ),

    # --- llama-2-70b (TP=4): general, codegen, roleplay ---
    ExperimentMeta(
        dir_name="20260217-202857-llama-2-70b-tp4-general",
        model_id="meta-llama/Llama-2-70b-hf",
        model_short="llama-2-70b",
        profile="general",
        tensor_parallelism=4,
        rates=(8.0, 20.0),
        num_stages=2,
        num_success=16800,
        num_failed=0,
        num_sampled_steps=5992,
        config_json_dir="Llama-2-70b-hf",
    ),
    ExperimentMeta(
        dir_name="20260217-203421-llama-2-70b-hf-tp4-codegen",
        model_id="meta-llama/Llama-2-70b-hf",
        model_short="llama-2-70b",
        profile="codegen",
        tensor_parallelism=4,
        rates=(5.0, 10.0),
        num_stages=2,
        num_success=9000,
        num_failed=0,
        num_sampled_steps=6679,
        config_json_dir="Llama-2-70b-hf",
    ),
    ExperimentMeta(
        dir_name="20260218-084319-llama-2-70b-tp4-roleplay",
        model_id="meta-llama/Llama-2-70b-hf",
        model_short="llama-2-70b",
        profile="roleplay",
        tensor_parallelism=4,
        rates=(6.0,),
        num_stages=1,
        num_success=7200,
        num_failed=0,
        num_sampled_steps=6741,
        config_json_dir="Llama-2-70b-hf",
    ),

    # --- mixtral-8x7b (TP=2): general, codegen, roleplay ---
    ExperimentMeta(
        dir_name="20260218-130541-mixtral-8x7b-v0-1-tp2-general",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        model_short="mixtral-8x7b",
        profile="general",
        tensor_parallelism=2,
        rates=(8.0, 20.0),
        num_stages=2,
        num_success=16800,
        num_failed=0,
        num_sampled_steps=5867,
        config_json_dir="Mixtral-8x7B-v0.1",
    ),
    ExperimentMeta(
        dir_name="20260218-120914-mixtral-8x7b-v0-1-tp2-codegen",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        model_short="mixtral-8x7b",
        profile="codegen",
        tensor_parallelism=2,
        rates=(5.0, 10.0),
        num_stages=2,
        num_success=9000,
        num_failed=0,
        num_sampled_steps=6544,
        config_json_dir="Mixtral-8x7B-v0.1",
    ),
    ExperimentMeta(
        dir_name="20260218-141024-mixtral-8x7b-v0-1-tp2-roleplay",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        model_short="mixtral-8x7b",
        profile="roleplay",
        tensor_parallelism=2,
        rates=(6.0,),
        num_stages=1,
        num_success=7200,
        num_failed=0,
        num_sampled_steps=6677,
        config_json_dir="Mixtral-8x7B-v0.1",
    ),

    # --- codellama-34b (TP=2): general, codegen, roleplay, reasoning ---
    ExperimentMeta(
        dir_name="20260218-150304-codellama-34b-tp2-general",
        model_id="codellama/CodeLlama-34b-Instruct-hf",
        model_short="codellama-34b",
        profile="general",
        tensor_parallelism=2,
        rates=(8.0, 20.0),
        num_stages=2,
        num_success=16800,
        num_failed=0,
        num_sampled_steps=7441,
        config_json_dir="CodeLlama-34b-Instruct-hf",
    ),
    ExperimentMeta(
        dir_name="20260218-150956-codellama-34b-tp2-codegen",
        model_id="codellama/CodeLlama-34b-Instruct-hf",
        model_short="codellama-34b",
        profile="codegen",
        tensor_parallelism=2,
        rates=(5.0, 10.0),
        num_stages=2,
        num_success=9000,
        num_failed=0,
        num_sampled_steps=8216,
        config_json_dir="CodeLlama-34b-Instruct-hf",
    ),
    ExperimentMeta(
        dir_name="20260218-155500-codellama-34b-tp2-roleplay",
        model_id="codellama/CodeLlama-34b-Instruct-hf",
        model_short="codellama-34b",
        profile="roleplay",
        tensor_parallelism=2,
        rates=(6.0,),
        num_stages=1,
        num_success=7200,
        num_failed=0,
        num_sampled_steps=8443,
        config_json_dir="CodeLlama-34b-Instruct-hf",
    ),
    ExperimentMeta(
        dir_name="20260218-160939-codellama-34b-tp2-reasoning",
        model_id="codellama/CodeLlama-34b-Instruct-hf",
        model_short="codellama-34b",
        profile="reasoning",
        tensor_parallelism=2,
        rates=(4.0,),
        num_stages=1,
        num_success=4796,
        num_failed=4,
        num_sampled_steps=4994,
        config_json_dir="CodeLlama-34b-Instruct-hf",
    ),
)

# Overload experiments excluded from fitting (>10% failure rate).
# The linear step-time model cannot capture nonlinear preemption cascade dynamics.
EXCLUDED_OVERLOAD: tuple[ExperimentMeta, ...] = (
    ExperimentMeta(
        dir_name="20260218-135247-mixtral-8x7b-v0-1-tp2-reasoning",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        model_short="mixtral-8x7b",
        profile="reasoning",
        tensor_parallelism=2,
        rates=(4.0,),
        num_stages=1,
        num_success=1506,
        num_failed=3294,
        num_sampled_steps=3676,
        config_json_dir="Mixtral-8x7B-v0.1",
    ),
    ExperimentMeta(
        dir_name="20260217-170634-llama-2-7b-tp1-reasoning",
        model_id="meta-llama/Llama-2-7b-hf",
        model_short="llama-2-7b",
        profile="reasoning",
        tensor_parallelism=1,
        rates=(4.0,),
        num_stages=1,
        num_success=732,
        num_failed=4068,
        num_sampled_steps=4966,
        config_json_dir="Llama-2-7b-hf",
    ),
    ExperimentMeta(
        dir_name="20260218-065057-llama-2-70b-hf-tp4-reasoning",
        model_id="meta-llama/Llama-2-70b-hf",
        model_short="llama-2-70b",
        profile="reasoning",
        tensor_parallelism=4,
        rates=(4.0,),
        num_stages=1,
        num_success=3200,
        num_failed=1600,
        num_sampled_steps=4448,
        config_json_dir="Llama-2-70b-hf",
    ),
)


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

# Root directory for all training data (relative to repo root).
DATA_ROOT = "training/default_args"

# Root directory for model config.json files (relative to repo root).
MODEL_CONFIGS_ROOT = "training/model_configs"


def get_active() -> list[ExperimentMeta]:
    """Return all active (non-overload) experiments."""
    return list(EXPERIMENTS)


def get_by_model(model_short: str) -> list[ExperimentMeta]:
    """Return active experiments for a given model."""
    return [e for e in EXPERIMENTS if e.model_short == model_short]


def get_by_profile(profile: str) -> list[ExperimentMeta]:
    """Return active experiments for a given profile."""
    return [e for e in EXPERIMENTS if e.profile == profile]


def experiment_dir(exp: ExperimentMeta, repo_root: str = ".") -> str:
    """Return absolute path to an experiment's data directory."""
    return os.path.join(repo_root, DATA_ROOT, exp.dir_name)


def config_json_path(exp: ExperimentMeta, repo_root: str = ".") -> str:
    """Return absolute path to the model's config.json."""
    return os.path.join(repo_root, MODEL_CONFIGS_ROOT, exp.config_json_dir, "config.json")


# =============================================================================
# INTEGRITY CHECKS — run on import to catch corruption early
# =============================================================================

def _validate_split_integrity() -> None:
    """Validate the experiment definitions. Raises AssertionError on any violation.

    Checks:
      1. No duplicate directory names (across both EXPERIMENTS and EXCLUDED_OVERLOAD).
      2. Exactly 13 active experiments.
      3. Active + excluded = 16 total.
      4. Every excluded experiment has >10% failure rate.
      5. No overlap between EXPERIMENTS and EXCLUDED_OVERLOAD.
    """
    # Check 1: no duplicates across both sets
    all_dirs = [e.dir_name for e in EXPERIMENTS] + [e.dir_name for e in EXCLUDED_OVERLOAD]
    assert len(all_dirs) == len(set(all_dirs)), (
        f"Duplicate experiment directory names: "
        f"{[d for d in all_dirs if all_dirs.count(d) > 1]}"
    )

    # Check 2: exactly 13 active experiments
    assert len(EXPERIMENTS) == 13, (
        f"Expected 13 active experiments, got {len(EXPERIMENTS)}"
    )

    # Check 3: total = 16
    assert len(EXPERIMENTS) + len(EXCLUDED_OVERLOAD) == 16, (
        f"Expected 16 total experiments, got {len(EXPERIMENTS) + len(EXCLUDED_OVERLOAD)}"
    )

    # Check 4: every excluded experiment has >10% failure rate
    for exp in EXCLUDED_OVERLOAD:
        assert exp.failure_rate > 0.10, (
            f"Excluded experiment {exp.dir_name} has {exp.failure_rate:.0%} failure rate, "
            f"expected >10%"
        )

    # Check 5: no overlap
    active_dirs = {e.dir_name for e in EXPERIMENTS}
    excluded_dirs = {e.dir_name for e in EXCLUDED_OVERLOAD}
    assert active_dirs.isdisjoint(excluded_dirs), (
        f"Overlap between EXPERIMENTS and EXCLUDED_OVERLOAD: "
        f"{active_dirs & excluded_dirs}"
    )


# Run checks on import — fail fast if experiment definitions are corrupted.
_validate_split_integrity()


# =============================================================================
# SUMMARY — printed when run as script
# =============================================================================

def print_summary() -> None:
    """Print a human-readable summary of active experiments."""
    total_req = sum(e.num_total for e in EXPERIMENTS)
    total_steps = sum(e.num_sampled_steps for e in EXPERIMENTS)
    total_success = sum(e.num_success for e in EXPERIMENTS)
    total_fail = sum(e.num_failed for e in EXPERIMENTS)

    print(f"\n{'=' * 72}")
    print(f"  ACTIVE — {len(EXPERIMENTS)} experiments, "
          f"{total_req:,} requests ({total_success:,} ok + {total_fail:,} fail), "
          f"{total_steps:,} steps")
    print(f"{'=' * 72}")

    for e in EXPERIMENTS:
        rates_str = "/".join(str(r) for r in e.rates)
        fail_str = f" ({e.failure_rate:.0%} fail)" if e.num_failed > 0 else ""
        print(
            f"  {e.model_short:<16} {e.profile:<12} TP={e.tensor_parallelism}  "
            f"{rates_str:>8} RPS  {e.num_total:>6,} req{fail_str:>12}  "
            f"{e.num_sampled_steps:>6,} steps"
        )

    print(f"\n  EXCLUDED (overload, >10% failure rate):")
    for e in EXCLUDED_OVERLOAD:
        print(f"    {e.model_short:<16} {e.profile:<12} "
              f"{e.failure_rate:.0%} failure ({e.num_failed}/{e.num_total})")


if __name__ == "__main__":
    print_summary()
