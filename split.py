"""
Train / Validate / Test split for BLIS latency model development.

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR DATA PARTITIONING.
All training, validation, and evaluation code MUST import splits from here.
Never hard-code experiment lists elsewhere.

Split Design Principles
=======================

1. The test unit is a FULL EXPERIMENT, not individual requests.
   BLIS simulates an entire arrival stream end-to-end — request N's queueing
   delay depends on requests 1..N-1. Splitting requests within an experiment
   would leak queueing dynamics.

2. No step-level data from validate/test experiments may be used for training.
   Step data encodes the batch composition patterns that emerge from the
   specific (model, rate, profile) combination. If the model has "seen" those
   patterns, BLIS might match because it learned the correlation structure,
   not the causal mechanism.

3. The split tests three generalization axes relevant to BLIS users:
   - Cross-rate: can the model trained at 5-20 RPS predict overload (saturation)?
   - Cross-profile: can it generalize from moderate workloads to extreme ones?
   - Cross-model: does codellama-34b (48 layers, unique intermediate_dim) work
     when only general-profile step data was used in training?

Split Layout
============

TRAIN (10 experiments, 107,400 requests, ~104K sampled steps):
    Step-level regression data: (model_arch, batch_features) → step_duration.
    Three models fully covered (general + codegen + roleplay).
    One model partially covered (codellama general only — provides architecture
    baseline without exposing codegen/roleplay/reasoning dynamics).

    Covers:
    - 4 model architectures (dense 7B/34B/70B + MoE 8x7B)
    - 3 workload profiles (general, codegen, roleplay)
    - 2 load stages per 2-stage experiment (rate variation within training)
    - MHA (7b) vs GQA (70b, mixtral, codellama) attention

VALIDATE (3 experiments, 16,200 requests, ~18K sampled steps):
    Full BLIS replay comparison. Used for model structure / hyperparameter tuning.
    - codellama codegen + roleplay: tests cross-profile generalization on a model
      whose general-profile is in training (architecture known, dynamics unseen).
    - mixtral reasoning: tests overload regime on MoE architecture.

TEST (3 experiments, 9,600 requests, ~14K sampled steps):
    Full BLIS replay comparison. NEVER used for training or tuning decisions.
    - All reasoning profiles: the hardest regime (saturation, preemptions, timeouts).
    - 3 different model scales: 7B (85% failure), 70B (33% failure), 34B (0.1% failure).
    - Tests whether a model trained on moderate load can predict where the cliff is.

Evaluation Protocol
===================

For VALIDATE and TEST experiments, the evaluation is:

    1. Configure BLIS:
       - Learned latency model coefficients
       - Model architecture from config.json
       - Server config from exp-config.yaml (TP, max_batch_tokens, max_seqs)
       - KV blocks from step traces (kv.blocks_total_gpu)

    2. Replay workload:
       - Arrival times from per_request_lifecycle_metrics.json (start_time)
       - Per-request input_tokens, output_tokens
       - Prefix group assignment (derivable from prompt content)

    3. Compare at three levels:
       Level 1 (step):     simulated step_duration vs real step.duration_us
       Level 2 (request):  simulated TTFT, E2E vs real per-request TTFT, E2E
       Level 3 (aggregate): simulated p50/p99/throughput vs real summary metrics
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Split(str, Enum):
    """Data split assignment."""
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"


@dataclass(frozen=True)
class ExperimentMeta:
    """Immutable metadata for one experiment.

    All fields are derived from the experiment data at split-definition time
    and frozen here so downstream code never needs to re-parse metadata.
    """

    dir_name: str
    """Experiment directory name (e.g., '20260218-150304-codellama-34b-tp2-general')."""

    split: Split
    """Assigned data split."""

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
# SPLIT DEFINITION — the single source of truth
# =============================================================================
#
# IMPORTANT: If you need to change the split, change ONLY this section.
# All downstream code reads from EXPERIMENTS.
# =============================================================================

EXPERIMENTS: tuple[ExperimentMeta, ...] = (
    # =========================================================================
    # TRAIN — 10 experiments
    # =========================================================================

    # --- llama-2-7b (TP=1): general, codegen, roleplay ---
    ExperimentMeta(
        dir_name="20260217-231439-llama-2-7b-tp1-general",
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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
        split=Split.TRAIN,
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

    # --- codellama-34b (TP=2): general only ---
    ExperimentMeta(
        dir_name="20260218-150304-codellama-34b-tp2-general",
        split=Split.TRAIN,
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

    # =========================================================================
    # VALIDATE — 3 experiments
    # =========================================================================

    # codellama codegen + roleplay: cross-profile on architecture seen in train
    ExperimentMeta(
        dir_name="20260218-150956-codellama-34b-tp2-codegen",
        split=Split.VALIDATE,
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
        split=Split.VALIDATE,
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

    # mixtral reasoning: cross-regime (overload) on MoE architecture
    ExperimentMeta(
        dir_name="20260218-135247-mixtral-8x7b-v0-1-tp2-reasoning",
        split=Split.VALIDATE,
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

    # =========================================================================
    # TEST — 3 experiments (NEVER touch for training or tuning)
    # =========================================================================

    # All reasoning profiles: saturation regime, preemptions, timeouts.
    # Tests whether model trained on moderate load predicts the cliff.
    ExperimentMeta(
        dir_name="20260217-170634-llama-2-7b-tp1-reasoning",
        split=Split.TEST,
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
        split=Split.TEST,
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
    ExperimentMeta(
        dir_name="20260218-160939-codellama-34b-tp2-reasoning",
        split=Split.TEST,
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

# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

# Root directory for all training data (relative to repo root).
DATA_ROOT = "training/default_args"

# Root directory for model config.json files (relative to repo root).
MODEL_CONFIGS_ROOT = "training/model_configs"


def get_split(split: Split) -> list[ExperimentMeta]:
    """Return all experiments in the given split."""
    return [e for e in EXPERIMENTS if e.split == split]


def get_train() -> list[ExperimentMeta]:
    """Return training experiments. Safe to use for coefficient learning."""
    return get_split(Split.TRAIN)


def get_validate() -> list[ExperimentMeta]:
    """Return validation experiments. For model structure / hyperparameter tuning only."""
    return get_split(Split.VALIDATE)


def get_test() -> list[ExperimentMeta]:
    """Return test experiments. NEVER use for training or tuning."""
    return get_split(Split.TEST)


def get_by_model(model_short: str, split: Optional[Split] = None) -> list[ExperimentMeta]:
    """Return experiments for a given model, optionally filtered by split."""
    return [
        e for e in EXPERIMENTS
        if e.model_short == model_short and (split is None or e.split == split)
    ]


def get_by_profile(profile: str, split: Optional[Split] = None) -> list[ExperimentMeta]:
    """Return experiments for a given profile, optionally filtered by split."""
    return [
        e for e in EXPERIMENTS
        if e.profile == profile and (split is None or e.split == split)
    ]


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
    """Validate the split definition. Raises AssertionError on any violation.

    Checks:
      1. Every experiment appears exactly once.
      2. No directory name is duplicated.
      3. All 16 experiments are accounted for.
      4. Train/Validate/Test sets are mutually exclusive.
      5. No reasoning profile in training set (prevents overload regime leakage).
      6. Every model in validate/test has at least one experiment in train.
    """
    # Check 1 & 2: no duplicates
    dir_names = [e.dir_name for e in EXPERIMENTS]
    assert len(dir_names) == len(set(dir_names)), (
        f"Duplicate experiment directory names: "
        f"{[d for d in dir_names if dir_names.count(d) > 1]}"
    )

    # Check 3: exactly 16 experiments
    assert len(EXPERIMENTS) == 16, (
        f"Expected 16 experiments, got {len(EXPERIMENTS)}"
    )

    # Check 4: mutual exclusivity (implied by check 1+2, but explicit)
    train_dirs = {e.dir_name for e in get_train()}
    val_dirs = {e.dir_name for e in get_validate()}
    test_dirs = {e.dir_name for e in get_test()}
    assert train_dirs.isdisjoint(val_dirs), "Train/Validate overlap"
    assert train_dirs.isdisjoint(test_dirs), "Train/Test overlap"
    assert val_dirs.isdisjoint(test_dirs), "Validate/Test overlap"

    # Check 5: no reasoning in training (overload regime reserved for eval)
    train_profiles = {e.profile for e in get_train()}
    assert "reasoning" not in train_profiles, (
        "Reasoning profile must not be in training set — overload regime "
        "should only appear in validate/test for generalization testing."
    )

    # Check 6: every validate/test model has training coverage
    train_models = {e.model_short for e in get_train()}
    for exp in get_validate() + get_test():
        assert exp.model_short in train_models, (
            f"Model {exp.model_short!r} in {exp.split.value} set has no "
            f"training experiments. Cannot evaluate cross-profile generalization "
            f"without architecture baseline in training."
        )

    # Check 7: split sizes
    assert len(get_train()) == 10, f"Expected 10 train, got {len(get_train())}"
    assert len(get_validate()) == 3, f"Expected 3 validate, got {len(get_validate())}"
    assert len(get_test()) == 3, f"Expected 3 test, got {len(get_test())}"


# Run checks on import — fail fast if split definition is corrupted.
_validate_split_integrity()


# =============================================================================
# SUMMARY — printed when run as script
# =============================================================================

def print_summary() -> None:
    """Print a human-readable summary of the split."""
    for split in (Split.TRAIN, Split.VALIDATE, Split.TEST):
        exps = get_split(split)
        total_req = sum(e.num_total for e in exps)
        total_steps = sum(e.num_sampled_steps for e in exps)
        total_success = sum(e.num_success for e in exps)
        total_fail = sum(e.num_failed for e in exps)

        print(f"\n{'=' * 72}")
        print(f"  {split.value.upper()} — {len(exps)} experiments, "
              f"{total_req:,} requests ({total_success:,} ok + {total_fail:,} fail), "
              f"{total_steps:,} steps")
        print(f"{'=' * 72}")

        for e in exps:
            rates_str = "/".join(str(r) for r in e.rates)
            fail_str = f" ({e.failure_rate:.0%} fail)" if e.num_failed > 0 else ""
            print(
                f"  {e.model_short:<16} {e.profile:<12} TP={e.tensor_parallelism}  "
                f"{rates_str:>8} RPS  {e.num_total:>6,} req{fail_str:>12}  "
                f"{e.num_sampled_steps:>6,} steps"
            )

    # Cross-reference summary
    print(f"\n{'=' * 72}")
    print("  GENERALIZATION TESTS")
    print(f"{'=' * 72}")
    print("  Validate:")
    print("    codellama codegen/roleplay — cross-profile (arch in train via general)")
    print("    mixtral reasoning          — overload regime on MoE")
    print("  Test:")
    print("    llama-2-7b reasoning       — overload on small model (85% failure)")
    print("    llama-2-70b reasoning      — overload on large model (33% failure)")
    print("    codellama-34b reasoning    — overload on medium model (0.1% failure)")


if __name__ == "__main__":
    print_summary()
