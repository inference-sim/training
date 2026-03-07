# training

Training data and coefficient fitting pipeline for the [inference-sim](https://github.com/inference-sim/inference-sim) crossmodel latency prediction model.

## Overview

This repo contains 16 experiments (4 models x 4 workload profiles) of vLLM serving traces collected with [inference-perf](https://github.com/kubernetes-sigs/inference-perf), plus a pipeline that:

1. **Validates** trace data integrity (`validate_traces.py`)
2. **Reconstructs** per-step batch composition from journey events (`reconstruct_steps.py`)
3. **Computes** analytical basis functions for the latency model (`basis_functions.py`)
4. **Fits** 10 model coefficients via three-phase fitting (`fit_coefficients.py` — planned)
5. **Evaluates** accuracy against held-out experiments (`evaluate.py` — planned)

Design: [inference-sim/inference-sim#489](https://github.com/inference-sim/inference-sim/issues/489#issuecomment-4013680061) | Fitting spec: [inference-sim/training#3](https://github.com/inference-sim/training/issues/3)

## Dataset

**Models:** Llama-2-7b (TP=1), Llama-2-70b (TP=4), Mixtral-8x7B-v0.1 (TP=2), CodeLlama-34b (TP=2)

**Profiles:** general, codegen, roleplay, reasoning

**Split:** 10 train / 3 validate / 3 test (see `split.py` for rationale)

## Directory layout

```
split.py                Single source of truth for experiment metadata and data splits
schemas.py              Pydantic schemas for all data formats
trace_parser.py         Shared OTEL trace parsing utilities
validate_traces.py      Journey trace validation (5 correctness checks)
reconstruct_steps.py    Step reconstruction from journey events
basis_functions.py      Analytical basis functions for the latency model

tests/                  Behavioral unit tests
  conftest.py             JourneyBuilder fixture for synthetic trace data
  test_reconstruct_steps.py
  test_basis_functions.py

model_configs/          HuggingFace config.json per model
datasheets/             GPU hardware specs (H100 SXM)
default_args/           Raw experiment data
  <experiment>/
    exp-config.yaml       vLLM server parameters
    traces.json           OTEL journey + step traces (gitignored, large)
    results/              inference-perf aggregate metrics

output/                 Generated pipeline outputs (gitignored)
  validate/               Per-experiment validation JSON + summary
  reconstruct/            Per-experiment step + request JSON + summary
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Validate all 16 experiments (writes to output/validate/)
python3 validate_traces.py

# Reconstruct step batches and request labels (writes to output/reconstruct/)
python3 reconstruct_steps.py

# Run tests
pytest
```

## Key concepts

**Teacher-forced reconstruction:** We use the *real* batch compositions from the actual vLLM execution (reconstructed from journey events), not simulated ones. This avoids circular dependencies where predictions alter batch compositions.

**Greedy-fill prefill:** When a prompt spans multiple scheduler steps (chunked prefill), tokens are distributed using `max_num_batched_tokens` as the budget cap, mirroring vLLM's scheduler. Decode requests get their 1 token first, remaining budget goes to prefill.

**Preemption handling:** Requests preempted during decode resume with correct context length (accounting for the gap). Requests preempted during prefill re-enter the PREFILL phase with the correct remaining token count derived from `prefill.done_tokens`.
