# training

Training data and utilities for the inference-sim latency prediction model.

## Dataset

16 experiments across 4 models and 4 workload profiles, collected using
[inference-perf](https://github.com/kubernetes-sigs/inference-perf) against
instrumented vLLM servers.

**Models:** Llama-2-7b (TP=1), Llama-2-70b (TP=4), Mixtral-8x7B-v0.1 (TP=2),
CodeLlama-34b (TP=2)

**Profiles:** general, codegen, roleplay, reasoning

### Directory layout

```
default_args/           Raw experiment data (configs + aggregate metrics)
  <experiment>/
    exp-config.yaml       vLLM server parameters
    profile.yaml          inference-perf workload profile
    vllm_logging.json     Logging config
    results/
      config.yaml                       Resolved inference-perf config
      summary_lifecycle_metrics.json    Aggregate latency/throughput
      stage_N_lifecycle_metrics.json    Per-stage breakdown
replay_data/            Condensed workload + ground-truth for ML training
  <experiment>.json               Workload spec (arrival times, token counts)
  <experiment>_ground_truth.json  Observed e2e latency and TTFT per request
model_configs/          HuggingFace model config.json files
schemas.py              Pydantic schemas for all data formats
split.py                Train/validate/test split definitions
```

> **Note:** Large telemetry files (traces, KV events, per-request metrics,
> server logs) are excluded from this repo via `.gitignore`. See `schemas.py`
> for their full schema documentation.

## Setup

```bash
pip install -e .
```
