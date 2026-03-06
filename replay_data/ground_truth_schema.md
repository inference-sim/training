# Ground Truth Experiment Directory Schema

Each directory under `default_args/` is a self-contained experiment run from **inference-perf** against a live vLLM server. The directory name encodes the experiment identity.

## Directory Naming Convention

```
<YYYYMMDD>-<HHMMSS>-<model-slug>-tp<N>-<workload-type>
```

| Segment | Example | Meaning |
|---------|---------|---------|
| Timestamp | `20260217-162547` | When the experiment started |
| Model slug | `llama-2-7b` | HuggingFace model (abbreviated) |
| TP | `tp1` | Tensor parallelism degree |
| Workload | `roleplay` | Workload profile (roleplay, general, codegen, reasoning) |

## File Layout

```
<experiment>/
├── exp-config.yaml              # vLLM server-side configuration
├── profile.yaml                 # inference-perf client-side configuration (compact JSON)
├── results/                     # inference-perf output artifacts
│   ├── config.yaml              # Resolved/expanded inference-perf config
│   ├── summary_lifecycle_metrics.json   # Aggregate latency/throughput (all stages combined)
│   ├── stage_0_lifecycle_metrics.json   # Per-stage metrics (stage 0 = always present)
│   ├── stage_1_lifecycle_metrics.json   # Per-stage metrics (stage 1 = present in multi-rate runs)
│   ├── per_request_lifecycle_metrics.json  # Per-request token-level timings (~100MB+)
│   ├── stdout.log               # inference-perf stdout (config echo + run log)
│   └── stderr.log               # inference-perf stderr
├── traces.json                  # Per-request raw traces (prompt, token timestamps) — large
├── kv_events.jsonl              # KV cache block-level events (store/transfer/evict)
├── vllm_logging.json            # vLLM Python logging config used during the run
└── vllm.log                     # vLLM server log (~20K lines)
```

## File Schemas

### `exp-config.yaml` — vLLM Server Config

Captures the vLLM launch arguments for this experiment.

```python
class ExpConfig(BaseModel):
    model: str                    # e.g. "meta-llama/Llama-2-7b-hf"
    tensor_parallelism: int       # e.g. 1, 2, 4
    max_model_len: int            # e.g. 4096
    max_num_batched_tokens: int   # e.g. 2048
    max_num_seqs: int             # e.g. 128
    app: str                      # always "inference-perf"
```

### `profile.yaml` — Inference-Perf Client Config

A single-line JSON blob (loaded as YAML) specifying how the load generator was configured. The expanded version lives in `results/config.yaml`.

```python
class LoadStage(BaseModel):
    rate: float                   # requests/sec for this stage
    duration: int                 # stage duration in seconds

class SharedPrefixData(BaseModel):
    num_groups: int               # number of unique system prompts (or similar)
    num_prompts_per_group: int    # users per system prompt
    system_prompt_len: int        # tokens
    question_len: int             # tokens
    output_len: int               # max output tokens

class LoadConfig(BaseModel):
    type: str                     # "constant"
    stages: list[LoadStage]
    num_workers: int
    worker_max_concurrency: int

class InferencePerfProfile(BaseModel):
    api: dict                     # {"type": "completion", "streaming": true}
    data: dict                    # {"type": "shared_prefix", "shared_prefix": SharedPrefixData}
    load: LoadConfig
    server: dict                  # {"type": "vllm", "model_name": str, "base_url": str}
    tokenizer: dict               # {"pretrained_model_name_or_path": str}
    report: dict                  # which lifecycle reports to generate
    storage: dict                 # output path config
```

### `summary_lifecycle_metrics.json` / `stage_<N>_lifecycle_metrics.json` — Aggregate Metrics

Both share the same schema. Summary covers all stages combined; `stage_<N>` covers a single load stage. Multi-rate experiments have `stage_0` + `stage_1`; single-rate experiments have only `stage_0`.

```python
class PercentileDistribution(BaseModel):
    mean: float
    min: float
    p0_1: float    # field names use "p0.1", "p1", ..., "p99.9" in JSON
    p1: float
    p5: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    p95: float
    p99: float
    p99_9: float
    max: float

class LatencyMetrics(BaseModel):
    request_latency: PercentileDistribution          # end-to-end request latency (seconds)
    time_to_first_token: PercentileDistribution       # TTFT (seconds)
    time_per_output_token: PercentileDistribution     # TPOT — mean ITL per request (seconds)
    inter_token_latency: PercentileDistribution       # ITL — all individual token gaps (seconds)
    normalized_time_per_output_token: PercentileDistribution  # TPOT normalized by output length

class ThroughputMetrics(BaseModel):
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    requests_per_sec: float

class LoadSummary(BaseModel):
    count: int                                # total requests sent
    schedule_delay: PercentileDistribution    # delta between intended and actual send time
    send_duration: float | None               # wall-clock seconds (per-stage only, absent in summary)
    requested_rate: float | None              # configured RPS (per-stage only)
    achieved_rate: float | None               # actual RPS achieved (per-stage only)

class SuccessMetrics(BaseModel):
    count: int
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    prompt_len: PercentileDistribution        # input token count distribution
    output_len: PercentileDistribution        # output token count distribution

class FailureMetrics(BaseModel):
    count: int
    request_latency: PercentileDistribution | None
    prompt_len: PercentileDistribution | None

class LifecycleMetrics(BaseModel):
    load_summary: LoadSummary
    successes: SuccessMetrics
    failures: FailureMetrics
```

### `per_request_lifecycle_metrics.json` — Per-Request Token Timings

A JSON array of per-request records. This is the largest file (~100MB+), containing individual token arrival timestamps.

**Important**: All timestamps in this file are Python `time.monotonic()` clock seconds on the inference-perf client machine. They are NOT Unix epoch timestamps — only differences between them are meaningful.

```python
class RequestError(BaseModel):
    error_type: str               # e.g. "TimeoutError"
    error_msg: str                # may be empty string

class PerRequestMetrics(BaseModel):
    start_time: float             # monotonic clock seconds (time.monotonic()) — request send time
    end_time: float               # monotonic clock seconds — last token received
    request: str                  # JSON-encoded request body (prompt + params)
    response: str                 # typically empty for streaming
    info: RequestInfo
    error: RequestError | None    # null on success; present on failure (e.g., TimeoutError)

class RequestInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    output_token_times: list[float]   # monotonic clock seconds per SSE event (~2× output_tokens)
```

### `kv_events.jsonl` — KV Cache Events

One JSON array per line. Each line is a timestamped event batch:

```
[timestamp, [events...], flags, metadata]
```

Event types include:
- `BlockStored` — KV block written (includes token IDs, block hash, tier)
- `CacheStoreCommitted` — block committed to a tier (GPU/CPU)
- `TransferInitiated` / `TransferCompleted` — cross-tier block migration

### `traces.json` — Raw Request Traces

Large JSON file with the raw request/response data including full prompt text and per-token timestamps. Superset of `per_request_lifecycle_metrics.json` with the actual prompt content.

### `vllm.log` — Server Log

Standard vLLM server log output (~20K lines). Contains model loading, batch scheduling decisions, and engine metrics. Contains GPU KV cache size, which can be divided by block size to get total_kv_blocks.

### `vllm_logging.json` — Logging Config

Python `logging.dictConfig` format specifying how vLLM logs were routed during the experiment. Informational only.
