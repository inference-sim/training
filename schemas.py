"""
Pydantic schemas for vLLM training data collected via inference-perf.

This module defines the complete data model for all files produced by the
training data collection pipeline:

    inference-perf (load generator)
         │
         │  HTTP requests
         ▼
    vLLM Server (instrumented)
         │
         ├─► traces.json          (OTEL: journey + step tracing)
         ├─► kv_events.jsonl      (ZMQ: KV cache events)
         └─► vllm.log             (server log)
         │
         ▼
    inference-perf (results)
         ├─► per_request_lifecycle_metrics.json
         ├─► stage_N_lifecycle_metrics.json
         ├─► summary_lifecycle_metrics.json
         └─► config.yaml

Correlation keys between data sources:
    ┌──────────────────────┬──────────────────────┬────────────────────────┐
    │ Source A              │ Source B              │ Key                    │
    ├──────────────────────┼──────────────────────┼────────────────────────┤
    │ traces (journey)     │ traces (step)         │ scheduler.step=step.id │
    │ traces (journey)     │ kv_events             │ request_id + step      │
    │ kv_events            │ traces (step)         │ scheduler_step=step.id │
    │ traces (journey)     │ per_request_metrics   │ gen_ai.request.id      │
    │ per_request_metrics  │ summary/stage metrics │ aggregate of per-req   │
    └──────────────────────┴──────────────────────┴────────────────────────┘

Data provenance (2026-02-18 collection):
    - 16 experiments: 4 models × 4 workload profiles
    - Models: llama-2-7b (TP=1), llama-2-70b (TP=4), mixtral-8x7b (TP=2),
              codellama-34b (TP=2)
    - Profiles: general, codegen, roleplay, reasoning
    - Each experiment: 2 load stages (e.g., 8 RPS × 600s, then 20 RPS × 600s)
"""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# 1. EXPERIMENT CONFIGURATION
# =============================================================================


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration (exp-config.yaml).

    Defines the vLLM server parameters and the benchmarking tool used.
    One file per experiment directory.

    Example:
        model: codellama/CodeLlama-34b-Instruct-hf
        tensor_parallelism: 2
        max_model_len: 4096
        max_num_batched_tokens: 2048
        max_num_seqs: 128
        app: inference-perf
    """

    model: str = Field(
        description="HuggingFace model identifier (e.g., 'codellama/CodeLlama-34b-Instruct-hf')."
    )
    tensor_parallelism: int = Field(
        description="Tensor parallelism degree — number of GPUs used in parallel for one model."
    )
    max_model_len: int = Field(
        description="Maximum sequence length (prompt + output tokens) the server is configured to accept."
    )
    max_num_batched_tokens: int = Field(
        description="vLLM's maximum number of tokens that can be batched in a single scheduler step."
    )
    max_num_seqs: int = Field(
        description="Maximum number of concurrent sequences (requests) the server will process."
    )
    app: str = Field(
        description="Benchmarking tool used to generate load (always 'inference-perf' in this dataset)."
    )


# =============================================================================
# 2. INFERENCE-PERF PROFILE (profile.yaml)
# =============================================================================


class SharedPrefixConfig(BaseModel):
    """Configuration for shared-prefix multi-turn chat workload.

    Simulates multiple users sharing system prompts (prefix groups)
    with individual questions, as in chatbot deployments.

    Note: The profile.yaml also contains fields 'num_unique_system_prompts' and
    'num_users_per_system_prompt' which are the user-facing aliases. The resolved
    config.yaml uses 'num_groups' and 'num_prompts_per_group'.
    """

    num_groups: Optional[int] = Field(
        None,
        description="Number of unique system prompt groups (resolved config). "
        "Absent in profile.yaml; use num_unique_system_prompts instead.",
    )
    num_prompts_per_group: Optional[int] = Field(
        None,
        description="Prompts per group (resolved config). "
        "Absent in profile.yaml; use num_users_per_system_prompt instead.",
    )
    num_unique_system_prompts: Optional[int] = Field(
        None,
        description="Number of unique system prompt groups (profile.yaml alias for num_groups).",
    )
    num_users_per_system_prompt: Optional[int] = Field(
        None,
        description="Users per system prompt (profile.yaml alias for num_prompts_per_group).",
    )
    enable_multi_turn_chat: Optional[bool] = Field(
        None,
        description="Whether multi-turn chat is enabled. Present in profile.yaml but not resolved config.",
    )
    system_prompt_len: int = Field(
        description="Length of system prompt prefix in tokens."
    )
    question_len: int = Field(
        description="Length of user question (non-prefix portion) in tokens."
    )
    output_len: int = Field(
        description="Target number of output tokens to generate per request."
    )


class LoadStage(BaseModel):
    """A single load stage with constant request rate and duration.

    Experiments typically use two stages: a warm-up stage at lower rate
    followed by a steady-state stage at higher rate.
    """

    rate: float = Field(
        description="Target request arrival rate in requests per second."
    )
    duration: int = Field(
        description="Duration of this stage in seconds."
    )


class LoadConfig(BaseModel):
    """Load generation configuration.

    Defines how requests are sent over time. In this dataset, all experiments
    use 'constant' type with two stages.
    """

    type: str = Field(
        description="Load generation pattern. Always 'constant' in this dataset."
    )
    interval: Optional[float] = Field(
        None,
        description="Base interval between requests (resolved config only). "
        "Overridden by stages.",
    )
    stages: list[LoadStage] = Field(
        description="Ordered list of load stages. Typically 2: warm-up + steady-state."
    )
    sweep: Optional[Any] = Field(
        None,
        description="Sweep configuration for rate sweeps. Null in this dataset.",
    )
    num_workers: Optional[int] = Field(
        None,
        description="Number of concurrent HTTP workers (resolved config only).",
    )
    worker_max_concurrency: Optional[int] = Field(
        None,
        description="Maximum concurrent requests per worker (resolved config only).",
    )
    worker_max_tcp_connections: Optional[int] = Field(
        None,
        description="Maximum TCP connections per worker (resolved config only).",
    )
    circuit_breakers: Optional[list] = Field(
        None,
        description="Circuit breaker configs (resolved config only). Empty list in this dataset.",
    )
    request_timeout: Optional[float] = Field(
        None,
        description="Per-request timeout in seconds (resolved config only). Null = no timeout.",
    )


class APIConfig(BaseModel):
    """API configuration for inference-perf requests."""

    type: str = Field(
        description="API endpoint type. Always 'completion' in this dataset "
        "(uses /v1/completions, not /v1/chat/completions)."
    )
    streaming: bool = Field(
        description="Whether to use streaming SSE responses. Always true in this dataset."
    )
    headers: Optional[dict] = Field(
        None,
        description="Additional HTTP headers. Null in this dataset.",
    )


class DataConfig(BaseModel):
    """Workload data configuration."""

    type: str = Field(
        description="Workload data type. Always 'shared_prefix' in this dataset."
    )
    path: Optional[str] = Field(
        None,
        description="Path to external data file. Null for synthetic workloads.",
    )
    input_distribution: Optional[Any] = Field(
        None,
        description="Input token length distribution. Null when using shared_prefix.",
    )
    output_distribution: Optional[Any] = Field(
        None,
        description="Output token length distribution. Null when using shared_prefix.",
    )
    shared_prefix: Optional[SharedPrefixConfig] = Field(
        None,
        description="Shared-prefix workload configuration. Present when type='shared_prefix'.",
    )


class ReportConfig(BaseModel):
    """Report generation configuration."""

    class RequestLifecycleReport(BaseModel):
        summary: bool = Field(description="Generate summary_lifecycle_metrics.json.")
        per_stage: bool = Field(description="Generate stage_N_lifecycle_metrics.json for each stage.")
        per_request: bool = Field(description="Generate per_request_lifecycle_metrics.json.")

    class PrometheusReport(BaseModel):
        summary: bool = Field(description="Generate Prometheus summary metrics.")
        per_stage: bool = Field(description="Generate per-stage Prometheus metrics.")

    request_lifecycle: RequestLifecycleReport
    prometheus: Optional[PrometheusReport] = None


class ServerConfig(BaseModel):
    """Target server configuration."""

    type: str = Field(description="Server type. Always 'vllm' in this dataset.")
    model_name: str = Field(description="Model name to send in API requests.")
    base_url: str = Field(description="Base URL of the vLLM server (e.g., 'http://10.130.9.155:8000').")
    ignore_eos: bool = Field(
        description="Whether to ignore EOS token and generate exactly max_tokens. "
        "Always true in this dataset to ensure deterministic output lengths."
    )
    api_key: Optional[str] = Field(None, description="API key for authentication. Null in this dataset.")


class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    pretrained_model_name_or_path: str = Field(
        description="HuggingFace model path for the tokenizer."
    )
    trust_remote_code: Optional[bool] = None
    token: Optional[str] = None


class StorageConfig(BaseModel):
    """Storage configuration for results."""

    class LocalStorage(BaseModel):
        path: str = Field(description="Filesystem path where results are written.")
        report_file_prefix: Optional[str] = None

    local_storage: LocalStorage
    google_cloud_storage: Optional[Any] = None
    simple_storage_service: Optional[Any] = None


class InferencePerfConfig(BaseModel):
    """Fully resolved inference-perf configuration (results/config.yaml).

    This is the complete, resolved version of what was specified in profile.yaml.
    Contains all defaults filled in and all paths resolved.
    """

    api: APIConfig
    data: DataConfig
    load: LoadConfig
    report: ReportConfig
    server: ServerConfig
    tokenizer: TokenizerConfig
    storage: StorageConfig
    metrics: Optional[Any] = None
    circuit_breakers: Optional[Any] = None


# =============================================================================
# 3. PER-REQUEST LIFECYCLE METRICS (per_request_lifecycle_metrics.json)
# =============================================================================


class RequestInfo(BaseModel):
    """Token-level details for a single completed request.

    The output_token_times array captures the monotonic timestamp of each
    SSE event from the streaming response. For streaming completions, vLLM
    sends paired SSE data/event lines, resulting in approximately 2× entries
    relative to output_tokens. The exact ratio depends on vLLM's streaming
    implementation and can vary.

    Derivable metrics:
        TTFT = output_token_times[0] - start_time    (first SSE event)
        ITL  = diff(output_token_times[::2])          (every other = actual tokens)
        TPOT = (end_time - output_token_times[0]) / output_tokens
        E2E  = end_time - start_time
    """

    input_tokens: int = Field(
        description="Number of prompt tokens as counted by the server's tokenizer."
    )
    output_tokens: int = Field(
        description="Number of output tokens generated. May differ slightly from "
        "max_tokens due to tokenization boundaries."
    )
    output_token_times: list[float] = Field(
        description="Monotonic timestamps (seconds) of each SSE event received. "
        "Approximately 2× output_tokens entries due to paired SSE data/event lines. "
        "Clock is Python's time.monotonic() on the inference-perf client."
    )


class RequestError(BaseModel):
    """Error details for a failed request.

    Present in reasoning experiments where some requests time out due to
    long generation under high load.
    """

    error_type: str = Field(
        description="Error class name (e.g., 'TimeoutError')."
    )
    error_msg: str = Field(
        description="Error message. May be empty string."
    )


class PerRequestMetrics(BaseModel):
    """Metrics for a single request (one entry in per_request_lifecycle_metrics.json).

    Timestamps are monotonic clock seconds (Python time.monotonic() on the
    inference-perf client machine). They are NOT Unix timestamps — only
    differences between them are meaningful.

    The 'request' field contains the raw JSON string of the HTTP request body
    sent to vLLM. Parse it to extract model, prompt, max_tokens, etc.
    """

    start_time: float = Field(
        description="Monotonic timestamp (seconds) when the HTTP request was sent."
    )
    end_time: float = Field(
        description="Monotonic timestamp (seconds) when the last SSE event was received."
    )
    request: str = Field(
        description="Raw JSON string of the HTTP request body sent to vLLM. "
        "Contains model, prompt, max_tokens, ignore_eos, stream fields."
    )
    response: str = Field(
        description="Raw response body. Empty string for streaming requests "
        "(tokens delivered via SSE, not final body)."
    )
    info: RequestInfo = Field(
        description="Token counts and per-token timing data."
    )
    error: Optional[Union[str, RequestError]] = Field(
        None,
        description="Error details if the request failed. Null on success. "
        "Can be a plain string or a RequestError object with error_type and error_msg. "
        "Reasoning experiments produce RequestError objects (e.g., TimeoutError).",
    )


# =============================================================================
# 4. AGGREGATE LIFECYCLE METRICS (summary + stage)
# =============================================================================


class PercentileDistribution(BaseModel):
    """Statistical distribution with standard percentiles.

    All latency values are in seconds unless otherwise noted.
    """

    mean: float
    min: float
    p0_1: float = Field(alias="p0.1", description="0.1th percentile.")
    p1: float = Field(description="1st percentile.")
    p5: float = Field(description="5th percentile.")
    p10: float = Field(description="10th percentile.")
    p25: float = Field(description="25th percentile (Q1).")
    median: float = Field(description="50th percentile (median).")
    p75: float = Field(description="75th percentile (Q3).")
    p90: float = Field(description="90th percentile.")
    p95: float = Field(description="95th percentile.")
    p99: float = Field(description="99th percentile.")
    p99_9: float = Field(alias="p99.9", description="99.9th percentile.")
    max: float

    model_config = {"populate_by_name": True}


class LatencyMetrics(BaseModel):
    """Latency metrics for successful requests.

    All values are in seconds.

    Metric definitions:
        request_latency:                 end_time - start_time (full E2E)
        time_to_first_token:             first SSE event - start_time
        time_per_output_token:           mean inter-token interval
        normalized_time_per_output_token: request_latency / output_tokens
        inter_token_latency:             distribution of ALL inter-SSE-event gaps
                                         (includes sub-microsecond paired events,
                                         so median is very low and p75+ shows
                                         actual decode intervals)
    """

    request_latency: PercentileDistribution = Field(
        description="End-to-end request latency (seconds): end_time - start_time."
    )
    time_to_first_token: PercentileDistribution = Field(
        description="Time to first token (seconds): first SSE event timestamp - start_time."
    )
    time_per_output_token: PercentileDistribution = Field(
        description="Mean time per output token (seconds) per request: "
        "(end_time - first_token_time) / output_tokens."
    )
    normalized_time_per_output_token: PercentileDistribution = Field(
        description="Normalized time per output token (seconds): "
        "request_latency / output_tokens. Includes TTFT."
    )
    inter_token_latency: PercentileDistribution = Field(
        description="Distribution of ALL inter-SSE-event gaps across all requests. "
        "Includes sub-microsecond gaps between paired SSE lines, so median is ~36μs. "
        "Actual decode intervals appear at p75+ (~16ms for this dataset)."
    )


class ThroughputMetrics(BaseModel):
    """Throughput metrics computed over the measurement window."""

    input_tokens_per_sec: float = Field(
        description="Total input (prompt) tokens processed per second."
    )
    output_tokens_per_sec: float = Field(
        description="Total output tokens generated per second."
    )
    total_tokens_per_sec: float = Field(
        description="Total tokens (input + output) per second."
    )
    requests_per_sec: float = Field(
        description="Completed requests per second."
    )


class SuccessMetrics(BaseModel):
    """Metrics for successful requests."""

    count: int = Field(description="Number of successful requests.")
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    prompt_len: PercentileDistribution = Field(
        description="Distribution of prompt lengths in tokens."
    )
    output_len: PercentileDistribution = Field(
        description="Distribution of output lengths in tokens."
    )


class FailureMetrics(BaseModel):
    """Metrics for failed requests.

    When count=0, all distribution fields are null.
    """

    count: int = Field(description="Number of failed requests.")
    request_latency: Optional[PercentileDistribution] = Field(
        None,
        description="Latency distribution for failed requests. Null when count=0.",
    )
    prompt_len: Optional[PercentileDistribution] = Field(
        None,
        description="Prompt length distribution for failed requests. Null when count=0.",
    )


class LoadSummary(BaseModel):
    """Load generation summary statistics.

    Present in both summary and per-stage metrics.
    """

    count: int = Field(
        description="Total number of requests sent in this window."
    )
    schedule_delay: PercentileDistribution = Field(
        description="Distribution of actual-vs-intended send time (seconds). "
        "Positive = sent late; negative = sent early. "
        "Mean ~0.4ms indicates good scheduling accuracy."
    )
    send_duration: Optional[float] = Field(
        None,
        description="Total wall-clock duration of this stage (seconds). "
        "Only present in per-stage metrics, not summary.",
    )
    requested_rate: Optional[float] = Field(
        None,
        description="Target request rate for this stage (req/s). "
        "Only present in per-stage metrics.",
    )
    achieved_rate: Optional[float] = Field(
        None,
        description="Actual achieved request rate (req/s). "
        "Only present in per-stage metrics.",
    )


class LifecycleMetrics(BaseModel):
    """Aggregate lifecycle metrics (summary or per-stage).

    Used for:
        - summary_lifecycle_metrics.json (aggregate over all stages)
        - stage_N_lifecycle_metrics.json (per-stage breakdown)

    Stage files additionally have send_duration, requested_rate, and
    achieved_rate in their load_summary.
    """

    load_summary: LoadSummary
    successes: SuccessMetrics
    failures: FailureMetrics


# =============================================================================
# 5. KV CACHE EVENTS (kv_events.jsonl)
# =============================================================================


class StorageMedium(str, enum.Enum):
    """Storage medium for KV cache blocks."""

    GPU = "GPU"
    """GPU HBM (fastest, primary cache)."""
    CPU = "CPU"
    """System RAM (used for offloading)."""
    DISK = "DISK"
    """NVMe/SSD storage."""
    LMCACHE = "LMCACHE"
    """External LMCache system."""


class ConnectorType(str, enum.Enum):
    """Connector type for remote KV transfers."""

    NIXL = "NIXL"
    """NIXL connector (async RDMA reads)."""
    P2P = "P2P"
    """P2P NCCL connector (sync send/recv)."""
    MOONCAKE = "MOONCAKE"
    """Mooncake connector."""


class EvictionReason(str, enum.Enum):
    """Reason for cache eviction."""

    LRU = "lru"
    """Least recently used eviction."""
    CAPACITY = "capacity"
    """Capacity limit reached."""
    PREEMPTION = "preemption"
    """Request preemption forced eviction."""


class BlockStoredEvent(BaseModel):
    """A new KV cache block was stored (written to cache).

    Wire format (positional array):
        ["BlockStored", hash_chain, parent_hash, token_ids, block_size, lora_id, medium, lora_name]

    The hash_chain is a list of content hashes identifying this block and its
    prefix chain. Each hash is a uint64 integer representing the hash of that
    block's token content combined with its prefix hash chain.

    The token_ids array contains the actual token IDs cached in this block,
    enabling prefix matching and content reconstruction.
    """

    event_type: Literal["BlockStored"] = "BlockStored"
    hash_chain: list[int] = Field(
        description="Content hash chain as uint64 integers. Each hash encodes "
        "the block's tokens + its parent chain. Length = number of blocks in "
        "the prefix chain for this request (can be 1 to 36+ blocks)."
    )
    parent_hash: Optional[int] = Field(
        description="Hash of the parent block in the prefix chain. "
        "Null for the first block in a new prefix group."
    )
    token_ids: list[int] = Field(
        description="Token IDs stored in this block. Length = block_size. "
        "These are the actual vocabulary indices from the tokenizer."
    )
    block_size: int = Field(
        description="Number of tokens per KV cache block (always 16 in this dataset)."
    )
    lora_id: Optional[int] = Field(
        description="LoRA adapter ID if using LoRA. Null in this dataset (no LoRA)."
    )
    medium: str = Field(
        description="Storage medium where block was stored (always 'GPU' in this dataset)."
    )
    lora_name: Optional[str] = Field(
        description="LoRA adapter name. Null in this dataset."
    )


class BlockRemovedEvent(BaseModel):
    """KV cache blocks were removed (evicted from cache).

    Wire format (positional array):
        ["BlockRemoved", hash_chain, medium]

    IMPORTANT: The hash encoding in BlockRemoved differs from BlockStored.
    BlockStored uses uint64 integer hashes, while BlockRemoved uses
    base64-encoded string hashes in most experiments. One experiment
    (llama-2-7b-tp1-reasoning) uses integer hashes in BlockRemoved.
    This appears to be a vLLM serialization inconsistency — both represent
    the same underlying block content hashes.
    """

    event_type: Literal["BlockRemoved"] = "BlockRemoved"
    hash_chain: list[Union[int, str]] = Field(
        description="Content hash chain of removed blocks. Usually base64-encoded "
        "strings (e.g., 'CHHf3i3n5rT/FCJ3W/...'), but occasionally uint64 "
        "integers. Both encode the same underlying block content hashes."
    )
    medium: str = Field(
        description="Storage medium from which blocks were removed (e.g., 'GPU', 'CPU')."
    )


class AllBlocksClearedEvent(BaseModel):
    """All blocks in the cache were cleared.

    Wire format: ["AllBlocksCleared"]

    Not observed in this dataset but documented in the vLLM KV events spec.
    """

    event_type: Literal["AllBlocksCleared"] = "AllBlocksCleared"


class TransferInitiatedEvent(BaseModel):
    """A local DMA transfer between storage tiers was submitted.

    Wire format (positional array):
        ["TransferInitiated", transfer_id, request_id, source_medium, dest_medium, block_count, scheduler_step]

    Typically GPU→CPU offloading. Each initiated event will have a corresponding
    TransferCompleted event with the same transfer_id.

    Note: In the observed data, each TransferInitiated event appears twice in the
    same batch (duplicate). This appears to be a vLLM publisher behavior — consumers
    should deduplicate by (transfer_id, scheduler_step).
    """

    event_type: Literal["TransferInitiated"] = "TransferInitiated"
    transfer_id: int = Field(
        description="Globally unique transfer ID. Encoding: (emitter_rank << 32) | local_counter. "
        "Use to pair with TransferCompleted events."
    )
    request_id: str = Field(
        description="Request ID that owns the blocks being transferred. "
        "Format: 'cmpl-<uuid>-<seq>-<hash>' matching journey trace gen_ai.request.id."
    )
    source_medium: str = Field(
        description="Source storage medium (e.g., 'GPU')."
    )
    dest_medium: str = Field(
        description="Destination storage medium (e.g., 'CPU')."
    )
    block_count: int = Field(
        description="Number of KV cache blocks being transferred."
    )
    scheduler_step: int = Field(
        description="Scheduler step at which the transfer was initiated. "
        "Correlates with step.id in step tracing and scheduler.step in journey tracing."
    )


class TransferCompletedEvent(BaseModel):
    """A local DMA transfer completed.

    Wire format (positional array):
        ["TransferCompleted", transfer_id, request_id, source_medium, dest_medium, block_count, success, scheduler_step]

    Note: Like TransferInitiated, appears duplicated in batches. The success
    field has always been true in the observed data.
    """

    event_type: Literal["TransferCompleted"] = "TransferCompleted"
    transfer_id: int = Field(
        description="Transfer ID matching the corresponding TransferInitiated event."
    )
    request_id: str = Field(
        description="Request ID that owns the transferred blocks."
    )
    source_medium: str = Field(
        description="Source storage medium."
    )
    dest_medium: str = Field(
        description="Destination storage medium."
    )
    block_count: int = Field(
        description="Number of blocks transferred."
    )
    success: bool = Field(
        description="Whether the transfer completed successfully. "
        "Always true in the observed data."
    )
    scheduler_step: int = Field(
        description="Scheduler step at which transfer was completed. "
        "May differ from the initiated step if the transfer spanned multiple steps."
    )


class RemoteTransferInitiatedEvent(BaseModel):
    """A cross-machine RDMA transfer was initiated.

    Wire format (positional array):
        ["RemoteTransferInitiated", transfer_id, request_id, connector_type,
         source_rank, dest_rank, block_count]

    Not observed in this dataset (single-machine experiments) but documented
    in the vLLM KV events spec for disaggregated/P-D setups.
    """

    event_type: Literal["RemoteTransferInitiated"] = "RemoteTransferInitiated"
    transfer_id: int
    request_id: str
    connector_type: str = Field(description="Connector type: 'NIXL', 'P2P', or 'MOONCAKE'.")
    source_rank: int
    dest_rank: int
    block_count: int


class RemoteTransferCompletedEvent(BaseModel):
    """A cross-machine RDMA transfer completed.

    Wire format (positional array):
        ["RemoteTransferCompleted", transfer_id, request_id, connector_type,
         source_rank, dest_rank, block_count, success]

    Not observed in this dataset.
    """

    event_type: Literal["RemoteTransferCompleted"] = "RemoteTransferCompleted"
    transfer_id: int
    request_id: str
    connector_type: str
    source_rank: int
    dest_rank: int
    block_count: int
    success: bool


class CacheLoadCommittedEvent(BaseModel):
    """Scheduler committed to loading blocks from a cache tier.

    Wire format (positional array):
        ["CacheLoadCommitted", request_id, medium, block_count, scheduler_step]

    This means the scheduler decided to reload blocks from CPU/DISK back to GPU,
    typically after a preemption or to restore offloaded context.
    """

    event_type: Literal["CacheLoadCommitted"] = "CacheLoadCommitted"
    request_id: str = Field(
        description="Request ID whose blocks are being loaded back."
    )
    medium: str = Field(
        description="Source medium to load from (e.g., 'CPU')."
    )
    block_count: int = Field(
        description="Number of blocks to load."
    )
    scheduler_step: int = Field(
        description="Scheduler step at which the load was committed."
    )


class CacheStoreCommittedEvent(BaseModel):
    """Scheduler committed to storing (offloading) blocks to a cache tier.

    Wire format (positional array):
        ["CacheStoreCommitted", request_id, medium, block_count, scheduler_step]

    This is the scheduler's decision to offload blocks from GPU to CPU/DISK.
    The actual data movement happens via TransferInitiated/TransferCompleted.
    """

    event_type: Literal["CacheStoreCommitted"] = "CacheStoreCommitted"
    request_id: str = Field(
        description="Request ID whose blocks are being offloaded."
    )
    medium: str = Field(
        description="Destination medium to store to (e.g., 'CPU', 'DISK')."
    )
    block_count: int = Field(
        description="Number of blocks to store/offload."
    )
    scheduler_step: int = Field(
        description="Scheduler step at which the store was committed."
    )


class CacheEvictionEvent(BaseModel):
    """Blocks were evicted from a cache tier.

    Wire format (positional array):
        ["CacheEviction", medium, blocks_evicted, reason, scheduler_step, extra]

    Eviction frees blocks to make room for new allocations. The reason
    indicates why eviction was triggered.
    """

    event_type: Literal["CacheEviction"] = "CacheEviction"
    medium: str = Field(
        description="Storage medium where eviction occurred (e.g., 'CPU')."
    )
    blocks_evicted: int = Field(
        description="Number of blocks evicted."
    )
    reason: str = Field(
        description="Eviction reason: 'lru', 'capacity', or 'preemption'."
    )
    scheduler_step: int = Field(
        description="Scheduler step at which eviction occurred."
    )
    extra: Optional[Any] = Field(
        None,
        description="Reserved field. Always null in the observed data.",
    )


# Union of all KV event types for parsing
KVEvent = Annotated[
    Union[
        BlockStoredEvent,
        BlockRemovedEvent,
        AllBlocksClearedEvent,
        TransferInitiatedEvent,
        TransferCompletedEvent,
        RemoteTransferInitiatedEvent,
        RemoteTransferCompletedEvent,
        CacheLoadCommittedEvent,
        CacheStoreCommittedEvent,
        CacheEvictionEvent,
    ],
    Field(discriminator="event_type"),
]


class KVEventBatch(BaseModel):
    """A batch of KV cache events from one scheduler step.

    Wire format (positional array, one JSON line in kv_events.jsonl):
        [timestamp, [event1, event2, ...], data_parallel_rank, scheduler_step]

    Events within a batch share the same timestamp. Multiple events (e.g.,
    BlockStored + CacheStoreCommitted + TransferInitiated) are commonly batched
    together in a single line.

    Note on duplicates: TransferInitiated and TransferCompleted events frequently
    appear twice within the same batch. This is a vLLM publisher behavior (likely
    from dual DP ranks or duplicate publish). Consumers should deduplicate by
    (transfer_id, event_type).
    """

    ts: float = Field(
        description="Unix timestamp (seconds with microsecond precision) when this "
        "batch was created by the KV event publisher."
    )
    events: list[Any] = Field(
        description="List of KV events. Each event is a positional array where "
        "the first element is the event type string. See individual event types "
        "for the positional field layout."
    )
    data_parallel_rank: Optional[int] = Field(
        description="Data parallel rank that emitted this batch. "
        "Always 0 in this dataset (single DP rank). Null is possible."
    )
    scheduler_step: Optional[int] = Field(
        description="Scheduler step associated with this batch. "
        "Null for the first batch(es) before the scheduler starts stepping. "
        "Correlates with step.id in step tracing."
    )


# =============================================================================
# 6. OTEL TRACES (traces.json) — Journey + Step Tracing
# =============================================================================
#
# traces.json is JSONL where each line is an OTLP JSON export object.
# Three OTEL scopes are present:
#   - vllm.scheduler.step  : Step-level batch summaries + request snapshots
#   - vllm.scheduler       : Journey tracing core spans (llm_core)
#   - vllm.api             : Journey tracing API spans (llm_request)
# =============================================================================


class OTELAttributeValue(BaseModel):
    """OTLP attribute value (one of stringValue, intValue, doubleValue, boolValue).

    OTLP encodes all values as strings in JSON. Integers appear as
    {"intValue": "42"} (string-encoded), doubles as {"doubleValue": 0.5}.
    """

    stringValue: Optional[str] = None
    intValue: Optional[str] = Field(
        None,
        description="Integer value encoded as string (OTLP convention). Parse with int().",
    )
    doubleValue: Optional[float] = None
    boolValue: Optional[bool] = None


class OTELAttribute(BaseModel):
    """A single OTLP key-value attribute."""

    key: str
    value: OTELAttributeValue


class OTELEvent(BaseModel):
    """An OTLP span event (point-in-time annotation on a span).

    Events carry the actual tracing data — journey lifecycle transitions
    and step-level metrics are all encoded as span events.
    """

    timeUnixNano: str = Field(
        description="Event timestamp in nanoseconds since Unix epoch (string-encoded uint64)."
    )
    name: str = Field(
        description="Event name. One of: "
        "'step.BATCH_SUMMARY', 'step.REQUEST_SNAPSHOT', "
        "'journey.QUEUED', 'journey.SCHEDULED', 'journey.FIRST_TOKEN', "
        "'journey.PREEMPTED', 'journey.FINISHED', "
        "'api.ARRIVED', 'api.DEPARTED', 'api.ABORTED'."
    )
    attributes: list[OTELAttribute] = Field(
        default_factory=list,
        description="Event-specific attributes. See step tracing and journey tracing sections.",
    )


class OTELSpan(BaseModel):
    """An OTLP span representing a unit of work.

    Span types by scope:
        vllm.scheduler.step: name='scheduler_steps_N', kind=1 (INTERNAL)
            Contains step.BATCH_SUMMARY and step.REQUEST_SNAPSHOT events.
            Attributes: scheduler.span_sequence, scheduler.step_range_start,
                        scheduler.step_range_end, scheduler.closure_interval.

        vllm.scheduler: name='llm_core', kind=1 (INTERNAL)
            One span per traced request's engine lifecycle.
            Contains journey.QUEUED/SCHEDULED/FIRST_TOKEN/PREEMPTED/FINISHED events.
            Attribute: gen_ai.request.id (includes sequence suffix, e.g.,
                       'cmpl-xxx-0-yyy').
            Has parentSpanId linking to the llm_request API span.

        vllm.api: name='llm_request', kind=2 (SERVER)
            One span per traced request's API lifecycle.
            Contains api.ARRIVED/DEPARTED/ABORTED events.
            Attributes: gen_ai.request.id (base ID without suffix, e.g., 'cmpl-xxx'),
                        http.route, gen_ai.response.model, gen_ai.usage.prompt_tokens,
                        gen_ai.request.temperature, gen_ai.request.top_p,
                        gen_ai.request.max_tokens, gen_ai.request.n.
    """

    traceId: str = Field(description="128-bit trace ID as 32-char hex string.")
    spanId: str = Field(description="64-bit span ID as 16-char hex string.")
    parentSpanId: Optional[str] = Field(
        None,
        description="Parent span ID. Present on llm_core spans (child of llm_request). "
        "Absent on llm_request spans (root) and scheduler_steps spans.",
    )
    flags: Optional[int] = Field(
        None,
        description="W3C trace flags. 256=sampled (OTEL convention).",
    )
    name: str = Field(
        description="Span name: 'llm_request', 'llm_core', or 'scheduler_steps_N'."
    )
    kind: int = Field(
        description="Span kind per OTLP: 1=INTERNAL, 2=SERVER."
    )
    startTimeUnixNano: str = Field(
        description="Span start time in nanoseconds since Unix epoch (string-encoded uint64)."
    )
    endTimeUnixNano: str = Field(
        description="Span end time in nanoseconds since Unix epoch (string-encoded uint64)."
    )
    attributes: list[OTELAttribute] = Field(
        default_factory=list,
    )
    events: list[OTELEvent] = Field(
        default_factory=list,
    )
    status: Optional[dict] = Field(
        default_factory=dict,
        description="Span status. Empty dict ({}) for OK status.",
    )


class OTELScope(BaseModel):
    """OTEL instrumentation scope identifying the tracer."""

    name: str = Field(
        description="Scope name: 'vllm.scheduler.step', 'vllm.scheduler', or 'vllm.api'."
    )


class OTELScopeSpans(BaseModel):
    """Spans grouped by instrumentation scope."""

    scope: OTELScope
    spans: list[OTELSpan]


class OTELResource(BaseModel):
    """OTEL resource attributes identifying the service."""

    attributes: list[OTELAttribute] = Field(
        description="Resource attributes. Includes telemetry.sdk.language ('python'), "
        "telemetry.sdk.name ('opentelemetry'), telemetry.sdk.version, "
        "service.name ('vllm-inference-perf')."
    )


class OTELResourceSpans(BaseModel):
    """A group of spans sharing the same resource."""

    resource: OTELResource
    scopeSpans: list[OTELScopeSpans]


class OTELTraceExport(BaseModel):
    """One line of traces.json — an OTLP JSON export object.

    traces.json is JSONL format where each line is one of these objects.
    A single line may contain spans from multiple scopes (step + journey + API).
    """

    resourceSpans: list[OTELResourceSpans]


# =============================================================================
# 6a. PARSED STEP TRACING EVENTS (convenience types)
# =============================================================================
#
# These types represent the logical content of step tracing events after
# extracting from the OTLP attribute encoding. They are NOT directly
# deserialized from the file — they must be constructed by parsing the
# OTELEvent.attributes list.
# =============================================================================


class StepBatchSummary(BaseModel):
    """Parsed step.BATCH_SUMMARY event attributes.

    Represents aggregate scheduler metrics for one scheduler step.
    Not directly deserializable from traces.json — must be constructed
    by parsing OTELEvent.attributes where event name == 'step.BATCH_SUMMARY'.

    Relationships (not guaranteed as invariants, but typically hold):
        batch.prefill_tokens + batch.decode_tokens == batch.scheduled_tokens
        kv.blocks_free_gpu <= kv.blocks_total_gpu
        0.0 <= kv.usage_gpu_ratio <= 1.0
    """

    step_id: int = Field(description="Monotonically increasing scheduler step counter (from step.id).")
    step_ts_start_ns: int = Field(description="Step start timestamp (monotonic nanoseconds, from step.ts_start_ns).")
    step_ts_end_ns: int = Field(description="Step end timestamp (monotonic nanoseconds, from step.ts_end_ns).")
    step_duration_us: int = Field(description="Step duration in microseconds (from step.duration_us).")

    queue_running_depth: int = Field(description="Requests currently being processed on GPU (from queue.running_depth).")
    queue_waiting_depth: int = Field(description="Requests waiting in queue for scheduling (from queue.waiting_depth).")

    batch_num_prefill_reqs: int = Field(description="Requests in prefill phase this step (from batch.num_prefill_reqs).")
    batch_num_decode_reqs: int = Field(description="Requests in decode phase this step (from batch.num_decode_reqs).")
    batch_scheduled_tokens: int = Field(description="Total tokens scheduled this step (from batch.scheduled_tokens).")
    batch_prefill_tokens: int = Field(description="Prefill tokens scheduled this step (from batch.prefill_tokens).")
    batch_decode_tokens: int = Field(description="Decode tokens scheduled this step (from batch.decode_tokens).")
    batch_num_finished: int = Field(description="Requests that completed this step (from batch.num_finished).")
    batch_num_preempted: int = Field(description="Requests that were preempted this step (from batch.num_preempted).")

    kv_usage_gpu_ratio: float = Field(description="GPU KV cache usage ratio [0.0, 1.0] (from kv.usage_gpu_ratio).")
    kv_blocks_total_gpu: int = Field(description="Total GPU KV cache blocks available (from kv.blocks_total_gpu).")
    kv_blocks_free_gpu: int = Field(description="Free GPU KV cache blocks remaining (from kv.blocks_free_gpu).")


class StepRequestSnapshot(BaseModel):
    """Parsed step.REQUEST_SNAPSHOT event attributes.

    Represents per-request detail at a specific scheduler step.
    Only emitted when rich subsampling is enabled (two-stage sampling).

    Present in all 16 experiments in this dataset.
    """

    step_id: int = Field(description="Scheduler step for correlation with batch summary (from step.id).")
    request_id: str = Field(description="Unique request identifier (from request.id).")
    request_phase: str = Field(description="Current phase: 'PREFILL' or 'DECODE' (from request.phase).")
    request_num_prompt_tokens: int = Field(description="Total prompt tokens for this request (from request.num_prompt_tokens).")
    request_num_computed_tokens: int = Field(description="Tokens computed so far (prompt + output) (from request.num_computed_tokens).")
    request_num_output_tokens: int = Field(description="Output tokens generated so far (from request.num_output_tokens).")
    request_num_preemptions: int = Field(description="Number of times this request was preempted (from request.num_preemptions).")
    request_scheduled_tokens_this_step: int = Field(description="Tokens scheduled for this request this step (from request.scheduled_tokens_this_step).")
    kv_blocks_allocated_gpu: int = Field(description="GPU KV blocks allocated to this request (from kv.blocks_allocated_gpu).")
    kv_blocks_cached_gpu: int = Field(description="GPU KV blocks from prefix cache hits (from kv.blocks_cached_gpu).")
    request_effective_prompt_len: Optional[int] = Field(
        None,
        description="Effective prompt length after prefix cache reduction (from request.effective_prompt_len). "
        "Only present when prefix caching is enabled and cache was used.",
    )


# =============================================================================
# 6b. PARSED JOURNEY TRACING EVENTS (convenience types)
# =============================================================================


class JourneyEventBase(BaseModel):
    """Common attributes shared by all journey core events.

    All journey events (QUEUED, SCHEDULED, FIRST_TOKEN, PREEMPTED, FINISHED)
    carry these base attributes.
    """

    event_type: str = Field(description="Event type name (from event.type attribute).")
    ts_monotonic: float = Field(
        description="High-precision monotonic timestamp in seconds (from ts.monotonic). "
        "Use for latency calculations between events."
    )
    ts_monotonic_ns: int = Field(
        description="Same timestamp in nanoseconds (from ts.monotonic_ns). "
        "Higher precision than ts_monotonic for sub-microsecond deltas."
    )
    scheduler_step: int = Field(
        description="Scheduler step number (from scheduler.step). "
        "Correlates with step.id in step traces and scheduler_step in KV events."
    )
    phase: str = Field(
        description="Request phase: 'WAITING' (queued), 'PREFILL' (processing prompt), "
        "or 'DECODE' (generating output)."
    )
    prefill_done_tokens: int = Field(description="Prompt tokens processed so far (from prefill.done_tokens).")
    prefill_total_tokens: int = Field(description="Total prompt tokens for this request (from prefill.total_tokens).")
    decode_done_tokens: int = Field(description="Output tokens generated so far (from decode.done_tokens).")
    decode_max_tokens: int = Field(description="Maximum output tokens requested (from decode.max_tokens).")
    num_preemptions: int = Field(description="Cumulative preemption count for this request (from num_preemptions).")


class JourneyQueuedEvent(JourneyEventBase):
    """Request was added to the scheduler waiting queue.

    Event name: journey.QUEUED
    Phase: WAITING
    """

    pass


class JourneyScheduledEvent(JourneyEventBase):
    """Request was allocated GPU resources and scheduled for execution.

    Event name: journey.SCHEDULED
    Phase: PREFILL (first schedule) or DECODE (resume after preemption)
    """

    schedule_kind: str = Field(
        description="'FIRST' for initial scheduling, 'RESUME' after preemption (from schedule.kind)."
    )


class JourneyFirstTokenEvent(JourneyEventBase):
    """First output token was generated (prefill complete).

    Event name: journey.FIRST_TOKEN
    Phase: DECODE
    prefill_done_tokens == prefill_total_tokens at this point.
    """

    pass


class JourneyPreemptedEvent(JourneyEventBase):
    """Request was preempted — GPU resources reclaimed.

    Event name: journey.PREEMPTED
    Phase: PREFILL or DECODE

    Only observed in the llama-2-7b-tp1-reasoning experiment in this dataset.
    After preemption, a SCHEDULED event with kind=RESUME follows when the
    request is rescheduled.
    """

    pass


class JourneyFinishedEvent(JourneyEventBase):
    """Request completed in the scheduler.

    Event name: journey.FINISHED
    Phase: DECODE
    """

    finish_status: str = Field(
        description="Terminal status: 'length' (hit max_tokens), 'stopped' (EOS/stop token), "
        "'aborted' (client disconnect), 'ignored', 'error' (from finish.status)."
    )


class APIArrivedEvent(BaseModel):
    """Request arrived at the API server.

    Event name: api.ARRIVED
    """

    ts_monotonic: float = Field(
        description="Monotonic timestamp in seconds (from event.ts.monotonic)."
    )


class APIDepartedEvent(BaseModel):
    """Response fully sent to client.

    Event name: api.DEPARTED
    """

    ts_monotonic: float = Field(
        description="Monotonic timestamp in seconds (from event.ts.monotonic)."
    )


class APIAbortedEvent(BaseModel):
    """Request was aborted (error or client disconnect).

    Event name: api.ABORTED
    Only observed in llama-2-7b-tp1-reasoning experiment.
    """

    ts_monotonic: float = Field(
        description="Monotonic timestamp in seconds (from event.ts.monotonic)."
    )
    reason: str = Field(
        description="Abort reason (e.g., 'client_disconnect'). From 'reason' attribute."
    )


# =============================================================================
# 7. VLLM LOGGING CONFIG (vllm_logging.json)
# =============================================================================


class VLLMLoggingConfig(BaseModel):
    """Python logging configuration for the vLLM server.

    Standard Python logging dictConfig format. Routes vLLM logs to a file
    at INFO level.
    """

    version: int = Field(description="Logging config version. Always 1.")
    disable_existing_loggers: bool = Field(
        description="Whether to disable existing loggers. Always false."
    )
    handlers: dict = Field(
        description="Log handlers. Contains 'file' handler with FileHandler class."
    )
    formatters: dict = Field(
        description="Log formatters. Contains 'simple' with format string."
    )
    loggers: dict = Field(
        description="Logger configurations. Contains 'vllm' logger at INFO level."
    )


# =============================================================================
# 8. EXPERIMENT DIRECTORY STRUCTURE
# =============================================================================


class ExperimentDirectory(BaseModel):
    """Schema for the directory structure of one experiment.

    Each experiment is stored in a directory named:
        YYYYMMDD-HHMMSS-<model_short_name>-tp<N>-<profile>

    Example: 20260218-150304-codellama-34b-tp2-general

    The directory naming convention encodes:
        - Timestamp when the experiment started
        - Shortened model name (hyphens, no slashes)
        - Tensor parallelism degree
        - Workload profile name
    """

    # Root-level files
    exp_config: ExperimentConfig = Field(
        description="exp-config.yaml — vLLM server parameters."
    )
    profile: dict = Field(
        description="profile.yaml — inference-perf workload profile as single-line JSON. "
        "Parse the YAML, then the inner JSON to get the full profile."
    )
    vllm_logging: VLLMLoggingConfig = Field(
        description="vllm_logging.json — Python logging configuration."
    )
    kv_events: list[KVEventBatch] = Field(
        description="kv_events.jsonl — KV cache event stream. "
        "76K+ lines for a typical 1200s experiment."
    )
    traces: list[OTELTraceExport] = Field(
        description="traces.json — OTEL traces in OTLP JSON format. "
        "~484 lines containing step traces, journey core spans, and API spans."
    )

    # Results subdirectory
    results_config: InferencePerfConfig = Field(
        description="results/config.yaml — fully resolved inference-perf configuration."
    )
    per_request_metrics: list[PerRequestMetrics] = Field(
        description="results/per_request_lifecycle_metrics.json — "
        "per-request timing data. ~16,800 entries for a typical experiment."
    )
    summary_metrics: LifecycleMetrics = Field(
        description="results/summary_lifecycle_metrics.json — "
        "aggregate metrics across all stages."
    )
    stage_metrics: list[LifecycleMetrics] = Field(
        description="results/stage_N_lifecycle_metrics.json — "
        "per-stage breakdown. Typically 2 stages (some experiments have 1)."
    )


# =============================================================================
# 9. FULL DATASET
# =============================================================================


class TrainingDataset(BaseModel):
    """The complete training dataset: 16 experiments across 4 models × 4 profiles.

    Models:
        - llama-2-7b (TP=1)      — small decoder-only, single GPU
        - llama-2-70b (TP=4)     — large decoder-only, 4-GPU TP
        - mixtral-8x7b-v0.1 (TP=2) — MoE architecture, 2-GPU TP
        - codellama-34b (TP=2)   — code-specialized, 2-GPU TP

    Profiles:
        - general   — shared-prefix multi-turn chat
        - codegen   — code generation workload
        - roleplay  — roleplay conversation workload
        - reasoning — reasoning task workload (only one with preemptions/aborts)

    Each experiment runs two load stages:
        - Stage 0: warm-up at lower rate (e.g., 8 RPS × 600s)
        - Stage 1: steady-state at higher rate (e.g., 20 RPS × 600s)
        (Some experiments have only stage 0.)
    """

    experiments: dict[str, ExperimentDirectory] = Field(
        description="Map of experiment directory name to its contents. "
        "Key format: 'YYYYMMDD-HHMMSS-<model>-tp<N>-<profile>'."
    )


# =============================================================================
# 10. GPU HARDWARE SPECIFICATIONS
# =============================================================================
#
# Datasheet values from vendor-published specifications.
# One JSON file per GPU variant lives in datasheets/.
# =============================================================================


class ComputeSpecs(BaseModel):
    """Peak compute throughput at various precisions.

    All values in teraFLOPS (1e12 FLOP/s), as printed on the vendor
    datasheet.  Tensor Core entries marked with_sparsity=True include
    2:4 structured sparsity; divide by 2 for dense throughput.
    """

    fp64: float = Field(description="FP64 CUDA core peak (TFLOPS).")
    fp64_tensor_core: float = Field(description="FP64 Tensor Core peak (TFLOPS).")
    fp32: float = Field(description="FP32 CUDA core peak (TFLOPS).")
    tf32_tensor_core: float = Field(
        description="TF32 Tensor Core peak (TFLOPS). With sparsity on NVIDIA datasheets."
    )
    bf16_tensor_core: float = Field(
        description="BF16 Tensor Core peak (TFLOPS). With sparsity on NVIDIA datasheets."
    )
    fp16_tensor_core: float = Field(
        description="FP16 Tensor Core peak (TFLOPS). With sparsity on NVIDIA datasheets."
    )
    fp8_tensor_core: float = Field(
        description="FP8 Tensor Core peak (TFLOPS). With sparsity on NVIDIA datasheets."
    )
    int8_tensor_core: float = Field(
        description="INT8 Tensor Core peak (TOPS). With sparsity on NVIDIA datasheets."
    )
    with_sparsity: bool = Field(
        description="Whether Tensor Core values include 2:4 structured sparsity. "
        "If true, divide Tensor Core values by 2 for dense throughput."
    )


class MemorySpecs(BaseModel):
    """GPU memory capacity and bandwidth."""

    capacity_gb: float = Field(description="Total GPU memory (GB). E.g., 80 for H100 SXM.")
    bandwidth_tb_s: float = Field(
        description="Peak HBM bandwidth (TB/s). E.g., 3.35 for H100 SXM."
    )


class InterconnectSpecs(BaseModel):
    """Inter-GPU and host interconnect specifications."""

    nvlink_bandwidth_gb_s: float = Field(
        description="NVLink bidirectional bandwidth per GPU (GB/s). "
        "900 for H100 SXM, 600 for H100 NVL."
    )
    pcie_bandwidth_gb_s: float = Field(
        description="PCIe bandwidth (GB/s). 128 for PCIe Gen5."
    )


class PowerSpecs(BaseModel):
    """Thermal design power."""

    tdp_watts: float = Field(
        description="Maximum thermal design power (watts). "
        "700 for H100 SXM, 350-400 for H100 NVL."
    )


class DatasheetSpec(BaseModel):
    """GPU specifications extracted from the vendor datasheet.

    One instance per GPU variant (e.g., H100 SXM, H100 NVL).
    All values are peak/theoretical from the published datasheet.

    Source: datasheets/<vendor>-<gpu>-datasheet-<id>.pdf
    """

    name: str = Field(description="GPU variant name. E.g., 'H100 SXM', 'H100 NVL'.")
    vendor: str = Field(description="GPU vendor. E.g., 'NVIDIA'.")
    architecture: str = Field(description="GPU architecture. E.g., 'Hopper'.")
    compute: ComputeSpecs
    memory: MemorySpecs
    interconnect: InterconnectSpecs
    power: PowerSpecs
    mig_partitions: Optional[int] = Field(
        None,
        description="Maximum number of MIG partitions. 7 for H100.",
    )
    form_factor: Optional[str] = Field(
        None,
        description="Physical form factor. E.g., 'SXM', 'PCIe dual-slot air-cooled'.",
    )
    datasheet_id: Optional[str] = Field(
        None,
        description="Vendor datasheet document ID for traceability. "
        "E.g., '2430615' from nvidia-h100-datasheet-2430615.pdf.",
    )
