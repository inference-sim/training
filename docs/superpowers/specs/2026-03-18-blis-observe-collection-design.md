# Design: BLIS Observe Data Collection Pipeline

## Summary

Fork the existing `blis-inference-perf` campaign pipeline to collect training data using `blis observe` instead of `inference-perf`, with the instrumented vLLM (`values-observability.yaml`) always enabled. Same pipeline structure, same Tekton tasks, same TP×DLP loop.

**Two separate pipelines** to keep data clean:
1. **Calibration pipeline** — instrumented vLLM (same as collection), measures `decode_ms_per_token`, caches to PVC, tears down
2. **Data collection pipeline** — instrumented vLLM, fresh server, reads cached calibration, runs `blis observe`, collects all three data streams

**Scope:** Phase 1 only (38 experiments). Post-processing and Phase 2 out of scope.

**Branch strategy:** All tektonc-data-collection changes on a separate branch. Campaign data never touched.

## What Changes vs Campaign

| Component | Campaign | Training: Calibration | Training: Collection |
|-----------|----------|----------------------|---------------------|
| Install task | `install-inference-perf-blis` | — (not needed) | `install-blis` **NEW** |
| Run workload | `run-workload-inference-perf-blis` | `calibrate-decode-latency` (existing!) | `run-blis-observe` **NEW** |
| Base values | `values.yaml` (stock vLLM) | `values-observability.yaml` (instrumented) | `values-observability.yaml` (instrumented) |
| Tracing | `{% if stack.tracing %}` (optional) | Always on (measures real overhead) | Always on |
| KV events collection | — | — | `collect-kv-events` (existing!) |
| Workloads | `workloads.yaml` | — | `training-workloads.yaml` (W1-W6) |

Both pipelines deploy the instrumented vLLM so that calibration measures decode latency under the same runtime conditions (OTEL overhead, KV subscriber sidecar) as collection. The calibration pipeline's trace/KV data on the PVC is simply overwritten when the collection pipeline runs on a fresh server.

## Three Data Streams (collection pipeline only)

| Stream | Source | Format | Mechanism |
|--------|--------|--------|-----------|
| Client trace | `blis observe` | TraceV2 (YAML header + CSV) | blis CLI writes to PVC |
| Journey traces | vLLM `--enable-journey-tracing` | OTEL JSON | OTEL collector file exporter → PVC |
| KV events | vLLM KV events publisher | JSONL | ZMQ subscriber sidecar → PVC |

## Pipeline 1: Calibration

Deploys with the **instrumented vLLM** (same image, OTEL, KV sidecar as collection) so that calibration measures decode latency under real runtime conditions. Runs the existing `calibrate-decode-latency` task, caches `decode_ms_per_token` to PVC, tears down.

### Task Flow

```
download-model
    |
[per TP x DLP stack] ─────────────────────────────────
    |                                                   |
create-exp-config                              create-otel-collector
    |                                                   |
    └──────────────── deploy-model ─────────────────────┘
                          |
                   calibrate-decode-latency (existing task)
                          |
              ┌───────────┴───────────┐
        delete-model          delete-otel-collector
```

Same deploy-model overrides as collection (tracing, KV events). The `calibrate-decode-latency` task already exists — it sends one streaming request via Python urllib, measures inter-token timing, and writes `calibration.json` to the data workspace. Trace/KV data from the calibration request is left on the PVC but gets overwritten by the collection pipeline.

### Template: `tektoncsample/blis-observe-calibrate/data_pipeline.yaml.j2`

Fork of `blis-inference-perf/data_pipeline.yaml.j2`:
- Remove `install-inference-perf` (not needed)
- Replace `run-workload-inference-perf-blis` with `calibrate-decode-latency`
- Keep OTEL collector tasks and tracing overrides (same as collection)

### Values: `tektoncsample/blis-observe-calibrate/values.yaml`

Copy of `values-observability.yaml` (instrumented vLLM image `ghcr.io/inference-sim/vllm:0.15.1`, OTEL init containers, KV events sidecar). `stack.tracing` enabled so deploy-model applies all tracing overrides.

### Output on data-pvc

```
<experimentId>-<tp>-<dlp>/
  calibration.json         # {"model": "...", "decode_ms_per_token": 12.345, ...}
  traces.json              # Calibration probe traces (overwritten by collection)
  kv_events.jsonl          # Calibration probe KV events (overwritten by collection)
```

Tekton tasks access via `$(workspaces.data.path)/<results_dir>/...`. The vLLM pod mounts the same PVC at `/mnt/exp`.

## Pipeline 2: Data Collection

Fork of campaign pipeline with instrumented vLLM. Deploys a **fresh server** (no prior requests), reads the cached `calibration.json` from the PVC, runs `blis observe`.

### Task Flow

```
download-model
    |
install-blis (NEW)
    |
[per TP x DLP stack] ─────────────────────────────────
    |                                                   |
create-exp-config                              create-otel-collector
    |                                                   |
    └──────────────── deploy-model ─────────────────────┘
                          |
                   run-blis-observe (NEW — reads cached calibration)
                          |
                   collect-kv-events (existing)
                          |
              ┌───────────┼───────────┐
        delete-model  delete-otel-collector  raw-upload
```

**Key:** No calibration step in this pipeline. The server only sees `blis observe` traffic. All three data streams are clean.

**Ordering:** `collect-kv-events` runs BEFORE `delete-model` — needs `kubectl cp` from the live pod.

### Template diff vs campaign (conceptual)

```diff
-- name: install-inference-perf
-  taskRef: { name: install-inference-perf-blis }
+- name: install-blis
+  taskRef: { name: install-blis }

     # Inside per-stack loop:
-    - name: run-workload-{{ stackId }}
-      taskRef: { name: run-workload-inference-perf-blis }
-      runAfter: [ "deploy-model-{{ stackId }}" ]
-      params:
-        - { name: profileTemplate, value: {{ workload.profileTemplate }} }
+    - name: observe-{{ stackId }}
+      taskRef: { name: run-blis-observe }
+      runAfter: [ "deploy-model-{{ stackId }}" ]
+      params:
+        - { name: results_dir, value: "{{ stackModelLabel }}" }
+        - { name: workload_name, value: "{{ workload.name }}" }
+        - { name: rate_pct, value: "{{ workload.rate_pct }}" }
+        - { name: num_requests, value: "{{ workload.num_requests }}" }
+        - { name: prompt_tokens_mean, value: "{{ workload.prompt_tokens_mean }}" }
+        - { name: prompt_tokens_std, value: "{{ workload.prompt_tokens_std }}" }
+        - { name: output_tokens_mean, value: "{{ workload.output_tokens_mean }}" }
+        - { name: output_tokens_std, value: "{{ workload.output_tokens_std }}" }

+    - name: collect-kv-events-{{ stackId }}
+      taskRef: { name: collect-kv-events }
+      runAfter: [ "observe-{{ stackId }}" ]

     - name: delete-model-{{ stackId }}
-      runAfter: [ "run-workload-{{ stackId }}" ]
+      runAfter: [ "collect-kv-events-{{ stackId }}" ]
```

### Values: `tektoncsample/blis-observe/values.yaml`

Copy of `values-observability.yaml` with the `workload:` section changed:

```diff
-workload:
-  harness: inference-perf
-  profileTemplate:
-    load: { ... }
-    api: { ... }
-    ...
+workload: {}  # Populated by generate.py per experiment:
+              # {name, rate_pct, num_requests,
+              #  prompt_tokens_mean, prompt_tokens_std,
+              #  output_tokens_mean, output_tokens_std}
```

Everything else from `values-observability.yaml` stays: `stack.tracing` (journey, step, kv_events), instrumented vLLM image, OTEL init containers, kv-events-subscriber sidecar.

## New Tekton Tasks (2 only)

### 1. `install-blis`

Builds the Go binary. Used by the data collection pipeline only.

```yaml
apiVersion: tekton.dev/v1
kind: Task
metadata:
  name: install-blis
spec:
  params:
    - name: blisRef
      type: string
      description: Git branch, tag, or SHA to build from
      default: "main"
  workspaces:
    - name: data
  steps:
    - name: build-blis
      image: golang:1.23
      script: |
        TARGET="$(workspaces.data.path)/blis"
        if [ ! -f "$TARGET" ]; then
          cd /tmp
          git clone https://github.com/inference-sim/inference-sim.git blis-src
          cd blis-src
          git checkout $(params.blisRef)
          go build -o "$TARGET" main.go
        fi
        "$TARGET" --version
```

### 2. `run-blis-observe`

Reads cached `calibration.json` from PVC (written by the calibration pipeline), computes the request rate using the protocol formula, runs `blis observe`.

```yaml
apiVersion: tekton.dev/v1
kind: Task
metadata:
  name: run-blis-observe
spec:
  params:
    - name: modelLabel
    - name: model
    - name: namespace
    - name: results_dir
    - name: workload_name
    - name: rate_pct
      description: "Rate as percentage of safe_rps (e.g., 50 = 50%)"
    - name: num_requests
    - name: prompt_tokens_mean
    - name: prompt_tokens_std
    - name: output_tokens_mean
    - name: output_tokens_std
  workspaces:
    - name: data
  timeout: "60m"
  steps:
    - name: resolve-endpoint
      image: alpine/kubectl:1.34.1
      script: |
        POD_IP=$(kubectl -n $(params.namespace) get po \
          -l llm-d.ai/model=$(params.modelLabel),llm-d.ai/role=decode \
          -o jsonpath='{.items[0].status.podIP}')
        echo "http://${POD_IP}:8000" > /workspace/endpoint.txt

    - name: observe
      image: golang:1.23
      script: |
        set -eu
        ENDPOINT=$(cat /workspace/endpoint.txt)

        RESULTS="$(workspaces.data.path)/$(params.results_dir)"

        # Read cached decode_ms_per_token from calibration pipeline
        CAL_FILE="${RESULTS}/calibration.json"
        if [ ! -f "$CAL_FILE" ]; then
          echo "ERROR: calibration.json not found at $CAL_FILE" >&2
          echo "Run the calibration pipeline first." >&2
          exit 1
        fi
        DMS=$(grep -o '"decode_ms_per_token":[^,}]*' "$CAL_FILE" | cut -d: -f2 | tr -d ' ')
        echo "Cached decode_ms_per_token: ${DMS} ms"

        # Read max_num_batched_tokens from exp-config.yaml (written by create-exp-config)
        MBT=$(grep 'max_num_batched_tokens' "${RESULTS}/exp-config.yaml" | awk '{print $2}')

        # Protocol rate formula:
        #   R_seq = 1000 / (decode_ms_per_token * mean_output_tokens)
        #   R_mbt = max_num_batched_tokens / mean_input_tokens / decode_ms_per_token * 1000
        #   safe_rps = 0.5 * min(R_seq, R_mbt)
        #   actual_rate = safe_rps * rate_pct / 100
        RATE=$(awk "BEGIN {
          dms = ${DMS}
          r_seq = 1000.0 / (dms * $(params.output_tokens_mean))
          r_mbt = ${MBT} / $(params.prompt_tokens_mean) / dms * 1000.0
          safe = 0.5 * (r_seq < r_mbt ? r_seq : r_mbt)
          printf \"%.2f\", safe * $(params.rate_pct) / 100.0
        }")

        OUTDIR="$(workspaces.data.path)/$(params.results_dir)/$(params.workload_name)/r$(params.rate_pct)"
        mkdir -p "$OUTDIR"

        echo "Running blis observe at rate=${RATE} rps ($(params.rate_pct)% of safe_rps)"

        "$(workspaces.data.path)/blis" observe \
          --server-url ${ENDPOINT} \
          --model $(params.model) \
          --rate ${RATE} \
          --num-requests $(params.num_requests) \
          --prompt-tokens $(params.prompt_tokens_mean) \
          --prompt-tokens-stdev $(params.prompt_tokens_std) \
          --output-tokens $(params.output_tokens_mean) \
          --output-tokens-stdev $(params.output_tokens_std) \
          --trace-header "$OUTDIR/trace-header.yaml" \
          --trace-data "$OUTDIR/trace-data.csv"
```

## Existing Tasks Used (no changes needed)

| Task | Calibration pipeline | Collection pipeline |
|------|---------------------|---------------------|
| `download-model` | Yes | Yes |
| `create-exp-config` | Yes | Yes |
| `create-otel-collector` / `delete-otel-collector` | Yes | Yes |
| `deploy-model` / `delete-model` | Yes | Yes |
| `calibrate-decode-latency` | Yes | — |
| `collect-kv-events` | — | Yes |
| `upload-s3` | — | Yes |

## Workload Profiles: W1-W6

Stored in `tektonc-data-collection/training-workloads.yaml`. Synthetic workloads for coefficient identifiability — NOT campaign profiles.

```yaml
W1-prefill-heavy:
  prompt_tokens: { mean: 3000, std: 500 }
  output_tokens: { mean: 8, std: 2 }
  num_requests: 200

W2-decode-heavy:
  prompt_tokens: { mean: 32, std: 8 }
  output_tokens: { mean: 1500, std: 300 }
  num_requests: 200

W3-balanced-short:
  prompt_tokens: { mean: 256, std: 64 }
  output_tokens: { mean: 128, std: 32 }
  num_requests: 200

W4-balanced-long:
  prompt_tokens: { mean: 1024, std: 256 }
  output_tokens: { mean: 512, std: 128 }
  num_requests: 200

W5-batch-stressor:
  prompt_tokens: { mean: 64, std: 16 }
  output_tokens: { mean: 32, std: 8 }
  num_requests: 200

W6-kv-pressure:
  prompt_tokens: { mean: 512, std: 128 }
  output_tokens: { mean: 2048, std: 512 }
  num_requests: 200
```

## Training Experiment Schema

Stored in `blis-campaign/training-experiments.json`. Same schema as campaign plus `rate_pct` and `workload` referencing W1-W6.

```json
{
  "id": 1,
  "layer": "L1",
  "model": "Llama-3.1-8b",
  "hw": "H100",
  "tp": 1,
  "dp": 1,
  "precision": "BF16",
  "mbt": 2048,
  "max_model_len": 4096,
  "max_num_seqs": 128,
  "gpu_mem": 0.9,
  "cpu_offload": false,
  "workload": "W1-prefill-heavy",
  "rate_pct": 50,
  "done": false
}
```

38 experiments = 38 pipeline runs (collection), each with its own fresh model server.

## blis-campaign Integration

### Generator changes

`generate.py` gets a `--pipeline` flag with three modes:

```bash
# Step 1: Calibrate (instrumented vLLM, same deploy as collection)
python blis-campaign/generate.py \
  --pipeline calibrate \
  --experiments blis-campaign/training-experiments.json \
  --output output/calibrate/

# Step 2: Collect (instrumented vLLM, clean server)
python blis-campaign/generate.py \
  --pipeline training \
  --experiments blis-campaign/training-experiments.json \
  --output output/training/

# Campaign (unchanged, no --pipeline flag)
python blis-campaign/generate.py \
  --experiments blis-campaign/experiments.json \
  --output output/campaign/
```

When `--pipeline calibrate`:
1. Template: `tektoncsample/blis-observe-calibrate/data_pipeline.yaml.j2`
2. Base values: `tektoncsample/blis-observe-calibrate/values.yaml` (instrumented vLLM)
3. No workload resolution needed — calibration uses fixed probe params

When `--pipeline training`:
1. Template: `tektoncsample/blis-observe/data_pipeline.yaml.j2`
2. Base values: `tektoncsample/blis-observe/values.yaml` (instrumented vLLM)
3. Workloads from `training-workloads.yaml`
4. `build_values()` populates `v["workload"]` with resolved params
5. Everything else unchanged — `build_extra_overrides()`, `resolve_model()` work as-is

### Models requiring additions

```yaml
Mixtral-8x22B:
  hf_id: "mistralai/Mixtral-8x22B-Instruct-v0.1"
```

### Quantization handling

Same as campaign — `build_extra_overrides()` handles it:

| Precision | Method |
|-----------|--------|
| BF16 | No flag (default) |
| FP8 (Llama-4-Scout) | Pre-quantized checkpoint, no flag |
| FP8 (others) | Online `--quantization=fp8` |

### On-cluster data layout

After both pipelines run (paths relative to data-pvc root, accessed via `$(workspaces.data.path)` in tasks and `/mnt/exp` in vLLM pod):

```
<experimentId>-<tp>-<dlp>/
  calibration.json         # From calibration pipeline (instrumented vLLM)
  W1-prefill-heavy/
    r50/
      trace-header.yaml    # From collection pipeline (blis observe)
      trace-data.csv
  traces.json              # From collection pipeline (OTEL journey traces)
  kv_events.jsonl          # From collection pipeline (ZMQ subscriber)
```

## Layer Coverage (38 experiments)

| Layer | Count | What varies | Key overrides |
|-------|-------|------------|---------------|
| L1 Dense Model Diversity | 12 | Different models, TP=1-4 | Standard deploy |
| L2 TP Isolation | 6 | TP=1,2,4,8 | TP × DLP loop |
| L3 MoE + Expert Parallelism | 8 | MoE models, EP via DLP>1 | `--enable-expert-parallel` |
| L4 Quantization | 4 | FP8 only | `--quantization=fp8` or pre-quantized |
| L5 CPU KV Offload | 4 | offload on/off | `--kv-offloading-size=8.0` |
| L6 Rate Sweep | 4 | Rate 30/70/100/130% | Same model, different rate_pct |

## What This Design Does NOT Cover

- Post-processing (saturation filter, step reconstruction, basis functions, fitting)
- Phase 2 adaptive gap-filling
- GPTQ/INT4 experiments
- Campaign data or results
