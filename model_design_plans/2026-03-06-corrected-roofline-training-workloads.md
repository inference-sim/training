# Workloads for Corrected Roofline β/α Fitting

**Companion to:** `2026-03-06-corrected-roofline-features-design.md` §12
**Infrastructure:** [tektonc-data-collection](https://github.com/inference-sim/tektonc-data-collection) + [instrumented vLLM](https://github.com/inference-sim/vllm)

---

## W1: Prefill Length Sweep

**Targets:** β₁, β₇ | **Sweep:** `question_len` | **output_len=1** (pure prefill)

```yaml
# Run 4 times with question_len ∈ {128, 1024, 4096, 8192}
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 0.5
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 128       # SWEEP: {128, 1024, 4096, 8192}
    output_len: 1
```

---

## W2: Decode Context Sweep

**Targets:** β₂, β₇ | **Sweep:** `question_len` | **output_len=512** (decode-dominated)

```yaml
# Run 4 times with question_len ∈ {128, 1024, 4096, 8192}
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 0.5
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 128       # SWEEP: {128, 1024, 4096, 8192}
    output_len: 512
```

---

## W3: Batch Size Scaling

**Targets:** β₂ vs β₆, β₄ | **Sweep:** `rate` | **Fixed:** question_len=256, output_len=256

```yaml
# Run 5 times with rate ∈ {1, 8, 32, 64, 128}
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 8.0             # SWEEP: {1, 8, 32, 64, 128}
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 256
    output_len: 256
```

---

## W4: Prefill/Decode Ratio Mix

**Targets:** β₁ vs β₂ | **2 profiles** — prefill-heavy (A) and decode-heavy (C)

```yaml
# Profile A: Prefill-heavy
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 128.0
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 1024
    output_len: 4
```

```yaml
# Profile C: Decode-heavy
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 32.0
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 4
    num_users_per_system_prompt: 4
    system_prompt_len: 0
    question_len: 64
    output_len: 1024
```

---

## W5: TP Communication Scaling

**Targets:** β₅ | **Same as W3, run at TP=1, TP=2, and TP=4** | Subsumed by running W3 at each TP config.

---

## V1: Validation Sweep

**Purpose:** Held-out model evaluation | **3 runs per model** covering decode + batch scaling

```yaml
# Run 1: Decode at moderate context (W2 at question_len=1024)
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 0.5
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 1024
    output_len: 512
```

```yaml
# Runs 2–3: Batch scaling (W3 at rate=1 and rate=64)
load:
  type: constant
  base_seed: 42
  stages:
    - rate: 1.0             # SWEEP: {1, 64}
      duration: 120
api:
  type: completion
server:
  type: vllm
  base_url: http://0.0.0.0:8000
  ignore_eos: true
data:
  type: shared_prefix
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 0
    question_len: 256
    output_len: 256
```

---

## Summary

| Profile | Sweep | Runs | β targets |
|---------|-------|:----:|-----------|
| **Training** | | | |
| W1: Prefill sweep | 4 question_len, output_len=1 | 4 | β₁, β₇ |
| W2: Decode context | 4 question_len, output_len=512 | 4 | β₂, β₇ |
| W3: Batch scaling | 5 rates, question_len=256 | 5 | β₂ vs β₆, β₄ |
| W4: Pf/dc ratio | 2 profiles (A/C) | 2 | β₁ vs β₂ |
| W5: TP scaling | subsumed by W3 across TP | — | β₅ |
| | **Per training (model, TP) combo** | **15** | |
| **Validation** | | | |
| V1: Validation sweep | 1× W2 + 2× W3 | 3 | Cross-model accuracy |
| | **Per validation (model, TP) combo** | **3** | |
