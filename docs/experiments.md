# Experiments

Experiments are **config-driven** YAML files in `benchmarks/configs/`. A file
describes either a single experiment or a `base` + `sweep` document that expands
(Cartesian product) into many experiments.

## Running

```bash
# Single experiment (real backend: set backend.type/base_url in the YAML)
python -m benchmarks.runner.client --config benchmarks/configs/single_gpu_baseline.yaml --out results/

# No-GPU smoke test with the in-process simulator
python -m benchmarks.runner.client --config benchmarks/configs/single_gpu_baseline.yaml \
    --backend mock --max-seconds 10 --out results/

# A sweep (expands to one run per concurrency level)
python -m benchmarks.runner.client --config benchmarks/configs/concurrency_sweep.yaml --out results/
```

CLI overrides (handy without editing YAML): `--backend`, `--backend-url`,
`--model`, `--concurrency`, `--num-requests`, `--max-seconds`, `--dry-run`.

## Config schema

```yaml
name: <str>                 # required (single-experiment docs)
hardware: A100-80GB         # label only, recorded in results
precision: fp16             # label: fp16|bf16|fp8|int8|int4|awq|gptq
backend:
  type: vllm                # mock|vllm|tgi|triton|lmdeploy|openai
  base_url: http://localhost:8000
  model: meta-llama/Meta-Llama-3-8B-Instruct
  api_style: openai-chat    # openai-chat|openai-completions|triton-generate|mock|auto
  api_key: null             # optional bearer token
  mock_ttft_ms: 40.0        # mock backend only
  mock_itl_ms: 10.0         # mock backend only
  extra: {}                 # merged into the request body
workload:
  prompt_tokens: 512        # synthetic prompt length (≈ words) if prompt_text unset
  output_tokens: 256        # max_tokens requested
  prompt_text: null         # explicit prompt (recommended for rigor)
  temperature: 0.0
  top_p: 1.0
load:
  concurrency: 16           # in-flight requests (closed loop)
  num_requests: null        # fixed count … (takes precedence)
  duration_s: 30            # … or fixed duration
  warmup_requests: 4
  request_timeout_s: 120
notes: ""
```

### Sweep documents

```yaml
name: concurrency_sweep
base:
  precision: fp16
  backend: { type: vllm, base_url: "http://localhost:8000", model: "..." }
  workload: { prompt_tokens: 512, output_tokens: 256 }
  load: { duration_s: 30, warmup_requests: 4 }
sweep:
  load.concurrency: [1, 4, 16, 64]   # dotted keys index into nested fields
  precision: [fp16, fp8]             # multiple keys -> Cartesian product
```

Each expanded run is named `"<name>__key=value__…"`.

### Flat schema (alternative)

You can also use a flat, blueprint-style schema; the loader normalizes it into
the nested form (nested values win if both are provided):

```yaml
name: smoke_vllm
backend: vllm                 # bare string -> backend.type
model: meta-llama/Meta-Llama-3-8B-Instruct
base_url: http://localhost:8000
hardware_hint: A100-80GB      # -> hardware
precision: fp16
prompt_length: 512            # -> workload.prompt_tokens
output_length: 256            # -> workload.output_tokens
concurrency: 16               # -> load.concurrency
duration_seconds: 30          # -> load.duration_s (or use num_requests)
```

### Explicit experiment list

For heterogeneous runs (e.g. a precision sweep where each variant lives on a
different endpoint), use `experiments:` with shared `defaults:` (deep-merged):

```yaml
name: precision_sweep
defaults: { prompt_length: 512, output_length: 256, concurrency: 16, duration_seconds: 30 }
experiments:
  - { name: fp16_vllm,     precision: fp16, backend: vllm, base_url: "http://localhost:8000", model: "..." }
  - { name: int4_awq_vllm, precision: int4, backend: vllm, base_url: "http://localhost:8002", model: "...-AWQ" }
```

### Backend fan-out (comparison)

For apples-to-apples comparisons, list `backends:` plus per-backend endpoints;
each expands into one experiment (see `docs/comparisons.md`):

```yaml
name: backend_comparison
model: meta-llama/Meta-Llama-3-8B-Instruct
precision: fp16
prompt_length: 512
output_length: 256
concurrency: 16
duration_seconds: 30
backends: [vllm, triton_trtllm, lmdeploy, tgi]
backend_urls: { vllm: "http://localhost:8000", tgi: "http://localhost:8080" }
```

## Methodology

- **Warm-up:** `warmup_requests` are issued (and discarded) to prime caches and
  CUDA graphs before measurement.
- **Measurement:** closed-loop load until `num_requests` or `duration_s` is hit.
- Keep prompts/decoding params **identical across backends** to avoid
  confounding comparisons. Prefer setting `workload.prompt_text`.
- Run multiple trials per config to capture variance (re-run with distinct
  `--out` subdirs and aggregate the ledger).

## Metric definitions

| Metric | Definition | Source |
|--------|------------|--------|
| TTFT | submit → first non-empty token | `RequestResult.ttft_s` |
| ITL | mean gap between tokens = (latency − TTFT)/(out_tokens − 1) | `RequestResult.itl_s` |
| Latency p50/p95/p99 | end-to-end latency distribution | `latency_s` in summary |
| Output TPS | Σ output tokens / wall time | `throughput.output_tps` |
| RPS | successful requests / wall time | `throughput.rps` |
| GPU util / VRAM | `nvidia-smi` mean & peak | `gpu` in summary |

## Known approximation (documented deviation)

Output tokens are counted as **streamed chunks**, not via a shared tokenizer.
For most servers one SSE delta ≈ one token, but this is approximate. For
publication-grade numbers, set `workload.prompt_text` and add a shared tokenizer
(e.g. `transformers`/`tokenizers`) to count tokens consistently across backends.
This is tracked as a future improvement and intentionally avoids a heavy
dependency in the core harness.

## Smoke Test: vLLM Baseline

Validates the vLLM path end-to-end (config → load → metrics → summary).

**How to run** (no GPU; simulator):

```bash
python -m benchmarks.runner.client --config benchmarks/configs/smoke_vllm.yaml --backend mock
```

**Real vLLM:**

```bash
docker compose -f infra/docker-compose.yml --profile vllm up -d
python -m benchmarks.runner.client --config benchmarks/configs/smoke_vllm.yaml
```

**What to expect:** a `results/smoke_vllm/summary.json` with non-zero
`throughput.output_tps`, a TTFT in the tens-of-ms range on a warm GPU, and
`counts.errors == 0`. (No absolute numbers are asserted — this only proves
wiring.)

## Smoke Test: Triton + TensorRT-LLM Baseline

Validates the Triton + TensorRT-LLM FP16 path.

```bash
# 1. Build an FP16 engine into the model repo (GPU host with TRT-LLM):
python models/convert/convert_to_trtllm.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dtype float16 \
    --output-dir models/triton_model_repo/llama3-8b-trt/1 --run

# 2. Start Triton:
docker compose -f infra/docker-compose.yml --profile triton up -d

# 3. Benchmark it (text I/O goes through the ensemble model):
python -m benchmarks.runner.client --config benchmarks/configs/smoke_triton_trtllm.yaml
```

`backend: triton_trtllm` is an alias for the canonical `triton` backend (the
`triton-generate` API dialect). The same TTFT/latency/TPS metrics are recorded,
so results are directly comparable to the vLLM smoke test.

## Quantization Sweep

Answers: *how do FP16 / FP8 / INT8 / INT4 trade latency, throughput, and cost?*
(blueprint: *Quantization Recommendations*). Each precision is mapped to the
backend + model id that serves it in `multi_precision_sweep.yaml`.

```bash
# Build the quantized engines/checkpoints (GPU host):
python models/convert/quantize_int8.py --model <hf> --run      # INT8 SmoothQuant
python models/convert/quantize_awq.py  --model <hf> --run      # INT4 AWQ

# Run the sweep (drop --backend mock for real endpoints):
python -m benchmarks.runner.client --config benchmarks/configs/multi_precision_sweep.yaml \
    --backend mock --num-requests 8 --out results/

# Cost per precision (per provider):
python -m cost.cost_analysis --results results/int4_awq_vllm/summary.json --gpu-type H100_80GB
```

**Compare:** output TPS, p95 latency, VRAM, **cost per 1M tokens**, and a
*qualitative* accuracy spot-check (quantization trades accuracy for speed — see
`docs/quantization.md`).

## Experiment Catalog

Each major experiment type, with what it answers and how to read it.

### Baseline (`single_gpu_baseline.yaml`, `smoke_vllm.yaml`)
- **What it answers:** what is a single backend's latency/throughput at a fixed,
  moderate concurrency?
- **How to run:** `python -m benchmarks.runner.client --config benchmarks/configs/single_gpu_baseline.yaml`
- **How to interpret:** establishes the reference point; every other experiment
  is read relative to this.

### Concurrency sweep (`concurrency_sweep.yaml`)
- **What it answers:** how do TTFT, TPS, and p95 scale as in-flight requests
  grow (the payoff of continuous batching)?
- **How to run:** `python -m benchmarks.runner.client --config benchmarks/configs/concurrency_sweep.yaml`
- **How to interpret:** TPS should rise then plateau as the GPU saturates; TTFT
  and p95 climb once queueing dominates. The knee is your throughput-optimal
  operating point — also where cost/token is lowest.

### Precision / quantization sweep (`multi_precision_sweep.yaml`)
- **What it answers:** the speed/cost/accuracy trade-off across precisions.
- **How to interpret:** lower precision should raise TPS and cut cost/token and
  VRAM; weigh against any accuracy regression.

### Backend comparison (`backend_comparison.yaml`)
- **What it answers:** which runtime wins for this model/hardware/workload?
- **How to run:** `python -m benchmarks.runner.run_comparison --config benchmarks/configs/backend_comparison.yaml`
- **How to interpret:** read the generated `results/comparison.md`; compare on
  the *same* axis (e.g. p95 at fixed concurrency) and pair with cost/token.
