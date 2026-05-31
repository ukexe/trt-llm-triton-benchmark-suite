# Framework Comparisons

A condensed view of the runtimes this suite benchmarks. The full analysis,
including references, lives in [`blueprint.md`](../blueprint.md) (Competitive
Analysis). The point of the suite is to replace qualitative claims with
**apples-to-apples measurements on identical hardware and workloads**.

| Framework | Core idea | Strengths | Weaknesses |
|-----------|-----------|-----------|------------|
| **TensorRT-LLM + Triton** | Compiled TensorRT engines behind Triton scheduling | Deepest NVIDIA-specific optimization; FP8/FP4/INT4; in-flight batching; paged KV; multi-GPU/node | NVIDIA-only; build complexity; engine rebuilds per arch/shape |
| **vLLM** | PagedAttention + continuous batching | Great defaults, OpenAI API, strong throughput, big community | Less hardware-specific tuning than TRT-LLM |
| **LMDeploy / TurboMind** | Persistent batch + blocked KV cache | Up to ~1.8× vLLM throughput; strong 4-bit weight/KV quant | Smaller ecosystem, less documentation |
| **TGI** | Rust router + model server, continuous batching | Simple deploy, HF ecosystem, OpenAI-style API | Fewer exotic optimizations |
| **SGLang** | Structured/agentic runtime, RadixCache | Excellent for structured workflows; strong TTFT | Younger ecosystem, more complex model |
| **ONNX Runtime GenAI** | On-device generative loop | Portability (CPU/GPU/NPU) | Not aimed at peak GPU throughput |

## How the suite makes it fair

- **One harness, many backends:** the same closed-loop controller and workload
  drive every backend (`backends.py`), so differences reflect the runtime, not
  the client.
- **Identical workloads:** fix `workload.prompt_text`, `output_tokens`,
  `temperature`, and `top_p` across runs; sweep only the axis under test.
- **Same metrics everywhere:** TTFT, ITL, p50/p95/p99, TPS/RPS, VRAM, and
  cost/M tokens are computed identically downstream of the backend.
- **Sanity checks:** verify each backend returns comparable generations before
  trusting performance deltas (a blueprint risk-mitigation item).

## Backend wiring quick reference

| Backend | `backend.type` | `api_style` | Typical port |
|---------|----------------|-------------|--------------|
| vLLM | `vllm` | `openai-chat` | 8000 |
| TGI | `tgi` | `openai-chat` | 8080/80 |
| LMDeploy | `lmdeploy` | `openai-chat` | 23333 |
| Triton + TRT-LLM | `triton` (alias `triton_trtllm`) | `triton-generate` | 8000 (HTTP) |
| (any OpenAI API) | `openai` | `openai-chat`/`openai-completions` | — |

## Running a comparison

`benchmarks/runner/run_comparison.py` drives every backend through the *same*
harness (it calls the single-experiment `run_experiment`, so there's no logic
duplication) and emits a combined table.

```bash
# No GPU (simulator): validates the mechanics and table output
python -m benchmarks.runner.run_comparison \
    --config benchmarks/configs/backend_comparison.yaml --backend mock --num-requests 8

# Real backends (after bringing them up via infra/docker-compose.yml):
python -m benchmarks.runner.run_comparison --config benchmarks/configs/backend_comparison.yaml
```

**Output:** `results/comparison.md` and `results/comparison.csv` with one row per
backend:

| backend | precision | success | errors | output_tps | rps | ttft_p50_ms | ttft_p95_ms | latency_p50_s | latency_p95_s | latency_p99_s | itl_p50_ms | gpu_util_mean_pct |
|---------|-----------|---------|--------|-----------|-----|-------------|-------------|---------------|---------------|---------------|------------|-------------------|
| vllm | fp16 | … | … | … | … | … | … | … | … | … | … | … |

**How it ties back to the blueprint:** this is the concrete instrument behind
the *Competitive Analysis* table and the *Tradeoff Analysis* (latency vs
throughput, portability vs peak performance). Pair the table with
`cost/cost_analysis.py` to add the cost-per-million-tokens dimension, and feed
`comparison.csv` into `notebooks/analysis/` for latency-vs-TPS plots.

> Validity guardrails (blueprint *Risk Analysis*): keep `prompt_text`, output
> length, and decoding params identical across backends, warm up before
> measuring, and sanity-check that generations are comparable before trusting
> performance deltas.
