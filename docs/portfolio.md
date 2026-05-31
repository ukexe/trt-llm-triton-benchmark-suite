# For Reviewers & Hiring Managers

A 60-second tour of this project — what it demonstrates and where to look.

## The one-liner

A **multi-backend LLM inference benchmarking & cost-optimization suite**: it runs
the same workload against TensorRT-LLM/Triton, vLLM, LMDeploy, and TGI, measures
TTFT / inter-token latency / p50–p99 / TPS / GPU utilization, and converts the
results into **cost per million tokens** — turning runtime trade-offs into
apples-to-apples, business-relevant numbers.

## Skills it demonstrates

- **GPU inference internals:** continuous / in-flight batching, paged KV cache,
  FP8/INT8/INT4 quantization, Triton dynamic batching vs TRT-LLM in-flight
  batching (two-level batching tuning).
- **Benchmarking methodology:** closed-loop load generation, warm-up, TTFT/ITL,
  percentile latency, throughput, and reproducibility — aligned to GenAI-Perf /
  LLMPerf style metrics.
- **Systems/infra:** docker-compose + Kubernetes serving topologies, Prometheus
  + Grafana + DCGM observability, and a Prometheus textfile export tying
  benchmark KPIs into the metrics stack.
- **FinOps for ML:** GPU/hr pricing → cost per million tokens / per request, with
  a serverless reference point.
- **Engineering hygiene:** typed, config-driven harness; ruff-clean; unit-tested
  pure-Python core; clear docs and diagrams.

## What to open first (depth in ~5 files)

1. [`blueprint.md`](../blueprint.md) — the authoritative architecture/spec.
2. [`benchmarks/runner/client.py`](../benchmarks/runner/client.py) — the
   config-driven controller + closed-loop load generator.
3. [`benchmarks/runner/result_aggregator.py`](../benchmarks/runner/result_aggregator.py)
   — percentiles, TPS/RPS, TTFT/ITL, and the Prometheus export.
4. [`cost/cost_analysis.py`](../cost/cost_analysis.py) — the cost model.
5. [`models/triton_model_repo/llama3-8b-trt/config.pbtxt`](../models/triton_model_repo/llama3-8b-trt/config.pbtxt)
   — Triton + TRT-LLM scheduler/batching config.

Then skim the diagrams in [`docs/architecture.md`](./architecture.md) and the
experiment catalog in [`docs/experiments.md`](./experiments.md).

## Run it in 30 seconds (no GPU)

```bash
pip install -r requirements.txt
python -m benchmarks.runner.client --config benchmarks/configs/single_gpu_baseline.yaml \
    --backend mock --out results/
python -m cost.cost_analysis --results results/single_gpu_baseline/summary.json --gpu-type A100_80GB
```

The `mock` backend simulates TTFT/ITL so the full controller → metrics →
aggregation → cost pipeline runs anywhere; swap in a real `vllm`/`triton`
endpoint to produce publishable numbers.
