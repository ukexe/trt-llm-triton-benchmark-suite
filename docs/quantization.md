# Quantization

Implements the blueprint's *Quantization Recommendations* / *Quantization
Techniques*. The goal is to benchmark FP16 vs FP8 vs INT8 vs 4-bit for the same
model and quantify the **speed / cost / accuracy** trade-off — not just speed.

## Paths and which backend serves them

| Precision | How it's produced | Served by | Script |
|-----------|-------------------|-----------|--------|
| FP16 (baseline) | none (reference) | any backend | — |
| FP8 | TensorRT-LLM build on Hopper (Transformer Engine) | Triton + TRT-LLM | `convert_to_trtllm.py --dtype float8` |
| INT8 (SmoothQuant) | per-channel/token INT8 + optional INT8 KV | Triton + TRT-LLM | `quantize_int8.py` |
| INT4 (AWQ) | weight-only 4-bit (activation-aware) | vLLM (AWQ), LMDeploy | `quantize_awq.py` |
| INT4 + KV quant | 4-bit weights + blocked KV cache | LMDeploy / TurboMind | `quantize_awq.py` (+ backend flag) |

> KV-cache quantization (4-bit/8-bit) is most turnkey on LMDeploy/TurboMind; for
> TensorRT-LLM it is enabled at build time (`--int8_kv_cache`, etc.). Full KV
> automation across all backends is out of scope here and marked as TODO in the
> conversion scripts.

## Producing artifacts

```bash
# INT8 SmoothQuant -> TRT-LLM engine
python models/convert/quantize_int8.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --alpha 0.5 --int8-kv-cache --run

# INT4 AWQ -> TRT-LLM engine (or use a prebuilt AWQ checkpoint for vLLM/LMDeploy)
python models/convert/quantize_awq.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --calib-size 512 --run
```

Each script prints the canonical pipeline (`quantize.py` / `convert_checkpoint.py`
→ `trtllm-build`) by default and only executes with `--run` on a GPU host.

## Benchmarking the variants

`benchmarks/configs/multi_precision_sweep.yaml` maps each precision to the
endpoint + model that serves it, so the runner selects the correct base URL /
model id per variant:

```bash
python -m benchmarks.runner.client --config benchmarks/configs/multi_precision_sweep.yaml --out results/
```

## What to compare

- **Latency / TPS:** lower precision should increase TPS and reduce p95.
- **VRAM:** 4-bit weights and quantized KV free memory for more concurrency.
- **Cost per 1M tokens:** feed each summary to `cost/cost_analysis.py`.
- **Accuracy (qualitative):** run a small fixed prompt set (e.g. an MT-Bench
  subset) and eyeball degradation. Aggressive 4-bit / KV quantization can
  regress quality on some tasks — quantization is a *trade-off*, not free speed.
