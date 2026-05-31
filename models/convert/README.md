# Model Conversion

CLI wrappers around the canonical TensorRT-LLM / ONNX conversion pipelines.
Each script **prints** the exact commands by default and only executes with
`--run` (which requires a GPU host with the relevant toolchain: TensorRT-LLM and
its example `convert_checkpoint.py` / `quantize.py`, or ONNX Runtime GenAI).

| Script | Output | Notes |
|--------|--------|-------|
| `convert_to_trtllm.py` | FP16/BF16/FP8 engine | HF → TRT-LLM checkpoint → `trtllm-build` |
| `quantize_awq.py` | INT4-AWQ engine | ModelOpt `quantize.py --qformat int4_awq` |
| `quantize_int8.py` | INT8 SmoothQuant engine | `--smoothquant`, optional `--int8_kv_cache` |
| `convert_to_onnx.py` | ONNX model | ONNX Runtime GenAI builder or `optimum-cli` |

Example (dry-run prints the pipeline):

```bash
python models/convert/convert_to_trtllm.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16 --max-batch-size 64 --output-dir models/triton_model_repo/llama3-8b-trt/1
```

Add `--run` on a GPU host to actually build. The engine must match the
`config.pbtxt` in `models/triton_model_repo/llama3-8b-trt/`.
