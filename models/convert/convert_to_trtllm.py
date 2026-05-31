#!/usr/bin/env python
"""Build a TensorRT-LLM engine from a Hugging Face checkpoint.

This is a thin, documented wrapper around the canonical two-stage TensorRT-LLM
pipeline. It prints the exact commands by default (``--dry-run``) and only
executes them with ``--run`` (which requires a GPU host with TensorRT-LLM and
its example scripts installed).

Stage 1 -- convert HF weights to a TRT-LLM checkpoint:
    python convert_checkpoint.py --model_dir <hf> --output_dir <ckpt> --dtype <dtype>

Stage 2 -- build the optimized engine:
    trtllm-build --checkpoint_dir <ckpt> --output_dir <engines> \
        --gemm_plugin auto --max_batch_size N --max_input_len L --max_num_tokens T

The resulting engine goes under a Triton model repository (see
``models/triton_model_repo/llama3-8b-trt/1/``) and must match the
``config.pbtxt`` there.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def run_or_print(cmd: list[str], run: bool) -> None:
    """Print a command (always) and execute it when ``run`` is True."""
    print("  $ " + " ".join(shlex.quote(c) for c in cmd))
    if run:
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF model id or local path.")
    p.add_argument("--checkpoint-dir", default="trtllm_ckpt", help="Stage-1 output dir.")
    p.add_argument("--output-dir", default="trtllm_engines", help="Stage-2 engine dir.")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float8"])
    p.add_argument("--tp-size", type=int, default=1, help="Tensor-parallel size.")
    p.add_argument("--pp-size", type=int, default=1, help="Pipeline-parallel size.")
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=2048)
    p.add_argument("--max-num-tokens", type=int, default=8192)
    p.add_argument(
        "--convert-script",
        default="convert_checkpoint.py",
        help="Path to the model's TRT-LLM convert_checkpoint.py.",
    )
    p.add_argument("--run", action="store_true", help="Actually execute (needs a GPU + TRT-LLM).")
    args = p.parse_args(argv)

    print(f"[convert_to_trtllm] {args.model} -> {args.output_dir} (dtype={args.dtype})")

    stage1 = [
        sys.executable,
        args.convert_script,
        "--model_dir",
        args.model,
        "--output_dir",
        args.checkpoint_dir,
        "--dtype",
        args.dtype,
        "--tp_size",
        str(args.tp_size),
        "--pp_size",
        str(args.pp_size),
    ]
    stage2 = [
        "trtllm-build",
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--output_dir",
        args.output_dir,
        "--gemm_plugin",
        "auto",
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_input_len",
        str(args.max_input_len),
        "--max_num_tokens",
        str(args.max_num_tokens),
    ]

    print("Stage 1 - convert HF checkpoint:")
    run_or_print(stage1, args.run)
    print("Stage 2 - build TensorRT-LLM engine:")
    run_or_print(stage2, args.run)

    if not args.run:
        print("\n(dry-run) Re-run with --run on a GPU host to execute the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
