#!/usr/bin/env python
"""Produce an INT4-AWQ TensorRT-LLM checkpoint, then build an engine.

AWQ (Activation-aware Weight Quantization) keeps activations in higher precision
while quantizing weights to 4-bit. With TensorRT-LLM this is typically done via
the example ``quantize.py`` (ModelOpt-based) to emit a quantized checkpoint,
followed by ``trtllm-build``.

    python quantize.py --model_dir <hf> --qformat int4_awq \
        --output_dir <ckpt> --calib_size 512
    trtllm-build --checkpoint_dir <ckpt> --output_dir <engines> --gemm_plugin auto

Prints commands by default; pass ``--run`` to execute on a GPU host.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def run_or_print(cmd: list[str], run: bool) -> None:
    print("  $ " + " ".join(shlex.quote(c) for c in cmd))
    if run:
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF model id or local path.")
    p.add_argument("--checkpoint-dir", default="awq_ckpt")
    p.add_argument("--output-dir", default="awq_engines")
    p.add_argument("--calib-size", type=int, default=512, help="Calibration samples.")
    p.add_argument("--quantize-script", default="quantize.py", help="TRT-LLM quantize.py path.")
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)

    print(f"[quantize_awq] {args.model} -> INT4-AWQ engine in {args.output_dir}")
    quantize = [
        sys.executable,
        args.quantize_script,
        "--model_dir",
        args.model,
        "--qformat",
        "int4_awq",
        "--output_dir",
        args.checkpoint_dir,
        "--calib_size",
        str(args.calib_size),
    ]
    build = [
        "trtllm-build",
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--output_dir",
        args.output_dir,
        "--gemm_plugin",
        "auto",
        "--max_batch_size",
        str(args.max_batch_size),
    ]

    print("Stage 1 - AWQ quantization:")
    run_or_print(quantize, args.run)
    print("Stage 2 - build engine:")
    run_or_print(build, args.run)

    if not args.run:
        print("\n(dry-run) Re-run with --run on a GPU host to execute the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
