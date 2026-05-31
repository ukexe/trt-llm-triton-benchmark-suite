#!/usr/bin/env python
"""Produce an INT8 (SmoothQuant) TensorRT-LLM checkpoint, then build an engine.

SmoothQuant migrates activation outliers into weights so per-channel INT8 GEMMs
retain accuracy. Optionally enable INT8 KV-cache for additional memory savings.

    python convert_checkpoint.py --model_dir <hf> --output_dir <ckpt> \
        --dtype float16 --smoothquant 0.5 --per_token --per_channel \
        [--int8_kv_cache]
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
    p.add_argument("--checkpoint-dir", default="int8_ckpt")
    p.add_argument("--output-dir", default="int8_engines")
    p.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant migration strength.")
    p.add_argument("--int8-kv-cache", action="store_true", help="Also quantize the KV cache.")
    p.add_argument("--convert-script", default="convert_checkpoint.py")
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)

    print(f"[quantize_int8] {args.model} -> INT8 SmoothQuant engine in {args.output_dir}")
    convert = [
        sys.executable,
        args.convert_script,
        "--model_dir",
        args.model,
        "--output_dir",
        args.checkpoint_dir,
        "--dtype",
        "float16",
        "--smoothquant",
        str(args.alpha),
        "--per_token",
        "--per_channel",
    ]
    if args.int8_kv_cache:
        convert.append("--int8_kv_cache")
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

    print("Stage 1 - INT8 SmoothQuant conversion:")
    run_or_print(convert, args.run)
    print("Stage 2 - build engine:")
    run_or_print(build, args.run)

    if not args.run:
        print("\n(dry-run) Re-run with --run on a GPU host to execute the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
