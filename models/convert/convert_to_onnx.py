#!/usr/bin/env python
"""Export a Hugging Face model to ONNX for the ONNX Runtime GenAI contrast point.

Two common paths are documented; pick one with ``--method``:

* ``genai-builder`` (recommended for LLMs) -- ONNX Runtime GenAI model builder::

    python -m onnxruntime_genai.models.builder \
        -m <hf> -o <out> -p <precision> -e <execution_provider>

* ``optimum`` -- generic exporter::

    optimum-cli export onnx --model <hf> <out>

Prints commands by default; pass ``--run`` to execute (requires the relevant
toolchain installed).
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
    p.add_argument("--output-dir", default="onnx_model")
    p.add_argument("--method", default="genai-builder", choices=["genai-builder", "optimum"])
    p.add_argument("--precision", default="int4", choices=["fp32", "fp16", "int4"])
    p.add_argument("--execution-provider", default="cuda", choices=["cuda", "cpu", "dml"])
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)

    print(f"[convert_to_onnx] {args.model} -> {args.output_dir} via {args.method}")
    if args.method == "genai-builder":
        cmd = [
            sys.executable,
            "-m",
            "onnxruntime_genai.models.builder",
            "-m",
            args.model,
            "-o",
            args.output_dir,
            "-p",
            args.precision,
            "-e",
            args.execution_provider,
        ]
    else:
        cmd = ["optimum-cli", "export", "onnx", "--model", args.model, args.output_dir]

    run_or_print(cmd, args.run)
    if not args.run:
        print("\n(dry-run) Re-run with --run once the toolchain is installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
