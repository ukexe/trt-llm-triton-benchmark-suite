"""Standardized cross-backend comparison helper (blueprint: Competitive Analysis).

Loads a comparison config (typically a ``backends: [...]`` fan-out), runs each
backend through the *same* harness as the single-experiment runner (no logic
duplication -- it calls :func:`benchmarks.runner.client.run_experiment`), then
emits a combined comparison table as Markdown + CSV.

Usage::

    # No GPU (simulator): compare the wiring across "backends"
    python -m benchmarks.runner.run_comparison \
        --config benchmarks/configs/backend_comparison.yaml --backend mock --num-requests 8

    # Real backends (after bringing them up): drop --backend
    python -m benchmarks.runner.run_comparison --config benchmarks/configs/backend_comparison.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
from typing import Any

from .client import _apply_overrides, run_experiment
from .config import load_experiments

_COLUMNS = [
    "backend",
    "precision",
    "success",
    "errors",
    "output_tps",
    "rps",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "itl_p50_ms",
    "gpu_util_mean_pct",
]


def _gpu_util_mean(summary: dict[str, Any]) -> float:
    """Mean GPU utilization across sampled GPUs (0.0 when unavailable)."""
    gpus = summary.get("gpu", {}).get("gpus", {})
    means = [g.get("util_pct", {}).get("mean", 0.0) for g in gpus.values()]
    return round(sum(means) / len(means), 1) if means else 0.0


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    meta = summary["meta"]
    return {
        "backend": meta.get("backend", ""),
        "precision": meta.get("precision", ""),
        "success": summary["counts"]["success"],
        "errors": summary["counts"]["errors"],
        "output_tps": round(summary["throughput"]["output_tps"], 1),
        "rps": round(summary["throughput"]["rps"], 2),
        "ttft_p50_ms": round(summary["ttft_s"]["p50"] * 1000, 1),
        "ttft_p95_ms": round(summary["ttft_s"]["p95"] * 1000, 1),
        "latency_p50_s": round(summary["latency_s"]["p50"], 3),
        "latency_p95_s": round(summary["latency_s"]["p95"], 3),
        "latency_p99_s": round(summary["latency_s"]["p99"], 3),
        "itl_p50_ms": round(summary["itl_ms"]["p50"], 2),
        "gpu_util_mean_pct": _gpu_util_mean(summary),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in _COLUMNS) + " |" for r in rows]
    return "\n".join([header, sep, *body]) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a standardized cross-backend comparison.")
    parser.add_argument(
        "--config",
        default="benchmarks/configs/backend_comparison.yaml",
        help="Comparison config (backends fan-out or experiments list).",
    )
    parser.add_argument("--out", default="results", help="Output directory (default: results).")
    # Overrides shared with the single-experiment runner (reused via _apply_overrides).
    parser.add_argument("--backend", help="Force every run onto one backend.type (e.g. mock).")
    parser.add_argument("--backend-url", help="Override backend.base_url for all runs.")
    parser.add_argument("--model", help="Override backend.model for all runs.")
    parser.add_argument("--concurrency", type=int, help="Override load.concurrency.")
    parser.add_argument("--num-requests", type=int, help="Run a fixed number of requests.")
    parser.add_argument("--max-seconds", type=float, help="Run for a fixed duration (seconds).")
    args = parser.parse_args(argv)

    experiments = load_experiments(args.config)
    for exp in experiments:
        _apply_overrides(exp, args)
        exp.validate()

    print(f"Comparing {len(experiments)} backend run(s) from {args.config} -> {args.out}")
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        summary = asyncio.run(run_experiment(exp, args.out))
        rows.append(_row(summary))
        print(
            f"  [{exp.backend.type}] ok={summary['counts']['success']} "
            f"out_tps={summary['throughput']['output_tps']:.1f} "
            f"ttft_p50={summary['ttft_s']['p50'] * 1000:.0f}ms "
            f"lat_p95={summary['latency_s']['p95']:.2f}s"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "comparison.md"
    md_path.write_text(_markdown_table(rows), encoding="utf-8")

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + _markdown_table(rows))
    print(f"Wrote {md_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
