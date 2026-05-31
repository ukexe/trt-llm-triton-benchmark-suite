"""Cost modeling: convert benchmark throughput + GPU price into business metrics.

Core identity (from the blueprint's Cost Analysis Framework)::

    cost_per_M_tokens = (gpu_hours * price_per_hr) * 1e6 / tokens

The pure functions below are dependency-free and unit-tested; the CLI/IO layer
loads ``gpu_pricing.yaml`` and a benchmark ``summary.json`` to produce a
``cost.json`` next to the summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PRICING = Path(__file__).with_name("gpu_pricing.yaml")


# --------------------------------------------------------------------------- #
# Pure cost math
# --------------------------------------------------------------------------- #


def gpu_hours(wall_time_s: float, num_gpus: int = 1) -> float:
    """GPU-hours consumed: wall-clock hours multiplied by the GPU count."""
    return (wall_time_s / 3600.0) * num_gpus


def cost_per_million_tokens(gpu_hrs: float, price_per_hr: float, tokens: int) -> float:
    """USD per 1,000,000 tokens. Returns 0.0 if no tokens were produced."""
    if tokens <= 0:
        return 0.0
    return (gpu_hrs * price_per_hr) * 1_000_000 / tokens


def cost_per_request(gpu_hrs: float, price_per_hr: float, num_requests: int) -> float:
    """USD per successful request. Returns 0.0 if there were no requests."""
    if num_requests <= 0:
        return 0.0
    return (gpu_hrs * price_per_hr) / num_requests


def analyze(
    summary: dict[str, Any],
    price_per_hr: float,
    *,
    num_gpus: int = 1,
) -> dict[str, Any]:
    """Derive cost metrics from a benchmark summary dict.

    Reports cost per million *output* tokens (the primary infra metric) and per
    million *total* (prompt + output) tokens, plus per-request and per-1k-request
    costs and the GPU spend for the measured window.
    """
    wall = float(summary.get("wall_time_s", 0.0))
    tokens = summary.get("tokens", {})
    counts = summary.get("counts", {})
    output_tokens = int(tokens.get("total_output", 0))
    prompt_tokens = int(tokens.get("total_prompt", 0))
    total_tokens = output_tokens + prompt_tokens
    successes = int(counts.get("success", 0))

    gh = gpu_hours(wall, num_gpus)
    spend = gh * price_per_hr

    return {
        "inputs": {
            "price_per_gpu_hr": price_per_hr,
            "num_gpus": num_gpus,
            "wall_time_s": wall,
            "gpu_hours": gh,
        },
        "tokens": {
            "output": output_tokens,
            "prompt": prompt_tokens,
            "total": total_tokens,
        },
        "spend_usd": spend,
        "cost_per_million_output_tokens": cost_per_million_tokens(gh, price_per_hr, output_tokens),
        "cost_per_million_total_tokens": cost_per_million_tokens(gh, price_per_hr, total_tokens),
        "cost_per_request": cost_per_request(gh, price_per_hr, successes),
        "cost_per_1k_requests": cost_per_request(gh, price_per_hr, successes) * 1000,
    }


# --------------------------------------------------------------------------- #
# Pricing table IO
# --------------------------------------------------------------------------- #


def load_pricing(path: str | Path = DEFAULT_PRICING) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _norm_gpu(name: str) -> str:
    """Normalize a GPU key: case-insensitive, '-' and '_' equivalent."""
    return name.strip().upper().replace("-", "_")


def find_entries(
    pricing: dict[str, Any], gpu_type: str, provider: str | None = None
) -> list[dict[str, Any]]:
    """Return all pricing rows for ``gpu_type`` (optionally a single ``provider``).

    Raises ``KeyError`` with the available options if nothing matches.
    """
    entries = pricing.get("pricing", [])
    target = _norm_gpu(gpu_type)
    rows = [e for e in entries if _norm_gpu(str(e.get("gpu_type", ""))) == target]
    if not rows:
        available = sorted({e.get("gpu_type") for e in entries})
        raise KeyError(f"GPU type {gpu_type!r} not in pricing table. Available: {available}")
    if provider is not None:
        rows = [e for e in rows if str(e.get("provider")) == provider]
        if not rows:
            providers = sorted(
                {
                    e.get("provider")
                    for e in entries
                    if _norm_gpu(str(e.get("gpu_type", ""))) == target
                }
            )
            raise KeyError(
                f"Provider {provider!r} not listed for {gpu_type!r}. Available: {providers}"
            )
    return rows


def lookup_price(pricing: dict[str, Any], gpu_type: str, provider: str) -> float:
    """Look up USD/hr for ``gpu_type`` at ``provider``; raise with helpful context."""
    return float(find_entries(pricing, gpu_type, provider)[0]["on_demand_price_per_hour"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _provider_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Resolve the pricing rows to evaluate from CLI args (manual price or table)."""
    if args.price is not None:
        return (
            [{"provider": "manual", "on_demand_price_per_hour": args.price, "notes": ""}],
            "manual",
        )
    if not args.gpu:
        raise SystemExit("error: provide either --price or --gpu/--gpu-type")
    pricing = load_pricing(args.pricing)
    return (find_entries(pricing, args.gpu, args.provider), args.gpu)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate LLM serving cost from a benchmark run.")
    parser.add_argument(
        "--summary",
        "--results",
        dest="summary",
        required=True,
        help="Path to a benchmark summary.json (exported by result_aggregator).",
    )
    parser.add_argument("--pricing", default=str(DEFAULT_PRICING), help="Path to gpu_pricing.yaml.")
    parser.add_argument(
        "--gpu", "--gpu-type", dest="gpu", help="GPU type, e.g. A100_80GB / A100-80GB."
    )
    parser.add_argument(
        "--provider", help="Provider key, e.g. runpod / lambda / vast (default: all providers)."
    )
    parser.add_argument("--price", type=float, help="Manual USD/GPU-hr (overrides table lookup).")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs used (default: 1).")
    parser.add_argument("--out", help="Where to write cost.json (default: next to summary).")
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows, gpu_label = _provider_rows(args)

    reports: list[dict[str, Any]] = []
    for entry in rows:
        price = float(entry["on_demand_price_per_hour"])
        report = analyze(summary, price, num_gpus=args.num_gpus)
        report["gpu_type"] = gpu_label
        report["provider"] = entry.get("provider", "")
        report["notes"] = entry.get("notes", "")
        reports.append(report)
    reports.sort(key=lambda r: r["cost_per_million_output_tokens"])

    name = summary.get("meta", {}).get("name", "") or args.summary
    print(f"Cost analysis for {name} (GPU={gpu_label}, num_gpus={args.num_gpus}):")
    print(f"  {'provider':<10} {'$/GPU-hr':>9} {'$/1M out':>10} {'$/1M total':>11} {'$/req':>10}")
    for r in reports:
        print(
            f"  {r['provider']:<10} {r['inputs']['price_per_gpu_hr']:>9.3f} "
            f"{r['cost_per_million_output_tokens']:>10.4f} "
            f"{r['cost_per_million_total_tokens']:>11.4f} {r['cost_per_request']:>10.6f}"
        )

    payload: Any = reports[0] if len(reports) == 1 else {"gpu_type": gpu_label, "results": reports}
    out_path = Path(args.out) if args.out else Path(args.summary).with_name("cost.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
