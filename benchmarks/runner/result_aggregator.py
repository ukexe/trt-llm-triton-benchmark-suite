"""Result aggregation: turn per-request measurements into experiment summaries.

This module is intentionally dependency-light (standard library only) so it can
be imported and unit-tested without ``httpx``/``PyYAML`` or a live backend.

It defines :class:`RequestResult` (the atom produced by the load generator) and
pure functions that compute latency percentiles, throughput (TPS/RPS), TTFT and
inter-token-latency (ITL) statistics, then serialize summaries to JSON/CSV.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Data atom
# --------------------------------------------------------------------------- #


@dataclass
class RequestResult:
    """Measurements for a single benchmark request.

    All timestamps are seconds from a monotonic clock (``time.perf_counter``),
    relative to an arbitrary origin; only differences are meaningful.
    """

    index: int
    start_s: float
    end_s: float
    prompt_tokens: int
    output_tokens: int
    ttft_s: float | None = None
    success: bool = True
    error: str | None = None

    @property
    def latency_s(self) -> float:
        """End-to-end latency (request submit -> final token)."""
        return self.end_s - self.start_s

    @property
    def itl_s(self) -> float | None:
        """Mean inter-token latency for this request, if measurable."""
        if self.ttft_s is None or self.output_tokens <= 1:
            return None
        decode_time = self.latency_s - self.ttft_s
        if decode_time < 0:
            return None
        return decode_time / (self.output_tokens - 1)


# --------------------------------------------------------------------------- #
# Pure statistics helpers
# --------------------------------------------------------------------------- #


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (``q`` in [0, 100]).

    Returns ``0.0`` for an empty input so summaries stay JSON-clean.
    """
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _stats(values: list[float]) -> dict[str, float]:
    """Common descriptive statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #


def summarize(
    results: list[RequestResult],
    wall_time_s: float,
    meta: dict[str, Any] | None = None,
    gpu_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-request results into an experiment-level summary dict.

    Args:
        results: All completed requests (successful and failed).
        wall_time_s: Wall-clock duration of the measurement window (seconds).
        meta: Optional experiment metadata (name, backend, precision, ...).
        gpu_summary: Optional GPU-metrics summary from the metrics collector.
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    latencies = [r.latency_s for r in successful]
    ttfts = [r.ttft_s for r in successful if r.ttft_s is not None]
    itls_ms = [r.itl_s * 1000.0 for r in successful if r.itl_s is not None]

    total_output = sum(r.output_tokens for r in successful)
    total_prompt = sum(r.prompt_tokens for r in successful)
    wall = max(wall_time_s, 1e-9)

    summary: dict[str, Any] = {
        "meta": meta or {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": wall_time_s,
        "counts": {
            "total": len(results),
            "success": len(successful),
            "errors": len(failed),
            "error_rate": (len(failed) / len(results)) if results else 0.0,
        },
        "latency_s": _stats(latencies),
        "ttft_s": _stats(ttfts),
        "itl_ms": _stats(itls_ms),
        "throughput": {
            "output_tps": total_output / wall,
            "total_tps": (total_output + total_prompt) / wall,
            "rps": len(successful) / wall,
        },
        "tokens": {
            "total_output": total_output,
            "total_prompt": total_prompt,
            "mean_output_per_request": (total_output / len(successful)) if successful else 0.0,
        },
    }
    if gpu_summary:
        summary["gpu"] = gpu_summary
    return summary


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

# Flattened columns appended to the cross-experiment CSV ledger.
CSV_COLUMNS = [
    "name",
    "backend",
    "precision",
    "hardware",
    "concurrency",
    "success",
    "errors",
    "wall_time_s",
    "output_tps",
    "rps",
    "ttft_p50_s",
    "ttft_p95_s",
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "itl_p50_ms",
]


def _csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    meta = summary.get("meta", {})
    return {
        "name": meta.get("name", ""),
        "backend": meta.get("backend", ""),
        "precision": meta.get("precision", ""),
        "hardware": meta.get("hardware", ""),
        "concurrency": meta.get("concurrency", ""),
        "success": summary["counts"]["success"],
        "errors": summary["counts"]["errors"],
        "wall_time_s": round(summary["wall_time_s"], 4),
        "output_tps": round(summary["throughput"]["output_tps"], 4),
        "rps": round(summary["throughput"]["rps"], 4),
        "ttft_p50_s": round(summary["ttft_s"]["p50"], 4),
        "ttft_p95_s": round(summary["ttft_s"]["p95"], 4),
        "latency_p50_s": round(summary["latency_s"]["p50"], 4),
        "latency_p95_s": round(summary["latency_s"]["p95"], 4),
        "latency_p99_s": round(summary["latency_s"]["p99"], 4),
        "itl_p50_ms": round(summary["itl_ms"]["p50"], 4),
    }


def write_summary_json(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_results_jsonl(results: list[RequestResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    return path


def load_results_jsonl(path: str | Path) -> list[RequestResult]:
    path = Path(path)
    out: list[RequestResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(RequestResult(**json.loads(line)))
    return out


def append_summary_csv(summary: dict[str, Any], path: str | Path) -> Path:
    """Append a flattened summary row to a cross-experiment CSV ledger."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(_csv_row(summary))
    return path


def _prom_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _prom_labels(meta: dict[str, Any]) -> str:
    keys = ("name", "backend", "precision", "hardware", "model")
    parts = [f'{k}="{_prom_escape(str(meta.get(k, "")))}"' for k in keys]
    return "{" + ",".join(parts) + "}"


def write_prometheus_textfile(summary: dict[str, Any], path: str | Path) -> Path:
    """Write run metrics in Prometheus text-exposition format.

    Designed for the node_exporter *textfile collector* (or a Pushgateway), this
    ties each benchmark run to the metrics stack described in the blueprint's
    *Monitoring Strategy*. Labels carry the experiment identity so dashboards can
    slice by backend / precision / hardware.
    """
    meta = summary.get("meta", {})
    labels = _prom_labels(meta)
    thr = summary["throughput"]
    lat = summary["latency_s"]
    ttft = summary["ttft_s"]
    itl = summary["itl_ms"]
    counts = summary["counts"]
    tokens = summary["tokens"]

    lines: list[str] = []

    def add(metric: str, help_text: str, value: float, mtype: str = "gauge") -> None:
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} {mtype}")
        lines.append(f"{metric}{labels} {value}")

    add("llm_bench_output_tps", "Output tokens per second", thr["output_tps"])
    add("llm_bench_requests_per_second", "Completed requests per second", thr["rps"])
    add("llm_bench_latency_seconds_p50", "p50 end-to-end latency", lat["p50"])
    add("llm_bench_latency_seconds_p95", "p95 end-to-end latency", lat["p95"])
    add("llm_bench_latency_seconds_p99", "p99 end-to-end latency", lat["p99"])
    add("llm_bench_ttft_seconds_p50", "p50 time to first token", ttft["p50"])
    add("llm_bench_ttft_seconds_p95", "p95 time to first token", ttft["p95"])
    add("llm_bench_itl_milliseconds_p50", "p50 inter-token latency (ms)", itl["p50"])
    add("llm_bench_requests_total", "Successful requests", counts["success"], "counter")
    add("llm_bench_errors_total", "Failed requests", counts["errors"], "counter")
    add("llm_bench_output_tokens_total", "Total output tokens", tokens["total_output"], "counter")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass
class AggregatedRun:
    """Convenience container pairing raw results with their summary."""

    results: list[RequestResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
