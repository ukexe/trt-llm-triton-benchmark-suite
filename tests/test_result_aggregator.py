"""Unit tests for the result aggregator (pure stats + serialization)."""

from __future__ import annotations

import json

from benchmarks.runner.result_aggregator import (
    RequestResult,
    append_summary_csv,
    load_results_jsonl,
    percentile,
    summarize,
    write_results_jsonl,
    write_summary_json,
)


def test_percentile_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0
    assert abs(percentile(values, 95) - 48.0) < 1e-9
    assert percentile([], 50) == 0.0


def _sample_results() -> list[RequestResult]:
    return [
        RequestResult(0, start_s=0.0, end_s=1.0, prompt_tokens=5, output_tokens=11, ttft_s=0.1),
        RequestResult(1, start_s=0.0, end_s=2.0, prompt_tokens=5, output_tokens=21, ttft_s=0.2),
        RequestResult(2, start_s=0.0, end_s=3.0, prompt_tokens=5, output_tokens=31, ttft_s=0.3),
        RequestResult(
            3, start_s=0.0, end_s=0.5, prompt_tokens=5, output_tokens=0, success=False, error="boom"
        ),
    ]


def test_summarize_core_metrics():
    summary = summarize(_sample_results(), wall_time_s=3.0, meta={"name": "t"})

    assert summary["counts"] == {"total": 4, "success": 3, "errors": 1, "error_rate": 0.25}
    # 63 output tokens over 3s wall time.
    assert abs(summary["throughput"]["output_tps"] - 21.0) < 1e-9
    assert abs(summary["throughput"]["rps"] - 1.0) < 1e-9
    assert summary["latency_s"]["p50"] == 2.0
    assert abs(summary["ttft_s"]["mean"] - 0.2) < 1e-9
    # Each request has a constant 90 ms inter-token latency by construction.
    assert abs(summary["itl_ms"]["mean"] - 90.0) < 1e-6
    assert summary["tokens"]["total_output"] == 63


def test_itl_property():
    r = RequestResult(0, start_s=0.0, end_s=1.0, prompt_tokens=0, output_tokens=11, ttft_s=0.1)
    assert abs(r.itl_s - 0.09) < 1e-9
    # Single-token responses have no measurable ITL.
    assert RequestResult(1, 0.0, 1.0, 0, 1, ttft_s=0.1).itl_s is None


def test_serialization_roundtrip(tmp_path):
    results = _sample_results()
    jsonl = write_results_jsonl(results, tmp_path / "results.jsonl")
    loaded = load_results_jsonl(jsonl)
    assert len(loaded) == len(results)
    assert loaded[0].output_tokens == 11

    summary = summarize(results, 3.0, meta={"name": "exp", "backend": "mock", "concurrency": 4})
    sjson = write_summary_json(summary, tmp_path / "summary.json")
    assert json.loads(sjson.read_text())["counts"]["success"] == 3

    csv_path = tmp_path / "ledger.csv"
    append_summary_csv(summary, csv_path)
    append_summary_csv(summary, csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + two rows
    assert lines[0].startswith("name,backend,precision")
