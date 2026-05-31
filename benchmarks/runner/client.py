"""Experiment controller + closed-loop load generator (CLI entry point).

Reads a config (single experiment or sweep), drives ``concurrency`` requests in
flight against a backend, measures per-request TTFT / latency / token counts,
collects GPU metrics, aggregates results, and writes summaries to disk.

Usage::

    python -m benchmarks.runner.client \
        --config benchmarks/configs/single_gpu_baseline.yaml \
        --backend mock --out results/

Concurrency model: *closed-loop*. Exactly ``concurrency`` workers each issue one
request at a time, so there are always (up to) ``concurrency`` requests in
flight -- the standard way LLM serving "concurrency" is reported.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from .backends import Backend, build_backend
from .config import ExperimentConfig, canonical_backend, load_experiments
from .metrics_collector import GpuSampler
from .result_aggregator import (
    RequestResult,
    append_summary_csv,
    summarize,
    write_prometheus_textfile,
    write_results_jsonl,
    write_summary_json,
)

_BASE_SENTENCE = "The quick brown fox jumps over the lazy dog. "  # 9 words


def build_prompt(prompt_tokens: int, prompt_text: str | None) -> str:
    """Return an explicit prompt, or synthesize one of ~``prompt_tokens`` words.

    The synthetic prompt is a coarse approximation (≈1 token per word). For
    rigorous cross-backend comparisons supply ``workload.prompt_text`` or wire in
    a shared tokenizer (see docs/experiments.md).
    """
    if prompt_text:
        return prompt_text
    reps = max(prompt_tokens, 1) // 9 + 1
    filler = (_BASE_SENTENCE * reps).strip()
    return "Summarize the following text.\n\n" + filler


async def run_request(
    backend: Backend,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    index: int,
    prompt_tokens: int,
    timeout_s: float,
) -> RequestResult:
    """Issue one streaming request and measure it."""
    t0 = time.perf_counter()
    state: dict[str, object] = {"ttft": None, "count": 0}

    async def _consume() -> None:
        agen = backend.stream(prompt, max_tokens, temperature=temperature, top_p=top_p)
        async for _chunk in agen:
            if state["ttft"] is None:
                state["ttft"] = time.perf_counter() - t0
            state["count"] = int(state["count"]) + 1  # type: ignore[arg-type]

    ttft: float | None
    try:
        await asyncio.wait_for(_consume(), timeout=timeout_s)
        ttft = state["ttft"]  # type: ignore[assignment]
        return RequestResult(
            index=index,
            start_s=t0,
            end_s=time.perf_counter(),
            prompt_tokens=prompt_tokens,
            output_tokens=int(state["count"]),
            ttft_s=ttft,
            success=True,
        )
    except Exception as exc:  # noqa: BLE001 - record the failure, keep the run going
        ttft = state["ttft"]  # type: ignore[assignment]
        return RequestResult(
            index=index,
            start_s=t0,
            end_s=time.perf_counter(),
            prompt_tokens=prompt_tokens,
            output_tokens=int(state["count"]),
            ttft_s=ttft,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_closed_loop(
    backend: Backend,
    prompt: str,
    exp: ExperimentConfig,
    *,
    num_requests: int | None,
    duration_s: float | None,
    collect: bool = True,
) -> tuple[list[RequestResult], float]:
    """Drive load with ``exp.load.concurrency`` workers until the stop condition."""
    results: list[RequestResult] = []
    issued = 0
    deadline = (
        time.perf_counter() + duration_s
        if (duration_s is not None and num_requests is None)
        else None
    )

    def claim() -> int | None:
        nonlocal issued
        if num_requests is not None and issued >= num_requests:
            return None
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        idx = issued
        issued += 1
        return idx

    async def worker() -> None:
        while True:
            idx = claim()
            if idx is None:
                return
            res = await run_request(
                backend,
                prompt,
                exp.workload.output_tokens,
                exp.workload.temperature,
                exp.workload.top_p,
                idx,
                exp.workload.prompt_tokens,
                exp.load.request_timeout_s,
            )
            if collect:
                results.append(res)

    start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(exp.load.concurrency)])
    return results, time.perf_counter() - start


async def run_experiment(exp: ExperimentConfig, out_dir: str | Path) -> dict:
    """Run a single experiment end-to-end and persist its artifacts."""
    exp.validate()
    backend = build_backend(exp.backend, timeout_s=exp.load.request_timeout_s)
    prompt = build_prompt(exp.workload.prompt_tokens, exp.workload.prompt_text)

    sampler = GpuSampler()
    try:
        if exp.load.warmup_requests > 0:
            await run_closed_loop(
                backend,
                prompt,
                exp,
                num_requests=exp.load.warmup_requests,
                duration_s=None,
                collect=False,
            )
        sampler.start()
        results, wall = await run_closed_loop(
            backend,
            prompt,
            exp,
            num_requests=exp.load.num_requests,
            duration_s=exp.load.duration_s,
        )
    finally:
        sampler.stop()
        await backend.aclose()

    summary = summarize(results, wall, meta=exp.metadata(), gpu_summary=sampler.summary())

    exp_dir = Path(out_dir) / exp.name
    write_results_jsonl(results, exp_dir / "results.jsonl")
    write_summary_json(summary, exp_dir / "summary.json")
    write_prometheus_textfile(summary, exp_dir / "metrics.prom")
    append_summary_csv(summary, Path(out_dir) / "summary_ledger.csv")
    return summary


def _apply_overrides(exp: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.backend:
        exp.backend.type = canonical_backend(args.backend)
        exp.backend.api_style = "auto"  # let build_backend pick the type default
    if args.backend_url:
        exp.backend.base_url = args.backend_url
    if args.model:
        exp.backend.model = args.model
    if args.concurrency is not None:
        exp.load.concurrency = args.concurrency
    if args.num_requests is not None:
        exp.load.num_requests = args.num_requests
        exp.load.duration_s = None
    if args.max_seconds is not None:
        exp.load.duration_s = args.max_seconds
        exp.load.num_requests = None


def _print_summary(summary: dict) -> None:
    meta = summary["meta"]
    print(
        f"  [{meta.get('name')}] backend={meta.get('backend')} "
        f"conc={meta.get('concurrency')} "
        f"ok={summary['counts']['success']} err={summary['counts']['errors']} "
        f"| out_tps={summary['throughput']['output_tps']:.1f} "
        f"rps={summary['throughput']['rps']:.2f} "
        f"| ttft_p50={summary['ttft_s']['p50'] * 1000:.0f}ms "
        f"lat_p95={summary['latency_s']['p95']:.2f}s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM inference benchmark controller.")
    parser.add_argument("--config", required=True, help="Path to an experiment/sweep YAML.")
    parser.add_argument("--out", default="results", help="Output directory (default: results).")
    parser.add_argument(
        "--backend", help="Override backend.type (mock|vllm|tgi|triton|triton_trtllm|lmdeploy)."
    )
    parser.add_argument("--backend-url", help="Override backend.base_url.")
    parser.add_argument("--model", help="Override backend.model.")
    parser.add_argument("--concurrency", type=int, help="Override load.concurrency.")
    parser.add_argument("--num-requests", type=int, help="Run a fixed number of requests.")
    parser.add_argument("--max-seconds", type=float, help="Run for a fixed duration (seconds).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve and print experiments, then exit."
    )
    args = parser.parse_args(argv)

    experiments = load_experiments(args.config)
    for exp in experiments:
        _apply_overrides(exp, args)
        exp.validate()

    if args.dry_run:
        print(f"Resolved {len(experiments)} experiment(s) from {args.config}:")
        for exp in experiments:
            print(f"  - {exp.name}: {exp.metadata()}")
        return 0

    print(f"Running {len(experiments)} experiment(s) -> {args.out}")
    for exp in experiments:
        summary = asyncio.run(run_experiment(exp, args.out))
        _print_summary(summary)
    print(f"Done. Per-experiment summaries in {args.out}/<name>/summary.json")
    print(f"Cross-experiment ledger: {Path(args.out) / 'summary_ledger.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
