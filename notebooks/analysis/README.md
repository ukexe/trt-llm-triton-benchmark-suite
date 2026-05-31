# Analysis Notebooks

Plot the artifacts the benchmark harness writes (`results/<name>/summary.json`
and the `results/summary_ledger.csv` cross-experiment ledger), per the
blueprint's *Visualization Strategy*.

Install the plotting extras first:

```bash
pip install -e ".[analysis]"
```

| Notebook | What it plots | Reads |
|----------|---------------|-------|
| `latency_vs_throughput.ipynb` | Output TPS vs p95 latency frontier; TTFT vs concurrency | `results/summary_ledger.csv` |
| `cost_per_token.ipynb` | Cost per 1M output tokens by provider/experiment | `results/*/summary.json` + `cost/gpu_pricing.yaml` |

Generate data to plot (no GPU needed):

```bash
python -m benchmarks.runner.client --config benchmarks/configs/concurrency_sweep.yaml \
    --backend mock --out results/
```

Suggested additions (good next charts): **VRAM vs concurrent sequences** (paged
KV-cache payoff) and **cost per 1M tokens vs TPS** across GPUs/backends. Use one
consistent color per backend and a separate chart per model/precision for
readability.
