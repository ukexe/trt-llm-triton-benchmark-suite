# Cost Model

Bridges infra-level metrics (throughput) and business metrics (cost per million
tokens), per the blueprint's Cost Analysis Framework.

## Core identity

\[
\text{cost\_per\_M\_tokens} = \frac{(\text{gpu\_hours} \times \text{price\_per\_hr}) \times 10^6}{\text{tokens}}
\]

where \(\text{gpu\_hours} = \frac{\text{wall\_time\_s}}{3600} \times \text{num\_gpus}\).

Implemented as pure functions in `cost/cost_analysis.py`
(`gpu_hours`, `cost_per_million_tokens`, `cost_per_request`, `analyze`).

## Running

```bash
# All providers for a GPU type (table sorted by cost), then write cost.json:
python -m cost.cost_analysis --results results/single_gpu_baseline/summary.json --gpu-type A100_80GB

# A single provider, or a manual price:
python -m cost.cost_analysis --results results/single_gpu_baseline/summary.json \
    --gpu-type A100_80GB --provider runpod
python -m cost.cost_analysis --results results/single_gpu_baseline/summary.json --price 1.19
```

`--results`/`--summary` and `--gpu-type`/`--gpu` are interchangeable. GPU-type
matching is case-insensitive and treats `-`/`_` as equivalent (so `A100-80GB`
and `A100_80GB` are the same). Writes a `cost.json` next to the summary and
prints, per provider:

- GPU spend for the measured window
- **cost / 1M output tokens** (primary) and **cost / 1M total tokens**
- cost / request

For multi-GPU runs pass `--num-gpus N` (multiplies GPU-hours accordingly).

## Pricing table (`cost/gpu_pricing.yaml`)

A flat list; each entry has `provider`, `gpu_type`, `on_demand_price_per_hour`,
`notes`:

```yaml
pricing:
  - { gpu_type: A100_80GB, provider: runpod, on_demand_price_per_hour: 1.19, notes: "Secure Cloud" }
  - { gpu_type: H100_80GB, provider: spheron, on_demand_price_per_hour: 1.03, notes: "Marketplace low-end" }
  - { gpu_type: H100_80GB, provider: aws,     on_demand_price_per_hour: 6.88, notes: "p5 per-GPU approx" }
  - { gpu_type: L40S_48GB, provider: runpod,  on_demand_price_per_hour: 0.99 }
serverless_per_mtok:
  llama-3-8b: { together_input: 0.18, together_output: 0.18 }
```

Prices are **approximate 2026 snapshots** anchored on the blueprint references
(Spheron H100 ~$1.03/hr, AWS H100 ~$6.88/hr, Azure ND H100 v5 ~$12.29/hr).
Marketplace/spot rates fluctuate — refresh before quoting numbers.

## Serverless comparison

`serverless_per_mtok` lets you contrast self-hosted cost/M tokens against
serverless APIs (e.g. Together AI), enabling statements like *"A100 via RunPod
costs ~$X per million output tokens vs $Y for the comparable serverless model."*

## Caveats

- Cost-per-token uses **output** tokens by default (the dominant decode cost);
  `cost_per_million_total_tokens` includes prompt tokens for completeness.
- The token count inherits the chunk-counting approximation from the harness
  (see `docs/experiments.md`).
- Utilization matters: idle GPU time still bills. Compare cost at the
  **throughput-optimal** operating point, not just batch-1.
