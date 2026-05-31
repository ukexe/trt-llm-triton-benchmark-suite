"""Unit tests for the cost model (pure math + pricing lookup)."""

from __future__ import annotations

import pytest

from cost.cost_analysis import (
    analyze,
    cost_per_million_tokens,
    cost_per_request,
    find_entries,
    gpu_hours,
    load_pricing,
    lookup_price,
)


def test_pure_cost_math():
    assert gpu_hours(3600.0, 1) == 1.0
    assert gpu_hours(3600.0, 4) == 4.0
    # 1 GPU-hour at $2/hr over exactly 1M tokens -> $2 per 1M tokens.
    assert cost_per_million_tokens(1.0, 2.0, 1_000_000) == 2.0
    assert cost_per_million_tokens(1.0, 2.0, 0) == 0.0
    assert cost_per_request(1.0, 2.0, 100) == 0.02
    assert cost_per_request(1.0, 2.0, 0) == 0.0


def test_analyze_from_summary():
    summary = {
        "wall_time_s": 3600.0,
        "counts": {"success": 100},
        "tokens": {"total_output": 1_000_000, "total_prompt": 0},
    }
    report = analyze(summary, price_per_hr=2.0, num_gpus=1)
    assert report["spend_usd"] == 2.0
    assert report["cost_per_million_output_tokens"] == 2.0
    assert report["cost_per_request"] == 0.02
    assert report["cost_per_1k_requests"] == 20.0
    assert report["inputs"]["gpu_hours"] == 1.0


def test_pricing_lookup():
    pricing = load_pricing()  # bundled cost/gpu_pricing.yaml
    # '-' and '_' are equivalent; case-insensitive.
    assert lookup_price(pricing, "A100-80GB", "runpod") == 1.19
    assert lookup_price(pricing, "a100_80gb", "runpod") == 1.19
    with pytest.raises(KeyError):
        lookup_price(pricing, "GPU-9000", "runpod")
    with pytest.raises(KeyError):
        lookup_price(pricing, "A100-80GB", "no-such-provider")


def test_find_entries_multi_provider():
    pricing = load_pricing()
    # Without a provider filter, all A100 providers are returned.
    rows = find_entries(pricing, "A100_80GB")
    assert len(rows) >= 3
    assert {r["provider"] for r in rows} >= {"runpod", "lambda", "vast"}
