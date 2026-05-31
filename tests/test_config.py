"""Unit tests for experiment config loading, validation, and sweep expansion."""

from __future__ import annotations

import pytest

from benchmarks.runner.config import load_experiments

_SINGLE = """
name: unit_single
hardware: A100-80GB
precision: fp16
backend:
  type: vllm
  base_url: http://localhost:8000
  model: test-model
workload:
  prompt_tokens: 128
  output_tokens: 64
load:
  concurrency: 8
  duration_s: 10
"""

_SWEEP = """
name: unit_sweep
base:
  precision: fp16
  backend:
    type: vllm
    base_url: http://localhost:8000
    model: test-model
  workload:
    prompt_tokens: 128
    output_tokens: 64
  load:
    duration_s: 10
sweep:
  load.concurrency: [1, 4, 16, 64]
"""

_INVALID = """
name: bad
backend:
  type: not-a-real-backend
load:
  concurrency: 4
  duration_s: 10
"""


def test_load_single(tmp_path):
    path = tmp_path / "single.yaml"
    path.write_text(_SINGLE, encoding="utf-8")
    exps = load_experiments(path)
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "unit_single"
    assert exp.backend.type == "vllm"
    assert exp.load.concurrency == 8
    assert exp.workload.output_tokens == 64
    assert exp.metadata()["hardware"] == "A100-80GB"


def test_sweep_expansion(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(_SWEEP, encoding="utf-8")
    exps = load_experiments(path)
    assert len(exps) == 4
    by_conc = {e.load.concurrency: e for e in exps}
    assert set(by_conc) == {1, 4, 16, 64}
    # Dotted override was applied to the nested load.concurrency field.
    assert by_conc[16].load.concurrency == 16
    assert "concurrency=16" in by_conc[16].name


def test_invalid_backend_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(_INVALID, encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiments(path)
