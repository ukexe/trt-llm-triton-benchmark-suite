"""Compatibility shim exposing the config loader under the blueprint's name.

The Phase 4 spec references ``config_loader.py``; the implementation lives in
``config.py``. This module re-exports the public surface so either import path
works:

    from benchmarks.runner.config_loader import load_experiments, ExperimentConfig
"""

from __future__ import annotations

from .config import (
    BackendConfig,
    ExperimentConfig,
    LoadConfig,
    WorkloadConfig,
    canonical_backend,
    load_experiments,
)

__all__ = [
    "BackendConfig",
    "ExperimentConfig",
    "LoadConfig",
    "WorkloadConfig",
    "canonical_backend",
    "load_experiments",
]
