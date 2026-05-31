"""Experiment configuration: dataclasses, YAML loading, and sweep expansion.

A config file describes a single experiment or a *sweep* that expands (via a
Cartesian product over dotted-key overrides) into many experiments. This keeps
the harness fully config-driven, as required by the blueprint's benchmark
methodology section.

Example (single experiment)::

    name: single_gpu_baseline
    hardware: A100-80GB
    precision: fp16
    backend: {type: vllm, base_url: "http://localhost:8000", model: "..."}
    workload: {prompt_tokens: 512, output_tokens: 256}
    load: {concurrency: 16, duration_s: 60}

Example (sweep)::

    name: concurrency_sweep
    base: { ... same keys as above ... }
    sweep:
      load.concurrency: [1, 16, 64]
      precision: [fp16, fp8]
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_BACKENDS = {"mock", "vllm", "tgi", "triton", "lmdeploy", "openai"}
SUPPORTED_PRECISIONS = {"fp16", "bf16", "fp8", "int8", "int4", "awq", "gptq"}

# Backend-type aliases (canonical name on the right). Lets configs/CLI use the
# blueprint's names (e.g. ``triton_trtllm``) while the harness keys off ``triton``.
_BACKEND_ALIASES = {
    "triton_trtllm": "triton",
    "tensorrt_llm": "triton",
    "trtllm": "triton",
    "turbomind": "lmdeploy",
}


def canonical_backend(backend_type: str) -> str:
    """Resolve a backend alias to its canonical harness name."""
    return _BACKEND_ALIASES.get(backend_type, backend_type)


@dataclass
class BackendConfig:
    """How to reach a serving backend and which API dialect it speaks."""

    type: str = "mock"
    base_url: str = "http://localhost:8000"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # openai-chat | openai-completions | triton-generate | mock
    api_style: str = "openai-chat"
    api_key: str | None = None
    # Mock backend tuning (only used when type == "mock").
    mock_ttft_ms: float = 40.0
    mock_itl_ms: float = 10.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Canonicalize aliases (e.g. ``triton_trtllm`` -> ``triton``) at construction.
        self.type = canonical_backend(self.type)


@dataclass
class WorkloadConfig:
    """The request shape sent to the backend."""

    prompt_tokens: int = 512
    output_tokens: int = 256
    prompt_text: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True


@dataclass
class LoadConfig:
    """Closed-loop load: ``concurrency`` requests are kept in flight.

    Exactly one of ``num_requests`` or ``duration_s`` governs when the
    measurement window ends (``num_requests`` takes precedence if both set).
    """

    concurrency: int = 16
    num_requests: int | None = None
    duration_s: float | None = 60.0
    warmup_requests: int = 8
    request_timeout_s: float = 120.0


@dataclass
class ExperimentConfig:
    """A single, fully-specified benchmark experiment."""

    name: str
    hardware: str = "unknown-gpu"
    precision: str = "fp16"
    backend: BackendConfig = field(default_factory=BackendConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    seed: int = 1234
    notes: str = ""

    def validate(self) -> None:
        """Raise ``ValueError`` on obviously-invalid configuration."""
        if self.backend.type not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend.type={self.backend.type!r} not in {sorted(SUPPORTED_BACKENDS)}"
            )
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"precision={self.precision!r} not in {sorted(SUPPORTED_PRECISIONS)}")
        if self.load.concurrency < 1:
            raise ValueError("load.concurrency must be >= 1")
        if self.load.num_requests is None and self.load.duration_s is None:
            raise ValueError("set one of load.num_requests or load.duration_s")
        if self.workload.output_tokens < 1:
            raise ValueError("workload.output_tokens must be >= 1")

    def metadata(self) -> dict[str, Any]:
        """Compact metadata embedded into result summaries."""
        return {
            "name": self.name,
            "backend": self.backend.type,
            "model": self.backend.model,
            "precision": self.precision,
            "hardware": self.hardware,
            "concurrency": self.load.concurrency,
            "prompt_tokens": self.workload.prompt_tokens,
            "output_tokens": self.workload.output_tokens,
        }


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping for {cls.__name__}, got {type(data).__name__}")
    field_types = {f.name: f.type for f in fields(cls)}
    known = {}
    for key, value in data.items():
        if key not in field_types:
            raise ValueError(f"unknown key {key!r} for {cls.__name__}")
        # Nested dataclass fields are declared as forward-ref strings under
        # ``from __future__ import annotations``; resolve by name.
        nested = _NESTED.get((cls.__name__, key))
        if nested is not None and isinstance(value, dict):
            known[key] = _build_dataclass(nested, value)
        else:
            known[key] = value
    return cls(**known)


# Map of (dataclass name, field) -> nested dataclass type for the loader.
_NESTED: dict[tuple[str, str], type] = {
    ("ExperimentConfig", "backend"): BackendConfig,
    ("ExperimentConfig", "workload"): WorkloadConfig,
    ("ExperimentConfig", "load"): LoadConfig,
}


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c = value`` inside a nested dict, creating dicts as needed."""
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"cannot descend into non-mapping at {part!r} in {dotted_key!r}")
    cursor[parts[-1]] = value


def _expand_sweep(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a ``base`` + ``sweep`` document into concrete experiment dicts."""
    base = dict(raw.get("base", {}))
    sweep = raw.get("sweep", {})
    name = raw.get("name", base.get("name", "sweep"))

    if not sweep:
        single = dict(base)
        single.setdefault("name", name)
        return [single]

    keys = list(sweep.keys())
    value_lists = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]

    experiments: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        # Deep-ish copy of base via yaml round-trip keeps nested dicts independent.
        exp = yaml.safe_load(yaml.safe_dump(base)) or {}
        suffix_parts = []
        for key, value in zip(keys, combo, strict=True):
            _set_dotted(exp, key, value)
            suffix_parts.append(f"{key.split('.')[-1]}={value}")
        exp["name"] = f"{name}__{'__'.join(suffix_parts)}"
        experiments.append(exp)
    return experiments


# --------------------------------------------------------------------------- #
# Flat-schema normalization + multi-experiment expansion
# --------------------------------------------------------------------------- #

# Flat (blueprint-style) keys -> nested dataclass path. Lets a config use either
# the flat schema (name / backend / model / concurrency / prompt_length / ...)
# or the nested schema (backend:{...} / workload:{...} / load:{...}).
_FLAT_TO_NESTED: dict[str, tuple[str, ...]] = {
    "hardware_hint": ("hardware",),
    "model": ("backend", "model"),
    "api_style": ("backend", "api_style"),
    "base_url": ("backend", "base_url"),
    "prompt_length": ("workload", "prompt_tokens"),
    "output_length": ("workload", "output_tokens"),
    "prompt_text": ("workload", "prompt_text"),
    "temperature": ("workload", "temperature"),
    "top_p": ("workload", "top_p"),
    "concurrency": ("load", "concurrency"),
    "num_requests": ("load", "num_requests"),
    "duration_seconds": ("load", "duration_s"),
    "warmup_requests": ("load", "warmup_requests"),
    "request_timeout_s": ("load", "request_timeout_s"),
}


def _set_if_absent(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set ``data[path...] = value`` unless an explicit nested value already exists."""
    cursor = data
    for part in path[:-1]:
        nxt = cursor.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"cannot merge flat key into non-mapping at {part!r}")
        cursor = nxt
    cursor.setdefault(path[-1], value)


def _normalize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Translate flat blueprint-style keys into the nested schema (non-destructive).

    Nested values always win over their flat aliases, so mixing the two is safe.
    """
    if not isinstance(doc, dict):
        raise ValueError("experiment document must be a mapping")
    out = copy.deepcopy(doc)
    # A bare ``backend: vllm`` string means the backend *type*.
    if isinstance(out.get("backend"), str):
        out["backend"] = {"type": out["backend"]}
    for flat, path in _FLAT_TO_NESTED.items():
        if flat in out:
            _set_if_absent(out, path, out.pop(flat))
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a deep copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_experiments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand an explicit ``experiments: [...]`` list (each merged over defaults)."""
    defaults = raw.get("defaults") or raw.get("base") or {}
    name = raw.get("name", "experiments")
    docs: list[dict[str, Any]] = []
    for i, item in enumerate(raw["experiments"]):
        merged = _deep_merge(defaults, item)
        merged.setdefault("name", f"{name}__{i}")
        docs.append(merged)
    return docs


def _expand_backends(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a ``backends: [...]`` list into one experiment per backend.

    Optional ``backend_urls`` / ``models`` maps set a per-backend base_url/model.
    Used by the competitive-comparison helper (blueprint: Competitive Analysis).
    """
    name = raw.get("name", "backend_comparison")
    reserved = {"backends", "backend_urls", "models", "name"}
    base = {k: v for k, v in raw.items() if k not in reserved}
    urls = raw.get("backend_urls", {})
    models = raw.get("models", {})

    docs: list[dict[str, Any]] = []
    for backend_type in raw["backends"]:
        exp = copy.deepcopy(base)
        backend = exp.get("backend")
        backend = {"type": backend} if isinstance(backend, str) else dict(backend or {})
        backend["type"] = backend_type
        if backend_type in urls:
            backend["base_url"] = urls[backend_type]
        if backend_type in models:
            backend["model"] = models[backend_type]
        exp["backend"] = backend
        exp["name"] = f"{name}__backend={backend_type}"
        docs.append(exp)
    return docs


def load_experiments(path: str | Path) -> list[ExperimentConfig]:
    """Load one config file into one or more validated ``ExperimentConfig``.

    Supported document shapes:
      * single experiment (flat or nested schema);
      * ``base`` + ``sweep`` (Cartesian product over dotted/flat keys);
      * ``experiments: [...]`` explicit list (merged over ``defaults``);
      * ``backends: [...]`` comparison fan-out (one run per backend).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    if "experiments" in raw:
        docs = _expand_experiments(raw)
    elif "backends" in raw:
        docs = _expand_backends(raw)
    elif "sweep" in raw or "base" in raw:
        docs = _expand_sweep(raw)
    else:
        docs = [raw]

    experiments: list[ExperimentConfig] = []
    for doc in docs:
        doc = _normalize_doc(doc)
        if "name" not in doc:
            raise ValueError(f"{path}: experiment is missing required 'name'")
        exp = _build_dataclass(ExperimentConfig, doc)
        exp.validate()
        experiments.append(exp)
    return experiments


# Validate the nested map at import time to catch typos early.
assert is_dataclass(BackendConfig) and is_dataclass(WorkloadConfig) and is_dataclass(LoadConfig)
