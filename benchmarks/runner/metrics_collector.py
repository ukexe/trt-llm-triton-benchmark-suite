"""GPU metrics collection during a benchmark run.

Samples ``nvidia-smi`` on a background thread and reports per-GPU average/peak
utilization, memory, and power. If ``nvidia-smi`` is unavailable (e.g. CPU-only
dev machines), the collector degrades gracefully and reports
``{"available": False, ...}`` instead of raising.

A future extension is a ``PrometheusScraper`` that pulls DCGM-exporter and
backend metrics; the interface below is intentionally simple so it can be
swapped or composed (see docs/architecture.md, Monitoring Strategy).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

_QUERY_FIELDS = "index,utilization.gpu,memory.used,memory.total,power.draw"


@dataclass
class _GpuSamples:
    util_pct: list[float] = field(default_factory=list)
    mem_used_mb: list[float] = field(default_factory=list)
    mem_total_mb: float = 0.0
    power_w: list[float] = field(default_factory=list)


def _safe_float(token: str) -> float | None:
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        return None  # e.g. "[N/A]" for power on some GPUs


class GpuSampler:
    """Context-managed background sampler for ``nvidia-smi`` metrics."""

    def __init__(self, interval_s: float = 0.5, smi_path: str | None = None) -> None:
        self.interval_s = interval_s
        self.smi_path = smi_path or shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpus: dict[str, _GpuSamples] = {}
        self._sample_count = 0
        self._error: str | None = None

    @property
    def available(self) -> bool:
        return self.smi_path is not None

    def _poll_once(self) -> None:
        assert self.smi_path is not None
        proc = subprocess.run(
            [
                self.smi_path,
                f"--query-gpu={_QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=self.interval_s + 5.0,
            check=False,
        )
        if proc.returncode != 0:
            self._error = proc.stderr.strip() or f"nvidia-smi exit {proc.returncode}"
            return
        for line in proc.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 5:
                continue
            idx = parts[0].strip()
            util = _safe_float(parts[1])
            mem_used = _safe_float(parts[2])
            mem_total = _safe_float(parts[3])
            power = _safe_float(parts[4])
            bucket = self._gpus.setdefault(idx, _GpuSamples())
            if util is not None:
                bucket.util_pct.append(util)
            if mem_used is not None:
                bucket.mem_used_mb.append(mem_used)
            if mem_total is not None:
                bucket.mem_total_mb = mem_total
            if power is not None:
                bucket.power_w.append(power)
        self._sample_count += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001 - sampler must never crash the run
                self._error = str(exc)
            self._stop.wait(self.interval_s)

    def start(self) -> GpuSampler:
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> GpuSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @staticmethod
    def _agg(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "max": 0.0}
        return {"mean": sum(values) / len(values), "max": max(values)}

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the sampled metrics."""
        if not self.available:
            return {"available": False, "reason": "nvidia-smi not found"}
        gpus: dict[str, Any] = {}
        for idx, s in self._gpus.items():
            gpus[idx] = {
                "util_pct": self._agg(s.util_pct),
                "mem_used_mb": self._agg(s.mem_used_mb),
                "mem_total_mb": s.mem_total_mb,
                "power_w": self._agg(s.power_w),
            }
        out: dict[str, Any] = {
            "available": True,
            "samples": self._sample_count,
            "interval_s": self.interval_s,
            "gpus": gpus,
        }
        if self._error:
            out["error"] = self._error
        return out


def sample_gpu_once(smi_path: str | None = None) -> dict[str, Any]:
    """One-shot GPU snapshot (handy for sanity checks / CLI)."""
    sampler = GpuSampler(smi_path=smi_path)
    if not sampler.available:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        sampler._poll_once()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
    return sampler.summary()


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    import json

    print(json.dumps(sample_gpu_once(), indent=2))
