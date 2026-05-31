"""Serving-backend clients used by the benchmark harness.

Every backend exposes the same minimal streaming interface::

    async for chunk in backend.stream(prompt, max_tokens, temperature=..., top_p=...):
        ...   # chunk is a piece of generated text

The harness timestamps chunk *arrival* to derive TTFT and inter-token latency,
so backends only need to yield text as it streams in.

Implemented dialects:
  * ``openai-chat``        -> ``POST /v1/chat/completions`` (vLLM, TGI, others)
  * ``openai-completions`` -> ``POST /v1/completions``
  * ``triton-generate``    -> ``POST /v2/models/<m>/generate_stream`` (TRT-LLM)
  * ``mock``               -> in-process simulator (no server required)

Token accounting note: the harness counts streamed chunks as a proxy for output
tokens. For most servers one SSE delta corresponds to ~one token, but this is an
approximation; plugging in a shared tokenizer is a documented future improvement
(see docs/experiments.md).
"""

from __future__ import annotations

import abc
import asyncio
import json
import random
from collections.abc import AsyncIterator

import httpx

from .config import BackendConfig


class Backend(abc.ABC):
    """Abstract streaming backend."""

    name: str = "backend"

    @abc.abstractmethod
    def stream(
        self,
        prompt: str,
        max_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        """Yield generated text chunks as they arrive."""
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        """Release any held resources (HTTP connections, etc.)."""
        return None


def _sse_data_lines(raw_line: str) -> str | None:
    """Extract the payload of an SSE ``data:`` line, or ``None`` to skip it."""
    if not raw_line or raw_line.startswith(":"):
        return None
    if raw_line.startswith("data:"):
        return raw_line[len("data:") :].strip()
    return None


class _HttpBackend(Backend):
    """Shared HTTP plumbing for the streaming HTTP backends."""

    def __init__(self, cfg: BackendConfig, timeout_s: float = 120.0) -> None:
        self.cfg = cfg
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIChatBackend(_HttpBackend):
    """OpenAI-compatible ``/v1/chat/completions`` streaming (vLLM, TGI, ...)."""

    name = "openai-chat"

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **self.cfg.extra,
        }
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                data = _sse_data_lines(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    content = delta.get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if content:
                    yield content


class OpenAICompletionsBackend(_HttpBackend):
    """OpenAI-compatible ``/v1/completions`` streaming."""

    name = "openai-completions"

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **self.cfg.extra,
        }
        async with self._client.stream("POST", "/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                data = _sse_data_lines(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    text = obj["choices"][0].get("text")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if text:
                    yield text


class TritonGenerateBackend(_HttpBackend):
    """Triton + TensorRT-LLM ``/v2/models/<model>/generate_stream`` (best-effort).

    The exact request/response field names depend on the deployed
    ``tensorrtllm``/``ensemble`` model's ``config.pbtxt``. This implementation
    targets the common ``{"text_input", "max_tokens", "stream"}`` request and a
    ``{"text_output": "..."}`` streamed response; adjust ``backend.extra`` and
    these field names to match your model repository if needed.
    """

    name = "triton-generate"

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        url = f"/v2/models/{self.cfg.model}/generate_stream"
        payload = {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **self.cfg.extra,
        }
        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                data = _sse_data_lines(line)
                if data is None:
                    continue
                try:
                    obj = json.loads(data)
                    text = obj.get("text_output")
                except json.JSONDecodeError:
                    continue
                if text:
                    yield text


class MockBackend(Backend):
    """In-process simulator: emits ``max_tokens`` chunks with configured timing.

    Useful for exercising the full controller -> metrics -> aggregation -> cost
    pipeline with no GPU or server. Timing has mild jitter so percentiles and
    ITL statistics are non-degenerate.
    """

    name = "mock"

    def __init__(self, cfg: BackendConfig) -> None:
        self.ttft_s = max(cfg.mock_ttft_ms, 0.0) / 1000.0
        self.itl_s = max(cfg.mock_itl_ms, 0.0) / 1000.0
        self._rng = random.Random(0xBEEF)

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        await asyncio.sleep(self.ttft_s * self._rng.uniform(0.85, 1.15))
        yield "tok "
        for _ in range(max(max_tokens - 1, 0)):
            await asyncio.sleep(self.itl_s * self._rng.uniform(0.85, 1.15))
            yield "tok "


_API_STYLE_TO_BACKEND: dict[str, type[_HttpBackend]] = {
    "openai-chat": OpenAIChatBackend,
    "openai-completions": OpenAICompletionsBackend,
    "triton-generate": TritonGenerateBackend,
}

# Default API dialect for each backend type when ``api_style`` is left implicit.
_TYPE_DEFAULT_STYLE = {
    "vllm": "openai-chat",
    "tgi": "openai-chat",
    "openai": "openai-chat",
    "lmdeploy": "openai-chat",
    "triton": "triton-generate",
}


def build_backend(cfg: BackendConfig, timeout_s: float = 120.0) -> Backend:
    """Instantiate the concrete backend described by ``cfg``."""
    if cfg.type == "mock":
        return MockBackend(cfg)

    style = cfg.api_style
    if style not in _API_STYLE_TO_BACKEND:
        style = _TYPE_DEFAULT_STYLE.get(cfg.type, "openai-chat")
    backend_cls = _API_STYLE_TO_BACKEND[style]
    return backend_cls(cfg, timeout_s=timeout_s)
