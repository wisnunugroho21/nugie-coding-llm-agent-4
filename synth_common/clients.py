"""
Real `TeacherModel` backends — drop-in replacements for the offline `MockTeacher`
used by the annealing (Sec. 2.3) and SFT (Sec. 4) synthesis pipelines.

Three implementations, all satisfying the `TeacherModel` protocol
(`generate(prompt, temperature=..., max_tokens=...) -> str`):

  * `ClaudeTeacher`            — the official Anthropic SDK (default model
    `claude-opus-4-8`). This is the recommended production teacher.
  * `OpenAICompatibleTeacher`  — any OpenAI-compatible endpoint, i.e. a local
    **vLLM** server (`--api-key`/`--base-url`) or another self-hosted model.
  * `CachingTeacher`           — wraps any teacher with an on-disk cache so
    identical prompts aren't regenerated (dedup, resumability, reproducibility).

Two facts drive the Claude client's shape (both from the Anthropic API reference):
  1. Opus 4.8 / 4.7, Sonnet 5, and Fable 5 **reject `temperature`/`top_p`/`top_k`
     with a 400**. The pipelines still pass a `temperature` (the paper sets T=1.0
     for diverse questions), so `ClaudeTeacher` accepts it for protocol
     compatibility but does **not** forward it unless `forward_sampling=True` and
     you've pointed it at a model that accepts sampling params (e.g. Haiku 4.5).
  2. The SDK already retries 429/5xx with exponential backoff (`max_retries`), so
     we configure that instead of hand-rolling a retry loop.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any

from .teacher import MockTeacher, TeacherModel

# Default system prompt: frame the model as a code-instruction teacher.
_DEFAULT_SYSTEM = (
    "You are an expert programming instructor generating high-quality, correct, "
    "self-contained code and instruction-tuning data. Follow the requested format "
    "exactly and return fenced code blocks when code is asked for."
)


# --------------------------------------------------------------------------- #
#  Anthropic Claude
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ClaudeTeacher:
    """TeacherModel backed by the Anthropic Messages API (default: Opus 4.8).

    Credentials resolve the SDK way (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
    `ant auth login` profile) — nothing is hardcoded. Pass `client=` to inject a
    stub for testing.
    """

    model: str = "claude-opus-4-8"
    max_tokens: int = 8192
    system: str | None = _DEFAULT_SYSTEM
    # "adaptive" | "disabled" | None(omit). Adaptive thinking improves synthesis
    # quality; set None or "disabled" to cut latency/cost at scale.
    thinking: str | None = "adaptive"
    # Optional effort ("low".."max"); only sent when set (it errors on Haiku 4.5).
    effort: str | None = None
    # Only forward `temperature` when the target model actually accepts sampling
    # params. False by default because the default model (Opus 4.8) 400s on them.
    forward_sampling: bool = False
    max_retries: int = 4
    timeout: float | None = None
    # Injected Anthropic client (tests); built lazily from the SDK when None.
    client: Any = None

    def __post_init__(self) -> None:
        if self.client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - exercised without the dep
                raise RuntimeError(
                    "ClaudeTeacher needs the Anthropic SDK. Install it with:\n"
                    "    pip install anthropic\n"
                    "and provide credentials (ANTHROPIC_API_KEY or `ant auth login`)."
                ) from e
            kwargs: dict[str, Any] = {"max_retries": self.max_retries}
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            self.client = anthropic.Anthropic(**kwargs)

    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens, self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system
        if self.thinking:
            kwargs["thinking"] = {"type": self.thinking}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.forward_sampling:
            kwargs["temperature"] = temperature  # only for sampling-capable models

        resp = self.client.messages.create(**kwargs)

        # A safety refusal (e.g. Fable 5) yields stop_reason == "refusal" with no
        # usable content — skip it rather than crash the pipeline.
        if getattr(resp, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# --------------------------------------------------------------------------- #
#  OpenAI-compatible (vLLM / self-hosted)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class OpenAICompatibleTeacher:
    """TeacherModel for any OpenAI-compatible chat endpoint — e.g. a local vLLM
    server started with `vllm serve <model> --api-key ...`.

    Point `base_url` at the server (e.g. "http://localhost:8000/v1"). Sampling
    params are supported here (open models accept `temperature`).
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"           # vLLM accepts any non-empty key by default
    system: str | None = _DEFAULT_SYSTEM
    max_tokens: int = 8192
    max_retries: int = 4
    timeout: float | None = None
    client: Any = None

    def __post_init__(self) -> None:
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "OpenAICompatibleTeacher needs the OpenAI SDK. Install it with:\n"
                    "    pip install openai"
                ) from e
            kwargs: dict[str, Any] = {
                "base_url": self.base_url,
                "api_key": self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                "max_retries": self.max_retries,
            }
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max(max_tokens, self.max_tokens),
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
#  On-disk caching wrapper
# --------------------------------------------------------------------------- #
class CachingTeacher:
    """Wrap any TeacherModel with a persistent JSON cache keyed by
    (model_tag, prompt, temperature, max_tokens).

    Synthesizing millions of examples is expensive; caching makes reruns cheap,
    resumable after a crash, and reproducible. Set `model_tag` to something that
    identifies the underlying model so caches don't collide across teachers.
    """

    def __init__(self, inner: TeacherModel, cache_dir: str, model_tag: str = "") -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.model_tag = model_tag or getattr(inner, "model", inner.__class__.__name__)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(f"{self.model_tag}\0{temperature}\0{max_tokens}\0{prompt}".encode("utf-8"))
        return h.hexdigest()

    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        key = self._key(prompt, temperature, max_tokens)
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.hits += 1
                return json.load(fh)["response"]
        out = self.inner.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"prompt": prompt, "response": out, "temperature": temperature,
                       "max_tokens": max_tokens, "model_tag": self.model_tag}, fh)
        self.misses += 1
        return out


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_teacher(
    kind: str = "mock",
    *,
    model: str | None = None,
    base_url: str | None = None,
    cache_dir: str | None = None,
) -> TeacherModel:
    """Build a TeacherModel by name: "mock" (offline), "claude" (Anthropic SDK),
    or "vllm" (OpenAI-compatible endpoint). Wrap in a CachingTeacher when
    `cache_dir` is given."""
    if kind == "mock":
        teacher: TeacherModel = MockTeacher()
    elif kind == "claude":
        teacher = ClaudeTeacher(model=model or "claude-opus-4-8")
    elif kind == "vllm":
        if not model:
            raise ValueError("vllm teacher requires --model (the served model name)")
        teacher = OpenAICompatibleTeacher(model=model, base_url=base_url or "http://localhost:8000/v1")
    else:
        raise ValueError(f"unknown teacher kind: {kind!r} (use mock|claude|vllm)")
    if cache_dir:
        teacher = CachingTeacher(teacher, cache_dir=cache_dir)
    return teacher
