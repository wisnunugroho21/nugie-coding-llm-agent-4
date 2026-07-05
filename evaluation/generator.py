"""
Generators — anything that maps a prompt to a candidate completion.

* `ModelGenerator`   — samples from the trained Kimi-Linear/GDN-2 model via its
  streaming `step` API (temperature sampling; greedy when temperature == 0).
* `FunctionGenerator`— wraps any `callable(prompt, temperature, max_new_tokens)`,
  e.g. a `synth_common` TeacherModel (`ClaudeTeacher`) for evaluating a real LLM.
* `OracleGenerator`  — returns each problem's reference solution; used to verify
  the harness itself (it must score Pass@1 == 1.0).

All expose `generate(prompt, temperature=..., max_new_tokens=...) -> str`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


class ModelGenerator:
    def __init__(self, model, tokenizer, seed: int = 0):
        self.model = model
        self.tok = tokenizer
        self.key = jax.random.PRNGKey(seed)
        self.chunk = model.cfg.gdn_chunk_size

    def _sample(self, logits_row: jax.Array, temperature: float, key) -> int:
        if temperature and temperature > 0:
            return int(jax.random.categorical(key, logits_row / temperature, axis=-1)[0])
        return int(jnp.argmax(logits_row, axis=-1)[0])

    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        ids = self.tok.encode(prompt)
        if len(ids) % self.chunk:                     # GDN-2 prefill needs L % C == 0
            ids = ids + self.tok.encode(" ") * (self.chunk - len(ids) % self.chunk)
        caches = self.model.init_cache(1, len(ids) + max_new_tokens)
        logits, caches = self.model.step(jnp.asarray([ids], jnp.int32), caches)
        last = logits[:, -1, :]

        toks: list[int] = []
        for _ in range(max_new_tokens):
            self.key, sub = jax.random.split(self.key)
            nxt = self._sample(last, temperature, sub)
            toks.append(nxt)
            logits, caches = self.model.step(jnp.asarray([[nxt]], jnp.int32), caches)
            last = logits[:, -1, :]
        return self.tok.decode(toks)


class FunctionGenerator:
    """Adapt any callable(prompt, temperature, max_new_tokens) -> str."""

    def __init__(self, fn: Callable[[str, float, int], str]):
        self.fn = fn

    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        return self.fn(prompt, temperature, max_new_tokens)


class OracleGenerator:
    """Returns each problem's reference solution — for validating the harness."""

    def __init__(self, problems):
        self._by_prompt = {p.prompt_for_model(): p.oracle_solution() for p in problems}

    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        return self._by_prompt.get(prompt, "")
