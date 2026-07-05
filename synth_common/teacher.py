"""
The LLM interface used by every data-synthesis pipeline in the annealing and SFT
stages (OpenCoder Sec. 2.3 / Sec. 4.1 rely on a "teacher model" to generate
questions, answers, tests and educational rewrites).

`TeacherModel` is a tiny protocol — one `generate(prompt, ...)` call — so a real
backend (Claude via the Anthropic SDK, or a local vLLM server) drops in without
touching the pipelines. `MockTeacher` is a fully offline, deterministic stand-in
that returns *executable* code, so the whole synthesis + test-validation flow runs
and is testable with no API access. Swap it for a real client in production:

    from anthropic import Anthropic
    class ClaudeTeacher:
        def __init__(self): self.c = Anthropic()
        def generate(self, prompt, temperature=1.0, max_tokens=2048):
            msg = self.c.messages.create(
                model="claude-opus-4-8", max_tokens=max_tokens,
                temperature=temperature, messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


@runtime_checkable
class TeacherModel(Protocol):
    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        """Return the model's completion for `prompt`."""
        ...


def extract_code_blocks(text: str) -> list[str]:
    """Pull the bodies of ```...``` fenced code blocks out of an LLM response."""
    blocks = [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(text)]
    if blocks:
        return blocks
    # No fences -> treat the whole thing as one block if it looks like code.
    stripped = text.strip()
    return [stripped] if stripped else []


class MockTeacher:
    """Deterministic, offline teacher that emits runnable Python.

    It branches on intent markers the pipelines put in their prompts
    (see synth_common.prompts) so a single mock drives question -> answer ->
    tests -> refine coherently. A per-call counter varies the generated function
    so synthesized examples aren't all identical.
    """

    def __init__(self) -> None:
        self._n = 0
        self._fn = "solution_0"   # the function name of the most recent answer

    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        p = prompt.lower()

        if "[intent:question]" in p:
            return (
                "Write a Python function that returns the sum of the even numbers "
                "in a list `nums`."
            )
        if "[intent:tests]" in p:
            # Tests target the *current* answer's function name, so a
            # question -> answer -> tests round stays self-consistent and passes.
            fn = self._fn
            return (
                "```python\n"
                f"assert {fn}([1, 2, 3, 4]) == 6\n"
                f"assert {fn}([]) == 0\n"
                f"assert {fn}([1, 3, 5]) == 0\n"
                "```"
            )
        if "[intent:refine]" in p:
            # Echo the incoming solution with an explanatory comment prepended.
            body = extract_code_blocks(prompt)
            code = body[-1] if body else f"def {self._fn}(nums):\n    return 0"
            return "```python\n# Sums the even numbers in the input list.\n" + code + "\n```"
        if "[intent:educational]" in p:
            return (
                "# Understanding list comprehensions\n"
                "A list comprehension builds a list from an iterable in one expression. "
                "For example, `[x for x in nums if x % 2 == 0]` keeps only the even values."
            )
        # Default: an answer (solution) block. Bumps the counter and records the
        # function name so a subsequent [intent:tests] call can reference it.
        self._n += 1
        self._fn = f"solution_{self._n}"
        return (
            "```python\n"
            f"def {self._fn}(nums):\n"
            f"    return sum(n for n in nums if n % 2 == 0)\n"
            "```"
        )
