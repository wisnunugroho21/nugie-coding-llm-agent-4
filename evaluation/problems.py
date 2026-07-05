"""
Benchmark problem types and how each turns a model generation into a runnable
program (the crux of a correct harness).

* `HumanEvalProblem` — the model completes a function signature+docstring. The
  runnable program is `prompt + completion`, followed by the problem's `test`
  (a `check(candidate)` function) and a `check(entry_point)` call. If the model
  instead emitted a full fenced function that defines `entry_point`, that is used
  directly (handles instruct-style models).

* `MBPPProblem` — the model writes a function for a task; the program is the
  (optional) setup code + the model's code + the problem's `assert` `test_list`.

Both expose `prompt_for_model()` (what the generator sees), `build_program(gen)`
(runnable source), and `oracle_solution()` (the reference, for harness self-test).
"""

from __future__ import annotations

import dataclasses

from synth_common.teacher import extract_code_blocks

from .extract import extract_code, extract_humaneval_completion


@dataclasses.dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str            # function signature + docstring
    entry_point: str       # the function name the tests call
    test: str              # defines `def check(candidate): ...`
    canonical_solution: str = ""   # reference *body* (continuation of prompt)

    def prompt_for_model(self) -> str:
        return self.prompt

    def oracle_solution(self) -> str:
        return self.canonical_solution

    def build_program(self, generation: str) -> str:
        # Prefer a full fenced function that defines the entry point, if present.
        full = next(
            (b for b in extract_code_blocks(generation) if f"def {self.entry_point}" in b),
            None,
        )
        body = full if full is not None else self.prompt + extract_humaneval_completion(
            self.prompt, generation
        )
        return f"{body}\n\n{self.test}\n\ncheck({self.entry_point})\n"


@dataclasses.dataclass
class MBPPProblem:
    task_id: str
    text: str                       # task description
    test_list: list[str]            # assert statements
    test_setup_code: str = ""       # optional imports/fixtures
    code: str = ""                  # reference solution (full function)

    def prompt_for_model(self) -> str:
        tests = "\n".join(self.test_list)
        return (
            f'"""{self.text}"""\n'
            f"# Your code must pass these tests:\n{tests}\n"
        )

    def oracle_solution(self) -> str:
        return self.code

    def build_program(self, generation: str) -> str:
        parts = []
        if self.test_setup_code:
            parts.append(self.test_setup_code)
        parts.append(extract_code(generation))
        parts.extend(self.test_list)
        return "\n".join(parts) + "\n"
