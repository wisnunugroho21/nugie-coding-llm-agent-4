"""
Educational Instruction Synthesis (OpenCoder Sec. 4.1, Fig. 5b).

Key idea from the paper: "the educational value of the synthesized data largely
depends on the quality of the seed data", so a **scorer model** first selects
high-quality code seeds; only those are used to synthesize QA pairs. A teacher
then generates test cases for the code, which are executed — "Only the data
samples that successfully pass the tests are retained."

`ScorerModel` is pluggable (drop in a trained scorer); `HeuristicScorer` is a
transparent default so the pipeline runs offline. Yields test-validated
`InstructionExample`s tagged `source="educational_instruct"`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from synth_common.execution import validate_with_tests
from synth_common.prompts import answer_prompt, tests_prompt
from synth_common.teacher import MockTeacher, TeacherModel, extract_code_blocks

from .config import EducationalSynthConfig
from .models import InstructionExample


@runtime_checkable
class ScorerModel(Protocol):
    def score(self, snippet: str) -> float:
        """Return a quality score in [0, 1] for a code snippet."""
        ...


class HeuristicScorer:
    """A transparent stand-in for OpenCoder's learned seed scorer.

    Rewards snippets that parse, define functions, carry docstrings/comments, and
    have a reasonable size — proxies for 'educational' code. Replace with a trained
    scorer for production.
    """

    def score(self, snippet: str) -> float:
        s = 0.0
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            return 0.0
        s += 0.3  # parses
        has_func = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
        has_doc = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef))
            and ast.get_docstring(n)
            for n in ast.walk(tree)
        )
        if has_func:
            s += 0.3
        if has_doc or "#" in snippet:
            s += 0.2
        n_lines = snippet.count("\n") + 1
        if 5 <= n_lines <= 120:
            s += 0.2
        return min(s, 1.0)


def synthesize_educational(
    seed_snippets: Iterable[str],
    cfg: EducationalSynthConfig,
    teacher: TeacherModel | None = None,
    scorer: ScorerModel | None = None,
) -> Iterator[InstructionExample]:
    teacher = teacher or MockTeacher()
    scorer = scorer or HeuristicScorer()

    for snippet in seed_snippets:
        # Seed-quality gate: only high-quality seeds yield educational data.
        quality = scorer.score(snippet)
        if quality < cfg.min_seed_score:
            continue

        # Teacher writes an instructive solution + tests for the seed's concept.
        sol_blocks = extract_code_blocks(teacher.generate(answer_prompt(snippet, "Python")))
        if not sol_blocks:
            continue
        solution = sol_blocks[0]

        test_blocks = extract_code_blocks(teacher.generate(tests_prompt(snippet, solution, "Python")))
        if not test_blocks:
            continue
        if not validate_with_tests(solution, test_blocks[0], cfg.exec_timeout_s).passed:
            continue

        instruction = f"Study this snippet and write a well-documented solution:\n{snippet}"
        yield InstructionExample(
            instruction=instruction,
            response=solution,
            source="educational_instruct",
            language="Python",
            meta={"seed_quality": round(quality, 3), "validated": True},
        )
