"""
High-quality synthetic code snippets with test validation
(OpenCoder Sec. 2.3, Table 3: "High Quality Code Snippet 2.71B").

For each seed snippet the teacher model produces an improved solution and a set
of test cases; the pair is executed and **only kept if the tests pass** — the
paper's "LLM-synthesized with test validation". This yields code that is both
syntactically and semantically sound.

The teacher is pluggable (see synth_common.teacher); with `MockTeacher` the whole
flow runs offline and every accepted sample is genuinely executed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from data_pipeline.models import CodeDocument
from synth_common.execution import validate_with_tests
from synth_common.prompts import answer_prompt, tests_prompt
from synth_common.teacher import MockTeacher, TeacherModel, extract_code_blocks

from .config import SyntheticConfig


def synthesize_snippets(
    seeds: Iterable[str],
    cfg: SyntheticConfig,
    teacher: TeacherModel | None = None,
) -> Iterator[CodeDocument]:
    teacher = teacher or MockTeacher()
    for i, seed in enumerate(seeds):
        # 1. Teacher writes an improved, self-contained solution.
        sol_resp = teacher.generate(
            answer_prompt(seed, "Python"), temperature=cfg.teacher_temperature
        )
        sol_blocks = extract_code_blocks(sol_resp)
        if not sol_blocks:
            continue
        solution = sol_blocks[0]

        # 2. Teacher writes test cases for that solution.
        test_resp = teacher.generate(
            tests_prompt(seed, solution, "Python"), temperature=cfg.teacher_temperature
        )
        test_blocks = extract_code_blocks(test_resp)
        if not test_blocks:
            continue
        tests = test_blocks[0]

        # 3. Execute; keep only if the tests pass (real validation).
        result = validate_with_tests(solution, tests, timeout=cfg.exec_timeout_s)
        if not result.passed:
            continue

        yield CodeDocument(
            content=solution,
            path=f"synthetic/snippet_{i}.py",
            language="Python",
            category="code",
            source="synthetic_snippet",
            meta={"validated": True, "tests": tests},
        )
