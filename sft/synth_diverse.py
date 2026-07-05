"""
Large-scale Diverse Instruction Synthesis (OpenCoder Sec. 4.1, Fig. 5a).

The paper's four-component framework, implemented step-for-step:
  (1) clean irrelevant context from a web seed and keep useful sentences,
  (2) a task-specification module (languages, difficulty levels, task types) feeds
      a template-based prompt engine — questions generated at temperature T = 1.0,
  (3) an advanced LLM generates the question then the answer; a validation module
      runs the code against generated unit tests to check correctness,
  (4) an LLM refines the response by adding comments and explanation.

Yields test-validated `InstructionExample`s tagged `source="diverse_instruct"`.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Iterator

from synth_common.execution import validate_with_tests
from synth_common.prompts import answer_prompt, question_prompt, refine_prompt, tests_prompt
from synth_common.teacher import MockTeacher, TeacherModel, extract_code_blocks

from .config import DiverseSynthConfig
from .models import InstructionExample

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
_AD_MARKERS = ("subscribe", "advertisement", "cookie", "sign up", "buy now", "©")


def _clean_seed(seed: str) -> str:
    """Step 1: drop ad/boilerplate sentences, keep the useful ones."""
    sentences = _SENTENCE_RE.findall(seed) or [seed]
    kept = [s.strip() for s in sentences if not any(m in s.lower() for m in _AD_MARKERS)]
    return " ".join(kept).strip() or seed.strip()


def synthesize_diverse(
    web_seeds: Iterable[str],
    cfg: DiverseSynthConfig,
    teacher: TeacherModel | None = None,
    seed: int = 42,
) -> Iterator[InstructionExample]:
    teacher = teacher or MockTeacher()
    rng = random.Random(seed)

    for raw in web_seeds:
        clean = _clean_seed(raw)

        # Step 2: task specification + templated prompt (question at T = 1.0).
        language = rng.choice(cfg.languages)
        difficulty = rng.choice(cfg.difficulties)
        task_type = rng.choice(cfg.task_types)
        question = teacher.generate(
            question_prompt(clean, language, difficulty, task_type),
            temperature=cfg.question_temperature,
        ).strip()
        if not question:
            continue

        # Step 3: generate the answer, then validate it against generated tests.
        sol_blocks = extract_code_blocks(teacher.generate(answer_prompt(question, language)))
        if not sol_blocks:
            continue
        solution = sol_blocks[0]

        if cfg.validate_with_tests:
            test_blocks = extract_code_blocks(
                teacher.generate(tests_prompt(question, solution, language))
            )
            if not test_blocks:
                continue
            if not validate_with_tests(solution, test_blocks[0], cfg.exec_timeout_s).passed:
                continue

        # Step 4: refine the response (comments + explanation).
        refined_blocks = extract_code_blocks(teacher.generate(refine_prompt(question, solution)))
        response = refined_blocks[0] if refined_blocks else solution

        yield InstructionExample(
            instruction=question,
            response=response,
            source="diverse_instruct",
            language=language,
            meta={"difficulty": difficulty, "task_type": task_type, "validated": cfg.validate_with_tests},
        )
