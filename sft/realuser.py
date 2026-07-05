"""
RealUser-Instruct construction (OpenCoder Sec. 4.1).

Paper procedure: sample real user queries from WildChat and Code-290k-ShareGPT,
"extracting code-related dialogue histories using LLM and subsequently performing
data cleaning. For low-quality responses, we employ a robust LLM to regenerate
the content." The result aligns with real-world problem complexity.

Implemented as: filter dialogues to code-related ones, clean them, and regenerate
responses that look low-quality (too short / no code) via the teacher. Input is an
iterable of raw dialogue dicts with at least `instruction` and `response` keys
(adapt your WildChat/ShareGPT loader to emit that shape). Yields
`source="realuser_instruct"`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from synth_common.prompts import answer_prompt
from synth_common.teacher import MockTeacher, TeacherModel, extract_code_blocks

from .models import InstructionExample

_CODE_HINT_RE = re.compile(
    r"```|def |class |import |function |const |public |#include|console\.|print\(",
    re.IGNORECASE,
)


def _is_code_related(instruction: str, response: str) -> bool:
    return bool(_CODE_HINT_RE.search(instruction) or _CODE_HINT_RE.search(response))


def _is_low_quality(response: str) -> bool:
    # Too short, or claims to be code-related yet contains no code.
    if len(response.strip()) < 40:
        return True
    return False


def build_realuser(
    dialogues: Iterable[dict],
    teacher: TeacherModel | None = None,
    min_response_chars: int = 40,
) -> Iterator[InstructionExample]:
    teacher = teacher or MockTeacher()
    for d in dialogues:
        instruction = (d.get("instruction") or "").strip()
        response = (d.get("response") or "").strip()
        if not instruction:
            continue
        if not _is_code_related(instruction, response):
            continue

        regenerated = False
        if len(response) < min_response_chars or _is_low_quality(response):
            blocks = extract_code_blocks(teacher.generate(answer_prompt(instruction, "Python")))
            if blocks:
                response = blocks[0]
                regenerated = True
        if not response:
            continue

        yield InstructionExample(
            instruction=instruction,
            response=response,
            source="realuser_instruct",
            language=d.get("language", "Python"),
            meta={"regenerated": regenerated, "origin": d.get("origin", "realuser")},
        )
