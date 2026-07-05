"""
Code textbooks (OpenCoder Sec. 2.3, Table 3: "Code Textbooks 0.91B").

The paper derives these from an "educational analysis of the hqcode dataset":
high-quality code snippets are rewritten by a teacher model into educational
prose that teaches the concept the code demonstrates. Output is `text`-category
data (natural language about code), tagged `source="code_textbook"`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from data_pipeline.models import CodeDocument
from synth_common.prompts import educational_prompt
from synth_common.teacher import MockTeacher, TeacherModel

from .config import TextbookConfig


def synthesize_textbooks(
    snippets: Iterable[str],
    cfg: TextbookConfig,
    teacher: TeacherModel | None = None,
) -> Iterator[CodeDocument]:
    teacher = teacher or MockTeacher()
    for i, snippet in enumerate(snippets):
        if len(snippet) < cfg.min_snippet_chars:
            continue
        passage = teacher.generate(
            educational_prompt(snippet), temperature=cfg.teacher_temperature
        ).strip()
        if not passage:
            continue
        yield CodeDocument(
            content=passage,
            path=f"textbook/lesson_{i}.md",
            language="Markdown",
            category="text",
            source="code_textbook",
            meta={"from_snippet_chars": len(snippet)},
        )
