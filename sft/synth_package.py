"""
Package-related Instruction Synthesis (OpenCoder Sec. 4.1, Fig. 5c).

Motivation (paper): pretraining data contains outdated library usage, so the
model may emit obsolete API calls. Fix: "retrieved API signatures and usage
examples for widely used syntax and tools via PyDoc [and] prompt a teacher model
that generated accurate and up-to-date question-answer pairs."

This module does the **real PyDoc/inspect retrieval** — it imports each configured
library and reads live `inspect.signature` + docstrings from the installed
version — then asks the teacher to write up-to-date QA. Because the signatures
come from the actually-installed packages, the resulting instructions reflect
current APIs by construction. Yields `source="package_instruct"`.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator

from synth_common.prompts import package_qa_prompt
from synth_common.teacher import MockTeacher, TeacherModel, extract_code_blocks

from .config import PackageSynthConfig
from .models import InstructionExample


def _public_apis(library: str, limit: int) -> list[tuple[str, str, str]]:
    """Return up to `limit` (api_name, signature, first_doc_line) for a library."""
    try:
        mod = importlib.import_module(library)
    except ImportError:
        return []
    out: list[tuple[str, str, str]] = []
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if not callable(obj):
            continue
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "(...)"  # C-implemented builtins often expose no signature
        doc = (inspect.getdoc(obj) or "").strip().split("\n")[0]
        out.append((name, sig, doc))
        if len(out) >= limit:
            break
    return out


def synthesize_package(
    cfg: PackageSynthConfig,
    teacher: TeacherModel | None = None,
) -> Iterator[InstructionExample]:
    teacher = teacher or MockTeacher()
    for library in cfg.libraries:
        for api, signature, doc in _public_apis(library, cfg.max_apis_per_library):
            instruction = (
                f"Using the current version of `{library}`, show how to use "
                f"`{library}.{api}{signature}` correctly, with a short example."
            )
            resp = teacher.generate(package_qa_prompt(library, api, signature, doc))
            blocks = extract_code_blocks(resp)
            response = blocks[0] if blocks else resp.strip()
            if not response:
                continue
            yield InstructionExample(
                instruction=instruction,
                response=response,
                source="package_instruct",
                language="Python",
                meta={"library": library, "api": api, "signature": signature},
            )
