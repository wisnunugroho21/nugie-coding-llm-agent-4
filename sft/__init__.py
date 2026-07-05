"""
sft — OpenCoder post-training / two-stage instruction tuning (Sec. 4).

Synthesis pipelines (Sec. 4.1, Fig. 5):
  * synth_diverse.py     — Large-scale Diverse Instruction Synthesis (Stage 1)
  * synth_educational.py — Educational Instruction Synthesis        (Stage 2)
  * synth_package.py     — Package-related Instruction Synthesis via PyDoc (Stage 2)
  * realuser.py          — RealUser-Instruct from WildChat/ShareGPT  (Stage 1)

Assembly:
  * decontaminate.py     — entry-point + 10-gram overlap removal (Sec. 4.4)
  * compose.py           — two-stage composition per Table 5 (Sec. 4.2)
  * format.py            — chat formatting -> training-ready text
  * config.py            — Table 5 quotas + Sec. 4.3 training hyperparameters

Open-source pools (Evol-Instruct, Infinity-Instruct, McEval-Instruct) are loaded,
not synthesized; use `wrap_examples` to adapt your loader's rows to
`InstructionExample`s with the right source tag.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .compose import ComposeReport, compose_two_stage
from .config import SFTConfig
from .decontaminate import DecontamReport, TestSetReference, decontaminate
from .format import format_dataset, format_example
from .models import InstructionExample
from .realuser import build_realuser
from .synth_diverse import synthesize_diverse
from .synth_educational import synthesize_educational
from .synth_package import synthesize_package

__all__ = [
    "InstructionExample", "SFTConfig", "wrap_examples",
    "synthesize_diverse", "synthesize_educational", "synthesize_package", "build_realuser",
    "decontaminate", "TestSetReference", "DecontamReport",
    "compose_two_stage", "ComposeReport",
    "format_example", "format_dataset",
]


def wrap_examples(
    rows: Iterable[dict], source: str, language: str = "Python"
) -> Iterator[InstructionExample]:
    """Adapt raw {'instruction'/'prompt', 'response'/'output'} rows from an
    open-source dataset (Evol/Infinity/McEval) into tagged InstructionExamples."""
    for r in rows:
        instruction = r.get("instruction") or r.get("prompt") or r.get("input") or ""
        response = r.get("response") or r.get("output") or r.get("completion") or ""
        if not instruction or not response:
            continue
        yield InstructionExample(
            instruction=instruction, response=response,
            source=source, language=r.get("language", language),
        )
