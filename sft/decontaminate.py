"""
SFT decontamination (OpenCoder Sec. 4.4).

Two filters against evaluation leakage:
  1. **Entry-point removal:** drop any example that contains the entry-point
     function name of a test-set problem (HumanEval / MBPP).
  2. **10-gram overlap:** drop any example sharing a 10-gram with the test sets.

Provide the reference material via `TestSetReference` (the problem prompts/texts
used for n-gram banning, plus the list of entry-point names). Returns the clean
examples and a `DecontamReport`.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable

from synth_common.ngram import build_banned_ngrams, has_overlap

from .config import DecontamConfig
from .models import InstructionExample


@dataclasses.dataclass
class TestSetReference:
    texts: list[str]          # problem prompts / canonical solutions to ban n-grams from
    entry_points: list[str]   # function names, e.g. ["has_close_elements", "truncate_number"]


@dataclasses.dataclass
class DecontamReport:
    kept: int
    removed_entry_point: int
    removed_ngram: int

    def summary(self) -> str:
        return (f"Decontamination: kept {self.kept}, removed "
                f"{self.removed_entry_point} (entry-point) + "
                f"{self.removed_ngram} (10-gram overlap)")


def decontaminate(
    examples: Iterable[InstructionExample],
    reference: TestSetReference,
    cfg: DecontamConfig,
) -> tuple[list[InstructionExample], DecontamReport]:
    banned = build_banned_ngrams(reference.texts, cfg.ngram)
    # Word-boundary patterns for entry-point names.
    ep_patterns = [re.compile(rf"\b{re.escape(ep)}\b") for ep in reference.entry_points]

    kept: list[InstructionExample] = []
    removed_ep = removed_ng = 0
    for ex in examples:
        text = ex.text
        if cfg.remove_entry_point_matches and any(p.search(text) for p in ep_patterns):
            removed_ep += 1
            continue
        if has_overlap(text, banned, cfg.ngram):
            removed_ng += 1
            continue
        kept.append(ex)
    return kept, DecontamReport(len(kept), removed_ep, removed_ng)
