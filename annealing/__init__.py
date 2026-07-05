"""
annealing — OpenCoder's end-of-pretraining annealing-data stage (Sec. 2.3,
Table 3).

Builds the ~100B-token, ~84%-RefineCode annealing mixture:

    build_annealing_data(refinecode_docs, seeds, teacher, cfg) -> (docs, MixReport)

Components:
  * algorithmic.py  — keyword-sampled Algorithmic Corpus (12.44B)
  * synthetic.py    — HQ code snippets, LLM-synthesized + test-validated (2.71B)
  * textbooks.py    — Code Textbooks, educational rewrites of HQ snippets (0.91B)
  * mixer.py        — assemble sources to a token budget at Table-3 proportions
  * config.py       — Table 3 proportions + WSD annealing schedule (Sec. 3.2)
"""

from __future__ import annotations

from collections.abc import Iterable

from data_pipeline.models import CodeDocument
from synth_common.teacher import MockTeacher, TeacherModel

from .algorithmic import sample_algorithmic
from .config import AnnealingConfig
from .mixer import MixReport, assemble_mixture
from .synthetic import synthesize_snippets
from .textbooks import synthesize_textbooks

__all__ = [
    "AnnealingConfig", "MixReport", "build_annealing_data",
    "sample_algorithmic", "synthesize_snippets", "synthesize_textbooks",
    "assemble_mixture",
]


def build_annealing_data(
    refinecode_docs: list[CodeDocument],
    snippet_seeds: Iterable[str],
    total_token_budget: int,
    cfg: AnnealingConfig | None = None,
    teacher: TeacherModel | None = None,
) -> tuple[list[CodeDocument], MixReport]:
    """End-to-end: build all four annealing sources and assemble the mixture.

    `refinecode_docs` is the cleaned pretraining corpus (output of data_pipeline);
    it feeds both the Original-Data share and the algorithmic-keyword sampling.
    `snippet_seeds` are code snippets used to synthesize HQ snippets + textbooks.
    """
    cfg = cfg or AnnealingConfig()
    teacher = teacher or MockTeacher()
    seeds = list(snippet_seeds)

    algorithmic = list(sample_algorithmic(iter(refinecode_docs), cfg.algorithmic))
    synthetic = list(synthesize_snippets(seeds, cfg.synthetic, teacher))
    textbooks = list(synthesize_textbooks(seeds, cfg.textbook, teacher))

    sources = {
        "refinecode": refinecode_docs,
        "algorithmic": algorithmic,
        "synthetic_snippet": synthetic,
        "code_textbook": textbooks,
    }
    return assemble_mixture(sources, total_token_budget, cfg.mix)
