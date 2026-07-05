"""
Stage 5 — Heuristic filtering runner (paper App. A).

Ties the two steps together: compute quality signals, then execute the rules.
`filter_documents` yields only the survivors; `annotate_documents` yields every
document tagged with `filtered_by` (empty == kept) so you can inspect *why* files
were dropped — useful during the paper's iterative threshold-tuning loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..config import FilterConfig
from ..models import CodeDocument
from .quality_signals import compute_signals
from .rules import Rule, build_rules, execute_rules

__all__ = ["filter_documents", "annotate_documents", "build_rules", "Rule"]


def annotate_documents(
    docs: Iterable[CodeDocument], cfg: FilterConfig
) -> Iterator[CodeDocument]:
    rules = build_rules(cfg)
    for doc in docs:
        compute_signals(doc, cfg.code_long_word_char_len, cfg.long_string_word_count)
        doc.filtered_by = execute_rules(doc, rules)
        yield doc


def filter_documents(
    docs: Iterable[CodeDocument], cfg: FilterConfig
) -> Iterator[CodeDocument]:
    for doc in annotate_documents(docs, cfg):
        if not doc.filtered_by:
            yield doc
