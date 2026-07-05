"""
Annealing mixture assembler (OpenCoder Sec. 2.3, Table 3).

Given the four named sources and a total token budget, draw documents from each
source until it reaches its target share of the budget (Table 3 proportions,
~84% RefineCode). Token counts come from a pluggable counter (default: the
heuristic estimator; swap in the real tokenizer for exact budgeting). Returns the
shuffled mixture plus a report of target-vs-actual token shares so you can see how
close the assembled data is to Table 3 (a source can fall short if it runs dry).
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterable

from data_pipeline.models import CodeDocument
from synth_common.token_count import TokenCounter, estimate_tokens

from .config import AnnealingMixConfig


@dataclasses.dataclass
class MixReport:
    total_tokens: int
    per_source_tokens: dict[str, int]
    per_source_docs: dict[str, int]
    target_proportions: dict[str, float]

    @property
    def actual_proportions(self) -> dict[str, float]:
        t = self.total_tokens or 1
        return {k: v / t for k, v in self.per_source_tokens.items()}

    def summary(self) -> str:
        lines = [f"Annealing mixture — {self.total_tokens:,} tokens across "
                 f"{sum(self.per_source_docs.values()):,} docs:"]
        for src in self.target_proportions:
            tgt = self.target_proportions[src]
            act = self.actual_proportions.get(src, 0.0)
            lines.append(f"  {src:<18} target {tgt:6.1%}  actual {act:6.1%}  "
                         f"({self.per_source_tokens.get(src, 0):,} tok, "
                         f"{self.per_source_docs.get(src, 0):,} docs)")
        return "\n".join(lines)


def assemble_mixture(
    sources: dict[str, Iterable[CodeDocument]],
    total_token_budget: int,
    cfg: AnnealingMixConfig,
    counter: TokenCounter = estimate_tokens,
) -> tuple[list[CodeDocument], MixReport]:
    """Assemble the annealing mixture to `total_token_budget` tokens.

    `sources` keys should match `cfg.proportions` keys (refinecode, algorithmic,
    synthetic_snippet, code_textbook). Extra/missing keys are tolerated: any
    source without a target proportion is skipped.
    """
    rng = random.Random(cfg.seed)
    chosen: list[CodeDocument] = []
    per_tok: dict[str, int] = {}
    per_doc: dict[str, int] = {}

    for src, docs in sources.items():
        target_prop = cfg.proportions.get(src, 0.0)
        target_tokens = int(round(target_prop * total_token_budget))
        acc = 0
        n = 0
        for doc in docs:
            if acc >= target_tokens:
                break
            chosen.append(doc)
            tk = counter(doc.content)
            acc += tk
            n += 1
        per_tok[src] = acc
        per_doc[src] = n

    rng.shuffle(chosen)
    report = MixReport(
        total_tokens=sum(per_tok.values()),
        per_source_tokens=per_tok,
        per_source_docs=per_doc,
        target_proportions=dict(cfg.proportions),
    )
    return chosen, report
