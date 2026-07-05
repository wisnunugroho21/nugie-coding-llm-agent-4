"""
The RefineCode orchestrator: runs every stage in the paper's order and reports
per-stage counts.

Stage order (paper Sec. 2.1 + App. A.1's "filter as late as possible"):

    raw preprocess           (Sec. 2.1)  -> size/extension/language admission
    exact dedup              (Sec. 2.1)  -> SHA256, ~75% of GitHub is exact dup
    fuzzy dedup (MinHash+LSH)(Sec. 2.1)  -> 5-gram, 2048 perms, 16x128 bands
    transform                (Sec. 2.1)  -> PII redaction + copyright removal
    heuristic filtering      (App. A)    -> quality signals then rule execution
    downsample               (Sec. 2.1)  -> cap high-resource languages

Each stage is a generator over `CodeDocument`, but dedup and downsample are
whole-corpus operations, so `run_pipeline` materializes between stages and
records `PipelineStats.stage_counts` (documents surviving each stage). For true
out-of-core scale you would shard by hash/language and run this per shard.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from .config import PipelineConfig
from .dedup.exact import exact_dedup
from .dedup.minhash import minhash_dedup
from .filtering import annotate_documents
from .models import CodeDocument
from .raw.preprocess import preprocess
from .sampling.downsample import downsample
from .transform import transform_documents


@dataclasses.dataclass
class PipelineStats:
    stage_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    # How many times each filter rule fired (paper's per-rule audit).
    rule_hits: dict[str, int] = dataclasses.field(default_factory=dict)

    def summary(self) -> str:
        lines = ["RefineCode pipeline — documents surviving each stage:"]
        for stage, n in self.stage_counts.items():
            lines.append(f"  {stage:<22} {n:>8}")
        if self.rule_hits:
            lines.append("Heuristic rule hits (files removed by each rule):")
            for rule, n in sorted(self.rule_hits.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {rule:<28} {n:>6}")
        return "\n".join(lines)


def run_pipeline(
    docs: Iterable[CodeDocument], cfg: PipelineConfig | None = None
) -> tuple[list[CodeDocument], PipelineStats]:
    """Run all stages; return (surviving documents, stats)."""
    cfg = cfg or PipelineConfig()
    stats = PipelineStats()

    stage = list(docs)
    stats.stage_counts["0_input"] = len(stage)

    stage = list(preprocess(stage, cfg.preprocess))
    stats.stage_counts["1_preprocess"] = len(stage)

    stage = list(exact_dedup(stage, cfg.exact_dedup))
    stats.stage_counts["2_exact_dedup"] = len(stage)

    stage = list(minhash_dedup(stage, cfg.minhash_dedup))
    stats.stage_counts["3_fuzzy_dedup"] = len(stage)

    stage = list(transform_documents(stage, cfg.transform))
    stats.stage_counts["4_transform"] = len(stage)

    # Heuristic filtering: annotate everything (so we can audit rule hits), then keep survivors.
    annotated = list(annotate_documents(stage, cfg.filtering))
    for doc in annotated:
        for rule in doc.filtered_by:
            stats.rule_hits[rule] = stats.rule_hits.get(rule, 0) + 1
    stage = [d for d in annotated if not d.filtered_by]
    stats.stage_counts["5_filter"] = len(stage)

    stage = list(downsample(stage, cfg.downsample))
    stats.stage_counts["6_downsample"] = len(stage)

    return stage, stats
