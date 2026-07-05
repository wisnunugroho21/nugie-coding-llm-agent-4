"""
Two-stage instruction-tuning composition (OpenCoder Sec. 4.2, Table 5).

Stage 1 (theory / breadth): RealUser-Instruct (0.7M) + Large-scale
Diverse-Instruct (2.3M) + Filtered Infinity-Instruct (1.0M).
Stage 2 (practical coding): McEval-Instruct (36K) + Evol-Instruct (111K) +
Educational-Instruct (110K) + Package-Instruct (110K).

`compose_two_stage` draws up to each source's Table-5 quota from the provided
example pools, tags each example with its stage, shuffles, and reports
target-vs-actual counts (a pool can fall short of its quota).
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterable

from .config import STAGE1_COUNTS, STAGE2_COUNTS, SFTConfig
from .models import InstructionExample


@dataclasses.dataclass
class ComposeReport:
    stage_targets: dict[int, dict[str, int]]
    stage_actuals: dict[int, dict[str, int]]

    def summary(self) -> str:
        lines = ["Two-stage SFT composition (target vs actual examples):"]
        for stage in (1, 2):
            lines.append(f"  Stage {stage}:")
            for src, tgt in self.stage_targets[stage].items():
                act = self.stage_actuals[stage].get(src, 0)
                lines.append(f"    {src:<28} target {tgt:>9,}  actual {act:>9,}")
        return "\n".join(lines)


def _draw(pool: list[InstructionExample], quota: int, stage: int, source: str,
          rng: random.Random) -> list[InstructionExample]:
    rng.shuffle(pool)
    drawn = pool[:quota]
    for ex in drawn:
        ex.stage = stage
        if not ex.source:
            ex.source = source
    return drawn


def compose_two_stage(
    pools: dict[str, Iterable[InstructionExample]],
    cfg: SFTConfig | None = None,
) -> tuple[dict[int, list[InstructionExample]], ComposeReport]:
    """`pools` maps Table-5 source names to example iterables. Missing sources
    simply contribute nothing (their actual count is 0)."""
    cfg = cfg or SFTConfig()
    rng = random.Random(cfg.seed)
    materialized = {k: list(v) for k, v in pools.items()}

    stages: dict[int, list[InstructionExample]] = {1: [], 2: []}
    actuals: dict[int, dict[str, int]] = {1: {}, 2: {}}
    for stage, counts in ((1, STAGE1_COUNTS), (2, STAGE2_COUNTS)):
        for source, quota in counts.items():
            drawn = _draw(materialized.get(source, []), quota, stage, source, rng)
            stages[stage].extend(drawn)
            actuals[stage][source] = len(drawn)
        rng.shuffle(stages[stage])

    report = ComposeReport(
        stage_targets={1: dict(STAGE1_COUNTS), 2: dict(STAGE2_COUNTS)},
        stage_actuals=actuals,
    )
    return stages, report
