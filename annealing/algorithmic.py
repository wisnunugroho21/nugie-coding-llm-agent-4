"""
Algorithmic corpus construction (OpenCoder Sec. 2.3, Table 3: "Algorithmic
Corpus 12.44B").

The paper builds this by keyword-sampling the pretraining corpus for
algorithm/competitive-programming content (e.g. "leetcode", "solution"). We
implement that as a case-insensitive keyword filter over each document's path +
content, tagging survivors with `source="algorithmic"` and how many keywords hit
(useful for ranking / quota control).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator

from data_pipeline.models import CodeDocument

from .config import AlgorithmicConfig


def sample_algorithmic(
    docs: Iterable[CodeDocument], cfg: AlgorithmicConfig
) -> Iterator[CodeDocument]:
    kws = [k.lower() for k in cfg.keywords]
    for doc in docs:
        hay = (doc.path + "\n" + doc.content).lower()
        hits = sum(1 for k in kws if k in hay)
        if hits >= cfg.min_keyword_hits:
            # Emit an independent copy (fresh meta) so we never mutate the shared
            # RefineCode objects — otherwise the same file could be counted in
            # both the 'refinecode' and 'algorithmic' shares of the mixture.
            copy = dataclasses.replace(
                doc,
                source="algorithmic",
                meta={**doc.meta, "algorithmic_keyword_hits": hits},
            )
            yield copy
