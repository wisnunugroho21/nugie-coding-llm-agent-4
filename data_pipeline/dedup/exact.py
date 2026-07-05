"""
Stage 2 — Exact deduplication (paper Sec. 2.1).

The paper reports ~75% of GitHub files are *exact* duplicates. We hash each
file's content (SHA256) and collapse every hash-collision group to a single
survivor: the copy with the **highest star count**, breaking ties by the **most
recent commit time** (the paper's stated retention rule).

This is a whole-corpus pass, so it materializes the input in memory grouped by
hash. For out-of-core scale you would shard by hash prefix; the logic per shard
is identical.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from ..config import ExactDedupConfig
from ..models import CodeDocument


def _content_hash(doc: CodeDocument, algo: str) -> str:
    h = hashlib.new(algo)
    h.update(doc.content.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def exact_dedup(
    docs: Iterable[CodeDocument], cfg: ExactDedupConfig
) -> Iterator[CodeDocument]:
    """Yield one survivor per exact-content group (order not preserved)."""
    best: dict[str, CodeDocument] = {}
    for doc in docs:
        key = _content_hash(doc, cfg.hash_algorithm)
        incumbent = best.get(key)
        if incumbent is None or doc.sort_key(cfg.keep_by) > incumbent.sort_key(cfg.keep_by):
            best[key] = doc
    yield from best.values()
