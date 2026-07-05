"""
Stage 3 — Fuzzy (near-)deduplication via MinHash + LSH (paper Sec. 2.1).

Paper configuration:
  * 5-gram shingles over the file's tokens,
  * 2048 MinHash permutations,
  * LSH banded into 16 bands x 128 rows (16*128 == 2048).

Two files land in the same LSH bucket iff at least one of their 16 band-signatures
matches; with these parameters the implied Jaccard threshold is ~0.98, so only
*very* near-identical files collapse. Candidate pairs from any shared bucket are
unioned (connected components), and each component keeps a single representative
using the same stars-then-commit_time rule as exact dedup.

The paper chose **file-level** dedup over repo-level and chunk-level (App. B,
Table 13): file-level removed the most redundancy and gave the best HumanEval /
MBPP Pass@1. This module therefore operates on individual files.

Implementation uses numpy for the min-hashing; if numpy is unavailable it falls
back to a slower pure-Python path so the pipeline still runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from ..config import MinHashDedupConfig
from ..models import CodeDocument

try:  # numpy is strongly recommended but not strictly required.
    import numpy as _np
except Exception:  # pragma: no cover - fallback path
    _np = None

# Large prime for the universal-hash family h(x) = (a*x + b) mod p.
_MERSENNE_P = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _shingles(content: str, n: int) -> list[int]:
    """32-bit hashes of the n-gram (token) shingles of `content`."""
    tokens = content.split()
    if len(tokens) < n:
        # Too short for an n-gram — hash the whole (normalized) token stream once
        # so identical tiny files still collide.
        tokens = tokens or [content.strip()]
        grams = [" ".join(tokens)] if tokens != [""] else []
    else:
        grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    out = set()
    for g in grams:
        digest = hashlib.blake2b(g.encode("utf-8", errors="ignore"), digest_size=8).digest()
        out.add(int.from_bytes(digest, "big") & _MAX_HASH)
    return list(out)


class _Permutations:
    """Reusable (a, b) coefficients for `num_perm` independent hash functions."""

    def __init__(self, num_perm: int, seed: int = 1):
        import random

        rng = random.Random(seed)
        self.a = [rng.randrange(1, _MERSENNE_P) for _ in range(num_perm)]
        self.b = [rng.randrange(0, _MERSENNE_P) for _ in range(num_perm)]
        if _np is not None:
            self.a_np = _np.array(self.a, dtype=_np.uint64)
            self.b_np = _np.array(self.b, dtype=_np.uint64)


def _signature(shingle_hashes: list[int], perms: _Permutations, num_perm: int) -> tuple[int, ...]:
    """MinHash signature: for each permutation, the min over all shingles."""
    if not shingle_hashes:
        return tuple([_MERSENNE_P] * num_perm)  # empty doc -> constant sig
    if _np is not None:
        h = _np.array(shingle_hashes, dtype=_np.uint64)[:, None]      # [S, 1]
        vals = (perms.a_np[None, :] * h + perms.b_np[None, :]) % _MERSENNE_P  # [S, P]
        return tuple(int(x) for x in vals.min(axis=0))
    # Pure-Python fallback.
    sig = []
    for a, b in zip(perms.a, perms.b):
        sig.append(min((a * x + b) % _MERSENNE_P for x in shingle_hashes))
    return tuple(sig)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def minhash_dedup(
    docs: Iterable[CodeDocument], cfg: MinHashDedupConfig
) -> Iterator[CodeDocument]:
    """Yield one representative per near-duplicate cluster."""
    docs = list(docs)
    n = len(docs)
    if n <= 1:
        yield from docs
        return

    perms = _Permutations(cfg.num_perm)
    signatures: list[tuple[int, ...]] = [
        _signature(_shingles(d.content, cfg.ngram), perms, cfg.num_perm) for d in docs
    ]

    # LSH: bucket by each band; union any two docs sharing a band bucket.
    uf = _UnionFind(n)
    for band in range(cfg.bands):
        lo, hi = band * cfg.rows, (band + 1) * cfg.rows
        buckets: dict[tuple, int] = {}
        for i, sig in enumerate(signatures):
            key = (band,) + sig[lo:hi]
            key = hashlib.blake2b(repr(key).encode(), digest_size=16).digest()
            if key in buckets:
                uf.union(buckets[key], i)
            else:
                buckets[key] = i

    # Collapse each connected component to its best representative.
    best_by_root: dict[int, int] = {}
    for i, doc in enumerate(docs):
        root = uf.find(i)
        cur = best_by_root.get(root)
        if cur is None or doc.sort_key(cfg.keep_by) > docs[cur].sort_key(cfg.keep_by):
            best_by_root[root] = i

    for idx in best_by_root.values():
        yield docs[idx]
