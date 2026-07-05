"""
N-gram utilities for SFT decontamination (OpenCoder Sec. 4.4): "we performed
10-gram deduplication, removing any data with a 10-gram overlap with the test
sets." Word-level n-grams, lightly normalized (lowercased, whitespace-collapsed)
so trivial formatting differences don't defeat the overlap check.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> list[str]:
    return _WS_RE.sub(" ", text.lower()).strip().split(" ")


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Set of word-level n-grams of `text` (empty if shorter than n words)."""
    words = normalize(text)
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_banned_ngrams(texts: list[str], n: int) -> set[tuple[str, ...]]:
    """Union of n-grams over a set of reference texts (e.g. HumanEval/MBPP)."""
    banned: set[tuple[str, ...]] = set()
    for t in texts:
        banned |= ngrams(t, n)
    return banned


def has_overlap(text: str, banned: set[tuple[str, ...]], n: int) -> bool:
    """True if any n-gram of `text` appears in `banned`."""
    if not banned:
        return False
    return not ngrams(text, n).isdisjoint(banned)
