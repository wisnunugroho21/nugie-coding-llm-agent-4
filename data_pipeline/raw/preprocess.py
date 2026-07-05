"""
Stage 1 — Raw preprocessing / file-level admission (paper Sec. 2.1).

For every incoming file we:
  1. normalize the text (strip UTF-8 BOM, unify newlines to '\n'),
  2. resolve its language + category from the extension,
  3. drop it if it is too big (>8 MB), empty, or of an unknown language.

This is the cheapest possible gate and runs first so nothing downstream wastes
work on files that can never be admitted.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..config import PreprocessConfig
from ..models import CodeDocument
from .languages import detect_language, language_category


def _normalize(text: str) -> str:
    if text.startswith("﻿"):
        text = text[1:]  # strip BOM
    return text.replace("\r\n", "\n").replace("\r", "\n")


def preprocess(
    docs: Iterable[CodeDocument], cfg: PreprocessConfig
) -> Iterator[CodeDocument]:
    """Yield the admitted, normalized documents. Rejections are simply dropped."""
    for doc in docs:
        if cfg.normalize_newlines:
            doc.content = _normalize(doc.content)

        if len(doc.content) < cfg.min_content_chars:
            continue
        if doc.size_bytes > cfg.max_file_bytes:
            continue

        # Resolve language/category if not already provided.
        if not doc.language:
            lang = detect_language(doc.path)
            if lang is None:
                if cfg.require_known_language:
                    continue
                lang = "Unknown"
            doc.language = lang
        if not doc.category:
            doc.category = language_category(doc.language)

        yield doc
