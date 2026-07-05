"""
Stage 4 — Transformation runner (paper Sec. 2.1).

Applies PII reduction then copyright/licence-header removal to each document, in
that order (redact secrets before we potentially delete the header that contains
a licence email, etc.). Both steps are individually toggleable via `TransformConfig`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..config import TransformConfig
from ..models import CodeDocument
from .copyright import remove_copyright
from .pii import redact_pii

__all__ = ["transform_documents", "redact_pii", "remove_copyright"]


def transform_documents(
    docs: Iterable[CodeDocument], cfg: TransformConfig
) -> Iterator[CodeDocument]:
    for doc in docs:
        if cfg.redact_pii:
            doc = redact_pii(doc, cfg)
        if cfg.remove_copyright:
            doc = remove_copyright(doc, cfg)
        yield doc
