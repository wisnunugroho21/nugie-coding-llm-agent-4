"""
Token estimation for mixture budgeting.

The annealing mixer (Sec. 2.3) assembles data to a *token* budget, so we need a
token count per document. The real pipeline would use OpenCoder's tokenizer
(vocab 96,640); here we provide a fast, dependency-free estimator and a pluggable
seam so you can swap in the real tokenizer:

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("infly/OpenCoder-1.5B")
    count = lambda s: len(tok.encode(s))

Pass any `Callable[[str], int]` as `counter` wherever a token count is needed.
"""

from __future__ import annotations

import re
from typing import Callable

TokenCounter = Callable[[str], int]

_TOKEN_RE = re.compile(r"\w+|[^\s\w]")


def estimate_tokens(text: str) -> int:
    """Heuristic token count: word/punctuation pieces (~GPT-style granularity).

    Correlates well enough with real BPE counts for budgeting; not exact.
    """
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))
