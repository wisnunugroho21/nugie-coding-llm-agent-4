"""
Code-related web-data recall (paper Sec. 2.2 / App. C.2).

RefineCode augments raw code with ~75B tokens of code-related *web* text. The
recall pipeline the paper describes:

  1. Filename recall (StarCoder-style, App. C.2): keep a file if 'requirement' is
     in the lowercased name, or the stem is one of readme/notes/todo/description/
     cmakelists. (~3% of the text volume.)
  2. fastText recall: train a fastText classifier on an annotated seed corpus of
     code-like vs. non-code web pages, then recall documents scored P(code) >= τ
     (~extra 7% of volume). Domains where >=10% of sampled pages are code-related
     are promoted to "code-related" and mined further.

This module implements (1) exactly and (2) with a real fastText model when the
`fasttext` package is installed, transparently falling back to a lightweight
bag-of-keywords logistic-style scorer so the pipeline runs anywhere. The public
interface — `train`, `predict_proba`, `recall` — is identical either way.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator

from ..config import WebRecallConfig
from ..models import CodeDocument

try:
    import fasttext as _fasttext
except Exception:  # pragma: no cover - optional dependency
    _fasttext = None

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

# Signals of code-bearing web text used by the fallback scorer's seed vocabulary.
_CODE_HINTS = {
    "def", "class", "function", "return", "import", "from", "public", "static",
    "void", "const", "var", "let", "print", "println", "console", "for", "while",
    "if", "else", "try", "except", "catch", "throw", "async", "await", "npm",
    "pip", "install", "git", "commit", "traceback", "error", "exception", "stack",
    "compile", "runtime", "argument", "parameter", "method", "variable", "array",
    "list", "dict", "struct", "pointer", "null", "none", "true", "false", "code",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def filename_recall(doc: CodeDocument, cfg: WebRecallConfig) -> bool:
    """StarCoder-style filename rule (App. C.2)."""
    import os

    name = os.path.basename(doc.path).lower()
    stem = os.path.splitext(name)[0]
    return cfg.filename_substring in name or stem in cfg.filename_stems


class FastTextRecaller:
    """Trainable code-vs-noncode web classifier (fastText, with pure-Python fallback)."""

    def __init__(self, cfg: WebRecallConfig):
        self.cfg = cfg
        self._model = None                 # a real fastText model, when available
        self._weights: dict[str, float] = {}   # fallback log-odds weights
        self._bias = 0.0

    # --- training ----------------------------------------------------------
    def train(self, examples: list[tuple[str, int]]) -> "FastTextRecaller":
        """`examples`: list of (text, label) with label 1 == code-related."""
        if _fasttext is not None:
            self._train_fasttext(examples)
        else:
            self._train_fallback(examples)
        return self

    def _train_fasttext(self, examples: list[tuple[str, int]]) -> None:  # pragma: no cover
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for text, label in examples:
                lbl = "__label__code" if label == 1 else "__label__other"
                fh.write(f"{lbl} {' '.join(_tokenize(text))}\n")
            path = fh.name
        self._model = _fasttext.train_supervised(input=path, epoch=25, wordNgrams=2)

    def _train_fallback(self, examples: list[tuple[str, int]]) -> None:
        """Naive-Bayes-style log-odds over tokens; seeded with code hints."""
        pos, neg = Counter(), Counter()
        n_pos = n_neg = 0
        for text, label in examples:
            toks = set(_tokenize(text))
            if label == 1:
                pos.update(toks); n_pos += 1
            else:
                neg.update(toks); n_neg += 1
        vocab = set(pos) | set(neg) | _CODE_HINTS
        for w in vocab:
            p = (pos[w] + 1) / (n_pos + 2)
            q = (neg[w] + 1) / (n_neg + 2)
            self._weights[w] = math.log(p) - math.log(q)
        self._bias = math.log((n_pos + 1) / (n_neg + 1))

    # --- inference ---------------------------------------------------------
    def predict_proba(self, text: str) -> float:
        if self._model is not None:  # pragma: no cover
            labels, probs = self._model.predict(" ".join(_tokenize(text)), k=2)
            for lbl, pr in zip(labels, probs):
                if lbl == "__label__code":
                    return float(pr)
            return 0.0
        score = self._bias + sum(self._weights.get(t, 0.0) for t in set(_tokenize(text)))
        return 1.0 / (1.0 + math.exp(-score))

    def recall(self, docs: Iterable[CodeDocument]) -> Iterator[CodeDocument]:
        """Yield web docs that pass the filename rule OR score P(code) >= threshold."""
        for doc in docs:
            keep = filename_recall(doc, self.cfg)
            if not keep:
                p = self.predict_proba(doc.content)
                doc.meta["code_prob"] = round(p, 4)
                keep = p >= self.cfg.fasttext_threshold
            if keep:
                yield doc
