"""
Quality-signal computation (paper App. A.1, step 1 of 2).

Following RedPajama (which the paper cites), heuristic filtering is split into
(1) computing a *score* per signal for each file, and (2) a separate execution
step that compares those scores against thresholds. This module is step (1): it
produces a flat dict of numeric/boolean signals per `CodeDocument`. It is pure
and side-effect-free w.r.t. content, so signals are cacheable and auditable.

The signals cover the three rule families in the paper:
  * generic text stats  (fractions of char classes, line-length stats),
  * general-code signals (Table 11: long-string ratios, hex ratio, placeholder
    ratio, assert ratio),
  * language-specific signals (Table 12, Python: def/line ratio, AST-parseable,
    import-line ratio).
"""

from __future__ import annotations

import ast
import re
from typing import Any

from ..models import CodeDocument

_PLACEHOLDER_RE = re.compile(
    r"(your code here|you code here|todo|fixme|xxx|implement me|not implemented)",
    re.IGNORECASE,
)
_STRING_LITERAL_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
_HEX_TOKEN_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b")


def _char_class_fractions(content: str) -> dict[str, float]:
    n = len(content) or 1
    alpha = digit = space = 0
    for c in content:
        if c.isalpha():
            alpha += 1
        elif c.isdigit():
            digit += 1
        elif c.isspace():
            space += 1
    return {
        "fraction_alphabetic": alpha / n,
        "fraction_numeric": digit / n,
        "fraction_whitespace": space / n,
    }


def _line_stats(lines: list[str]) -> dict[str, float]:
    lengths = [len(l) for l in lines] or [0]
    return {
        "num_lines": float(len(lines)),
        "mean_line_length": sum(lengths) / len(lengths),
        "max_line_length": float(max(lengths)),
    }


def _hex_char_ratio(content: str) -> float:
    if not content:
        return 0.0
    hex_chars = sum(len(m.group(0)) for m in _HEX_TOKEN_RE.finditer(content))
    return hex_chars / len(content)


def _string_signals(lines: list[str], long_word_char_len: int, long_line_words: int) -> dict[str, float]:
    """Table 11 string-related signals, approximated by scanning quoted spans."""
    total_chars = sum(len(l) for l in lines) or 1
    long_word_chars = 0
    long_string_lines = 0
    for line in lines:
        line_flagged = False
        for m in _STRING_LITERAL_RE.finditer(line):
            inner = m.group(0)[1:-1]
            words = inner.split()
            if len(words) > long_line_words:
                line_flagged = True
            for w in words:
                if len(w) > long_word_char_len:
                    long_word_chars += len(w)
        if line_flagged:
            long_string_lines += 1
    return {
        "long_string_line_ratio": long_string_lines / (len(lines) or 1),
        "long_string_word_char_ratio": long_word_chars / total_chars,
    }


def _ratio_of_lines(lines: list[str], predicate) -> float:
    if not lines:
        return 0.0
    return sum(1 for l in lines if predicate(l)) / len(lines)


def _python_signals(content: str, lines: list[str]) -> dict[str, Any]:
    def_lines = sum(1 for l in lines if l.lstrip().startswith("def ") or l.lstrip().startswith("async def "))
    n_lines = len(lines) or 1
    import_ratio = _ratio_of_lines(
        lines, lambda l: l.lstrip().startswith("import ") or l.lstrip().startswith("from ")
    )
    try:
        ast.parse(content)
        parseable = True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        parseable = False
    return {
        "python_func_line_ratio": def_lines / n_lines,
        "python_ast_parseable": parseable,
        "python_import_line_ratio": import_ratio,
    }


def compute_signals(doc: CodeDocument, long_word_char_len: int = 20, long_line_words: int = 12) -> dict[str, Any]:
    """Compute and return the full quality-signal dict for `doc` (also stored on it)."""
    content = doc.content
    lines = content.split("\n")

    signals: dict[str, Any] = {}
    signals.update(_char_class_fractions(content))
    signals.update(_line_stats(lines))
    signals["hex_char_ratio"] = _hex_char_ratio(content)
    signals.update(_string_signals(lines, long_word_char_len, long_line_words))
    signals["placeholder_line_ratio"] = _ratio_of_lines(
        lines, lambda l: bool(_PLACEHOLDER_RE.search(l))
    )
    signals["assert_line_ratio"] = _ratio_of_lines(lines, lambda l: "assert" in l)

    if doc.language == "Python":
        signals.update(_python_signals(content, lines))

    doc.signals = signals
    return signals
