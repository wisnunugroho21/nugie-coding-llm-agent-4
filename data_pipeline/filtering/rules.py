"""
Rule definitions + filtering execution (paper App. A, step 2 of 2).

A `Rule` is a declarative record: which quality signal it reads, the comparison
operator, the threshold, and the scope (which document categories / languages it
applies to). `execute_rules` compares each in-scope rule's signal against its
threshold and returns the list of rule names that fired — a non-empty list means
"drop this file".

The rule set is organised into the paper's three families:
  * natural-language rules  -> 'text' category files,
  * general-code rules      -> 'code' and 'data' files (Table 11, exact numbers),
  * language-specific rules -> per-language (Table 12, Python, exact numbers).

Thresholds come from `FilterConfig`, so tuning is a config change, not a code
change — matching the paper's iterative "coarse then fine-grained" tuning loop.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Callable

from ..config import FilterConfig
from ..models import CodeDocument

# Comparison operators keyed by the symbol used in the paper's "Filtering Quota".
_OPS: dict[str, Callable[[object, object], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclasses.dataclass(frozen=True)
class Rule:
    name: str
    signal: str                 # key into doc.signals
    op: str                     # one of _OPS
    threshold: object           # compared against the signal value
    family: str                 # "natural_language" | "general_code" | "language_specific"
    categories: tuple[str, ...] = ()   # empty == any category
    languages: tuple[str, ...] = ()    # empty == any language
    description: str = ""

    def applies_to(self, doc: CodeDocument) -> bool:
        if self.categories and doc.category not in self.categories:
            return False
        if self.languages and doc.language not in self.languages:
            return False
        return True

    def fires(self, signals: dict) -> bool:
        if self.signal not in signals:
            return False
        return _OPS[self.op](signals[self.signal], self.threshold)


def build_rules(cfg: FilterConfig) -> list[Rule]:
    """Materialize the concrete rule set from thresholds in `cfg`."""
    f = cfg
    rules = [
        # ---- Natural-language rules (text files) ----------------------------
        Rule("nl_low_alpha", "fraction_alphabetic", "<", f.nl_min_alpha_frac,
             "natural_language", categories=("text",),
             description="Too few alphabetic chars for natural-language text."),
        Rule("nl_high_numeric", "fraction_numeric", ">", f.nl_max_numeric_frac,
             "natural_language", categories=("text",),
             description="Mostly digits — likely a data dump, not prose."),
        Rule("nl_too_few_lines", "num_lines", "<", f.nl_min_lines,
             "natural_language", categories=("text",),
             description="Trivially short document."),
        Rule("nl_long_mean_line", "mean_line_length", ">", f.nl_max_mean_line_len,
             "natural_language", categories=("text",),
             description="Unwrapped / minified-looking text."),
        Rule("nl_long_max_line", "max_line_length", ">", f.nl_max_line_len,
             "natural_language", categories=("text",),
             description="Contains an extremely long line."),

        # ---- General-code rules — Table 11 (exact paper thresholds) ---------
        Rule("code_long_string_lines", "long_string_line_ratio", ">", f.code_max_long_string_line_ratio,
             "general_code", categories=("code", "data"),
             description="Table 11: too many long-string lines -> lacks code logic."),
        Rule("code_long_string_chars", "long_string_word_char_ratio", ">", f.code_max_long_word_char_ratio,
             "general_code", categories=("code", "data"),
             description="Table 11: long in-string words (base64/hash/url-like)."),
        Rule("code_high_hex", "hex_char_ratio", ">", f.code_max_hex_char_ratio,
             "general_code", categories=("code", "data"),
             description="Table 11: too many hexadecimal chars -> lacks code logic."),
        Rule("code_placeholder_lines", "placeholder_line_ratio", ">", f.code_max_placeholder_line_ratio,
             "general_code", categories=("code", "data"),
             description="Table 11: TODO/FIXME/'your code here' placeholder spam."),
        Rule("code_assert_lines", "assert_line_ratio", ">", f.code_max_assert_line_ratio,
             "general_code", categories=("code", "data"),
             description="Table 11: assert-heavy -> likely a repetitive test file."),
        # General-code guards described in the paper text (line count / length).
        Rule("code_too_few_lines", "num_lines", "<", f.code_min_lines,
             "general_code", categories=("code", "data"),
             description="Trivially short code file."),
        Rule("code_long_mean_line", "mean_line_length", ">", f.code_max_mean_line_len,
             "general_code", categories=("code", "data"),
             description="Minified / generated -> abnormally long mean line."),
        Rule("code_long_max_line", "max_line_length", ">", f.code_max_line_len,
             "general_code", categories=("code", "data"),
             description="Contains an extremely long line (minified/generated)."),
        Rule("code_low_alpha", "fraction_alphabetic", "<", f.code_min_alpha_frac,
             "general_code", categories=("code",),
             description="Too few alphabetic chars for real source code."),

        # ---- Language-specific rules — Table 12 (Python, exact thresholds) --
        Rule("py_func_ratio", "python_func_line_ratio", ">", f.py_max_func_line_ratio,
             "language_specific", languages=("Python",),
             description="Table 12: #funcs/#lines too high -> overly trivial funcs."),
        Rule("py_import_ratio", "python_import_line_ratio", ">", f.py_max_import_line_ratio,
             "language_specific", languages=("Python",),
             description="Table 12: import-line ratio too high -> sparse logic."),
    ]

    # Table 12: drop Python files that do not parse into an AST (fires when the
    # 'python_ast_parseable' signal == False). Only added when enabled in config.
    if f.py_require_ast_parse:
        rules.append(
            Rule("py_not_parseable", "python_ast_parseable", "==", False,
                 "language_specific", languages=("Python",),
                 description="Table 12: file does not parse into a Python AST.")
        )
    return rules


def execute_rules(doc: CodeDocument, rules: list[Rule]) -> list[str]:
    """Return the names of all in-scope rules that fire on `doc.signals`."""
    fired = []
    for rule in rules:
        if rule.applies_to(doc) and rule.fires(doc.signals):
            fired.append(rule.name)
    return fired
