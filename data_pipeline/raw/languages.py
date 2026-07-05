"""
Extension -> language resolution and language -> category classification.

The paper (App. E) keeps 607 languages, grouped into three categories — **code**
(rich in logic), **data** (structured), **text** (natural language) — because
"the threshold settings for the filtering rules vary slightly depending on the
data type." We reproduce that structure with a curated, representative mapping
(the eight languages that get language-specific rules in the paper, plus the
common long-tail). It is intentionally extensible: add rows to `EXT_TO_LANG` and
`LANGUAGE_CATEGORY` to widen coverage toward the full linguist set.
"""

from __future__ import annotations

import os

# Extension (lowercase, no dot) -> canonical language name.
EXT_TO_LANG: dict[str, str] = {
    # --- the 8 languages OpenCoder gives language-specific rules to ------------
    "py": "Python", "pyi": "Python", "pyx": "Python",
    "c": "C", "h": "C",
    "cc": "C++", "cpp": "C++", "cxx": "C++", "hpp": "C++", "hh": "C++",
    "cs": "C#",
    "java": "Java",
    "js": "JavaScript", "mjs": "JavaScript", "cjs": "JavaScript", "jsx": "JavaScript",
    "go": "Go",
    "html": "HTML", "htm": "HTML",
    # --- common long-tail 'code' -------------------------------------------------
    "ts": "TypeScript", "tsx": "TypeScript",
    "rb": "Ruby", "rs": "Rust", "php": "PHP", "swift": "Swift",
    "kt": "Kotlin", "kts": "Kotlin", "scala": "Scala", "sc": "Scala",
    "sh": "Shell", "bash": "Shell", "zsh": "Shell",
    "pl": "Perl", "pm": "Perl", "lua": "Lua", "r": "R",
    "dart": "Dart", "hs": "Haskell", "clj": "Clojure", "ex": "Elixir", "exs": "Elixir",
    "m": "Objective-C", "mm": "Objective-C++", "jl": "Julia",
    "sql": "SQL", "css": "CSS", "scss": "SCSS", "vue": "Vue", "svelte": "Svelte",
    # --- 'data' (structured) -----------------------------------------------------
    "json": "JSON", "yaml": "YAML", "yml": "YAML", "toml": "TOML",
    "xml": "XML", "csv": "CSV", "tsv": "TSV", "proto": "Protocol Buffer",
    "ini": "INI", "cfg": "INI",
    # --- 'text' (natural language) -----------------------------------------------
    "md": "Markdown", "markdown": "Markdown", "rst": "reStructuredText",
    "txt": "Text", "tex": "TeX",
}

# Language -> {"code","data","text"} (paper App. E's three-way split).
LANGUAGE_CATEGORY: dict[str, str] = {}
for _lang in {
    "Python", "C", "C++", "C#", "Java", "JavaScript", "Go", "HTML", "TypeScript",
    "Ruby", "Rust", "PHP", "Swift", "Kotlin", "Scala", "Shell", "Perl", "Lua", "R",
    "Dart", "Haskell", "Clojure", "Elixir", "Objective-C", "Objective-C++", "Julia",
    "SQL", "CSS", "SCSS", "Vue", "Svelte",
}:
    LANGUAGE_CATEGORY[_lang] = "code"
for _lang in {"JSON", "YAML", "TOML", "XML", "CSV", "TSV", "Protocol Buffer", "INI"}:
    LANGUAGE_CATEGORY[_lang] = "data"
for _lang in {"Markdown", "reStructuredText", "Text", "TeX"}:
    LANGUAGE_CATEGORY[_lang] = "text"


def detect_language(path: str) -> str | None:
    """Resolve a file path to a language via its extension, or None if unknown."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return EXT_TO_LANG.get(ext)


def language_category(language: str) -> str:
    """Return 'code' | 'data' | 'text' for a language (defaults to 'code')."""
    return LANGUAGE_CATEGORY.get(language, "code")
