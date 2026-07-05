"""
Stage 4b — Copyright / license header removal (paper Sec. 2.1).

The paper notes over 15% of files carry copyright notices (e.g. "Copyright Intel
Corporation (C) 2014-2016") that are "highly repetitive and irrelevant to the
coding tasks". We strip a *leading* comment block if it is licence/copyright
boilerplate, across the common comment syntaxes:

  * line comments:  #  //  --  ;  %  (leading run of them)
  * block comments: /* ... */   <!-- ... -->   ''' ... '''   \"\"\" ... \"\"\"

We only remove the block if it actually contains a copyright/licence marker, and
only when it sits at the very top of the file (optionally after a shebang), so we
never touch real leading code or a genuine module docstring.
"""

from __future__ import annotations

import re

from ..config import TransformConfig
from ..models import CodeDocument

_MARKER = re.compile(
    r"copyright|\(c\)|©|licen[sc]e|licensed under|SPDX-License-Identifier|all rights reserved|"
    r"permission is hereby granted|redistribution and use",
    re.IGNORECASE,
)

_LINE_COMMENT_PREFIXES = ("#", "//", "--", ";", "%")
# (open, close) for block-comment styles.
_BLOCK_COMMENTS = (("/*", "*/"), ("<!--", "-->"), ('"""', '"""'), ("'''", "'''"))


def _strip_leading_block(text: str) -> tuple[str, bool]:
    lines = text.split("\n")
    i = 0
    # Preserve a shebang line.
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1

    start = i
    # Skip leading blank lines.
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Case 1: a run of single-line comments.
    for prefix in _LINE_COMMENT_PREFIXES:
        if i < len(lines) and lines[i].lstrip().startswith(prefix):
            j = i
            block = []
            while j < len(lines) and (
                lines[j].strip() == "" or lines[j].lstrip().startswith(prefix)
            ):
                block.append(lines[j])
                j += 1
            if _MARKER.search("\n".join(block)):
                return "\n".join(lines[:start] + lines[j:]).lstrip("\n"), True
            return text, False

    # Case 2: a block comment.
    for open_tok, close_tok in _BLOCK_COMMENTS:
        if i < len(lines) and lines[i].lstrip().startswith(open_tok):
            joined_from = "\n".join(lines[i:])
            end = joined_from.find(close_tok, len(open_tok))
            if end == -1:
                continue
            block = joined_from[: end + len(close_tok)]
            if _MARKER.search(block):
                remainder = joined_from[end + len(close_tok):]
                head = "\n".join(lines[:start])
                out = (head + "\n" + remainder) if head else remainder
                return out.lstrip("\n"), True
            return text, False

    return text, False


def remove_copyright(doc: CodeDocument, cfg: TransformConfig) -> CodeDocument:
    new_text, removed = _strip_leading_block(doc.content)
    if removed:
        doc.content = new_text
        doc.meta["copyright_header_removed"] = True
    return doc
