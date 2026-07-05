"""
Post-processing model output into runnable code.

Two extraction modes:
  * `truncate_at_stops` / `extract_humaneval_completion` — HumanEval *completion*
    mode: the model continues a function signature, so we strip any echoed prompt
    and cut the continuation at the first token that starts a new top-level
    construct (the standard HumanEval truncation).
  * `extract_code` — instruct/chat mode: pull the code out of a fenced ```...```
    block (reuses synth_common's parser), or fall back to the whole text.
"""

from __future__ import annotations

from synth_common.teacher import extract_code_blocks

# Sequences that mark the end of a HumanEval completion (a new top-level thing).
_HUMANEVAL_STOPS = (
    "\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n@", "\n```",
    "\nassert ", "\nimport ", "\nfrom ",
)


def truncate_at_stops(text: str, stops: tuple[str, ...] = _HUMANEVAL_STOPS) -> str:
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def extract_humaneval_completion(prompt: str, generation: str) -> str:
    """Strip an echoed prompt, then truncate the continuation at the first stop."""
    g = generation
    if g.startswith(prompt):
        g = g[len(prompt):]
    return truncate_at_stops(g)


def extract_code(generation: str) -> str:
    """Return the last fenced code block, or the whole (stripped) text if none."""
    blocks = extract_code_blocks(generation)
    return blocks[-1] if blocks else generation.strip()
