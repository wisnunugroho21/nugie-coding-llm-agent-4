"""
Chat formatting — turn `InstructionExample`s into training-ready text.

Produces a single formatted string per example using a simple, explicit chat
template (system / user / assistant with role markers). Swap `TEMPLATE` for the
tokenizer's real chat template in production; the shape (one string per example,
plus a token-length estimate) is what the trainer consumes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from synth_common.token_count import TokenCounter, estimate_tokens

from .models import InstructionExample

DEFAULT_SYSTEM = "You are a helpful programming assistant."

TEMPLATE = "<|system|>\n{system}\n<|user|>\n{instruction}\n<|assistant|>\n{response}\n<|end|>"


def format_example(ex: InstructionExample, system: str = DEFAULT_SYSTEM) -> str:
    return TEMPLATE.format(system=system, instruction=ex.instruction.strip(),
                           response=ex.response.strip())


def format_dataset(
    examples: Iterable[InstructionExample],
    system: str = DEFAULT_SYSTEM,
    counter: TokenCounter = estimate_tokens,
) -> Iterator[dict]:
    """Yield {'text', 'tokens', 'source', 'stage'} records ready for tokenization."""
    for ex in examples:
        text = format_example(ex, system)
        yield {"text": text, "tokens": counter(text), "source": ex.source, "stage": ex.stage}
