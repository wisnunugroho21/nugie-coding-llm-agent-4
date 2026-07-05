"""
Byte-level tokenizer (vocab 256) — matches the model's tiny default
`vocab_size=256` so the whole training loop runs on a laptop CPU with no external
tokenizer. Every byte is a token; nothing is out-of-vocab.

For a real run, swap in OpenCoder's tokenizer (vocab 96,640) — anything exposing
`encode(str) -> list[int]` / `decode(list[int]) -> str` and a `vocab_size` plugs
straight into `training/data.py` and the model's `KimiLinearConfig.vocab_size`.
"""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")
