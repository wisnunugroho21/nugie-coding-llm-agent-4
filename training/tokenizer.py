"""
Tokenizers.

* `ByteTokenizer` — vocab 256, every byte a token, zero-dependency. Matches the
  model's tiny default so the loop runs on a laptop CPU with nothing installed.

* `BPETokenizer` — a real **byte-level BPE**, trained from the corpus (our own
  vocab from our own data, from scratch — not a downloaded one). Backed by the
  Hugging Face `tokenizers` library (Rust); handles arbitrary bytes with no UNK
  (good for code), carries the chat special tokens (`<|system|>` … `<|end|>`), and
  serializes to a single `tokenizer.json`. This is what you'd train at vocab
  96,640 for a real run (OpenCoder's size).

Both expose the same interface — `encode(str) -> list[int]`, `decode(list[int]) ->
str`, `vocab_size` — plus optional `eos_id` / `pad_id`, so either plugs straight
into `training/data.py`, the model's `KimiLinearConfig.vocab_size`, and eval.
"""

from __future__ import annotations

from collections.abc import Iterable

# Chat special tokens (mirror the template in training/data.py & sft/format.py),
# plus end-of-text (doc separator / eos) and pad. Order fixes their ids: eot=0.
SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")


class ByteTokenizer:
    vocab_size = 256
    eos_id = None
    pad_id = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class BPETokenizer:
    """Byte-level BPE with chat special tokens (Hugging Face `tokenizers` backend)."""

    def __init__(self, tokenizer):
        self._tok = tokenizer                       # a tokenizers.Tokenizer
        self.eos_id = self._tok.token_to_id("<|endoftext|>")
        self.pad_id = self._tok.token_to_id("<|pad|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    # --- training -----------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a byte-level BPE on an iterator of text (our corpus)."""
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "BPETokenizer needs the tokenizers library: pip install tokenizers"
            ) from e

        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
            # Seed all 256 bytes so nothing is ever UNK.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    # --- inference ----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # --- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))

