"""
Data iterators for the three training phases.

* Pretraining / annealing — `pretrain_batches`: concatenate document text into one
  token stream (documents separated by a newline), pack into fixed `seq_len`
  blocks, and predict every next token (loss weight 1 everywhere). This mirrors
  the paper's "randomly concatenated and segmented into chunks of context length".

* SFT — `sft_batches`: format each instruction/response into the chat template,
  tokenize, pad/truncate to `seq_len`, and mask the loss so only the **response**
  tokens are trained on (prompt + padding get weight 0). Standard instruction-
  tuning practice; the template mirrors `sft/format.py`.

`seq_len` must be a multiple of the model's `gdn_chunk_size` (the GDN-2 chunkwise
core reshapes L into L/C chunks) — enforced by the caller. Both iterators cycle
their input forever so you can run any number of steps on small corpora.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np

# Chat template pieces — kept in sync with sft/format.py's TEMPLATE.
_PROMPT_TMPL = "<|system|>\n{system}\n<|user|>\n{instruction}\n<|assistant|>\n"
_RESP_TMPL = "{response}\n<|end|>"
DEFAULT_SYSTEM = "You are a helpful programming assistant."


def _read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
#  Pretraining / annealing
# --------------------------------------------------------------------------- #
def _byte_stream(paths: list[str], tok, field: str) -> Iterator[int]:
    # Separate documents with the tokenizer's end-of-text token if it has one
    # (a real BPE tokenizer), else a newline (byte tokenizer).
    eos = getattr(tok, "eos_id", None)
    sep = [eos] if eos is not None else tok.encode("\n")
    for p in paths:
        for rec in _read_jsonl(p):
            text = rec.get(field) or rec.get("text") or ""
            yield from tok.encode(text)
            yield from sep


def pretrain_batches(
    paths: list[str], tok, seq_len: int, batch_size: int, field: str = "content"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (input_ids[B,L] int32, loss_weights[B,L] float32) forever."""
    while True:  # cycle epochs
        buf: list[int] = []
        batch: list[list[int]] = []
        for t in _byte_stream(paths, tok, field):
            buf.append(t)
            if len(buf) == seq_len:
                batch.append(buf)
                buf = []
                if len(batch) == batch_size:
                    ids = np.asarray(batch, dtype=np.int32)
                    yield ids, np.ones_like(ids, dtype=np.float32)
                    batch = []
        # remainder (partial block / partial batch) dropped; restart the stream


# --------------------------------------------------------------------------- #
#  SFT (with response-only loss masking)
# --------------------------------------------------------------------------- #
def _sft_example(rec: dict, tok, system: str) -> tuple[list[int], list[float]]:
    instr = rec.get("instruction") or rec.get("prompt") or rec.get("input") or ""
    resp = rec.get("response") or rec.get("output") or rec.get("completion") or ""
    prompt_ids = tok.encode(_PROMPT_TMPL.format(system=system, instruction=instr))
    resp_ids = tok.encode(_RESP_TMPL.format(response=resp))
    ids = prompt_ids + resp_ids
    mask = [0.0] * len(prompt_ids) + [1.0] * len(resp_ids)  # train on response only
    return ids, mask


def sft_batches(
    paths: list[str],
    tok,
    seq_len: int,
    batch_size: int,
    system: str = DEFAULT_SYSTEM,
    pad_id: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (input_ids[B,L] int32, loss_weights[B,L] float32) forever, padded to
    seq_len with response-only loss masking.

    NOTE: size `seq_len` to hold prompt + response. If the prompt alone exceeds
    seq_len the response is truncated away, every token is masked, and the batch
    contributes zero loss / zero gradient."""
    if pad_id is None:
        pad_id = getattr(tok, "pad_id", 0) or 0
    while True:  # cycle epochs
        bi: list[list[int]] = []
        bm: list[list[float]] = []
        for p in paths:
            for rec in _read_jsonl(p):
                ids, mask = _sft_example(rec, tok, system)
                ids, mask = ids[:seq_len], mask[:seq_len]
                pad = seq_len - len(ids)
                ids = ids + [pad_id] * pad
                mask = mask + [0.0] * pad
                bi.append(ids)
                bm.append(mask)
                if len(bi) == batch_size:
                    yield np.asarray(bi, np.int32), np.asarray(bm, np.float32)
                    bi, bm = [], []
