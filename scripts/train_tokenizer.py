"""
Train a byte-level BPE tokenizer from the corpus (from scratch — our own vocab
from our own data), and save it to a single tokenizer.json.

    python scripts/train_tokenizer.py --data sample_data/refined.jsonl \
        --vocab-size 4000 --output sample_data/tokenizer.json

Use a large vocab (e.g. 32000, or OpenCoder's 96640) on the real corpus; the
sample corpus only supports a small vocab. Point training at the result with
`--tokenizer <path>` (the model's vocab_size is set from it automatically).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.tokenizer import BPETokenizer


def _texts(paths: list[str], field: str):
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get(field) or rec.get("text") or rec.get("content") or ""
                if text:
                    yield text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    ap.add_argument("--data", nargs="+", required=True, help="corpus JSONL path(s)")
    ap.add_argument("--field", default="content", help="text field (content/text)")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--output", default="tokenizer.json")
    args = ap.parse_args(argv)

    tok = BPETokenizer.train(
        _texts(args.data, args.field),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tok.save(args.output)

    sample = "def add(a, b):\n    return a + b\n"
    ids = tok.encode(sample)
    print(f"Trained BPE: vocab_size={tok.vocab_size}, eos_id={tok.eos_id}, pad_id={tok.pad_id}")
    print(f"Saved -> {args.output}")
    print(f"Sample encode ({len(ids)} tokens): {ids[:16]}...")
    print(f"Round-trip: {tok.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
