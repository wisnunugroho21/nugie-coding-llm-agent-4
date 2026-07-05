"""
JSONL <-> CodeDocument serialization, the pipeline's on-disk boundary.

Each line of a `.jsonl` file is one record. `read_jsonl` streams documents lazily
so multi-GB shards never fully materialize; `write_jsonl` consumes an iterator and
returns the count written. Both transparently support gzip (`.jsonl.gz`).
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Iterator

from .models import CodeDocument


def _open(path: str, mode: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, mode.replace("t", "") + "b"), encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str) -> Iterator[CodeDocument]:
    with _open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield CodeDocument.from_dict(json.loads(line))


def write_jsonl(path: str, docs: Iterable[CodeDocument]) -> int:
    n = 0
    with _open(path, "wt") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
