"""
The unit of data that flows through every stage: `CodeDocument`.

One document == one source file (GitHub / The Stack v2 / web), carrying its raw
content plus the repository metadata the pipeline needs for dedup tie-breaking
(stars, commit time) and for language-aware filtering / downsampling.

Records are read from and written to JSONL, so `from_dict` / `to_dict` are the
serialization boundary. `signals` holds the per-file quality signals computed in
the filtering stage (paper App. A.1's "quality signal computation" step); it is
populated lazily and serialized alongside the doc for auditability.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class CodeDocument:
    content: str
    path: str = ""                      # original file path (used for extension)
    language: str = ""                  # resolved language (e.g. "Python")
    category: str = ""                  # "code" | "data" | "text" (paper App. E)
    repo_name: str = ""                 # "owner/repo"
    stars: int = 0                      # repo star count (dedup tie-break)
    commit_time: float = 0.0            # unix timestamp (dedup tie-break)
    source: str = "github"             # "github" | "the-stack-v2" | "web" ...
    doc_id: str = ""                    # stable id; falls back to hash if empty
    # Per-file quality signals (filled in during filtering); kept for audit.
    signals: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Rules that fired on this doc, if any (empty == passed all filters).
    filtered_by: list[str] = dataclasses.field(default_factory=list)
    # Free-form provenance the transform stage appends to (redaction counts, ...).
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8", errors="ignore"))

    def sort_key(self, keys: tuple[str, ...]) -> tuple:
        """Descending-preference key for dedup: higher stars / later commit win."""
        return tuple(getattr(self, k) for k in keys)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CodeDocument":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
