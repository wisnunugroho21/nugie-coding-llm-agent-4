"""
The Stack v2 loader: stream metadata from Hugging Face, resolve file contents
from Software Heritage S3, and emit `CodeDocument`s ready for `data_pipeline`.

The two external dependencies (the HF stream and the S3 resolver) are injectable,
so the mapping logic is fully unit-testable offline — see tests/test_ingestion.py,
which drives the loader with a fake stream + fake resolver (no network, no creds).

Field mapping (Stack v2 row -> CodeDocument):
    content       <- S3 blob fetched by `blob_id`, decoded with `src_encoding`
    path          <- row["path"]
    language      <- the HF config name (Stack v2 is sharded per language)
    repo_name     <- row["repo_name"]
    stars         <- row["star_events_count"]           (dedup tie-break)
    commit_time   <- first available GHA/visit timestamp (dedup tie-break)
    doc_id        <- row["blob_id"]
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Iterator
from typing import Callable

from data_pipeline.models import CodeDocument
from data_pipeline.raw.languages import language_category

from .config import StackV2Config

# A content resolver maps (blob_id, src_encoding) -> decoded file text.
ContentResolver = Callable[[str, "str | None"], str]

# Row fields to try, in order, for a commit/observation timestamp.
_TIME_FIELDS = ("gha_updated_at", "gha_created_at", "revision_date", "committer_date", "visit_date")


def _parse_time(row: dict) -> float:
    """Best-effort unix timestamp from whichever time field the row provides."""
    for field in _TIME_FIELDS:
        val = row.get(field)
        if not val:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, _dt.datetime):
            return val.timestamp()
        try:
            s = str(val).replace("Z", "+00:00")
            return _dt.datetime.fromisoformat(s).timestamp()
        except ValueError:
            continue
    return 0.0


def row_to_document(
    row: dict, content: str, language: str, source: str = "the-stack-v2"
) -> CodeDocument:
    """Pure mapping from a Stack v2 metadata row + resolved content to a CodeDocument."""
    return CodeDocument(
        content=content,
        path=row.get("path", "") or "",
        language=language,
        category=language_category(language),
        repo_name=row.get("repo_name", "") or "",
        stars=int(row.get("star_events_count") or 0),
        commit_time=_parse_time(row),
        source=source,
        doc_id=str(row.get("blob_id", "") or ""),
        meta={
            "src_encoding": row.get("src_encoding"),
            "detected_licenses": row.get("detected_licenses"),
            "gha_license_id": row.get("gha_license_id"),
            "length_bytes": row.get("length_bytes"),
        },
    )


def build_s3_content_resolver(cfg: StackV2Config) -> ContentResolver:
    """Construct the real Software Heritage S3 resolver (needs boto3 + smart_open).

    Raises a clear, actionable error if those optional deps are missing, so the
    package imports and tests fine without them.
    """
    try:
        import boto3
        from smart_open import open as smart_open_open
    except ImportError as e:  # pragma: no cover - exercised only without the deps
        raise RuntimeError(
            "The Stack v2 stores file contents in Software Heritage S3. Install the "
            "ingestion extras to fetch them:\n"
            "    pip install boto3 smart_open[s3]\n"
            "and provide AWS credentials (env or ~/.aws). See data_ingestion/README.md."
        ) from e

    session = boto3.Session()
    s3 = session.client("s3")

    def resolve(blob_id: str, src_encoding: "str | None") -> str:
        url = f"s3://{cfg.s3_bucket}/{cfg.s3_key_prefix}/{blob_id}"
        with smart_open_open(
            url, "rb", compression=cfg.s3_compression, transport_params={"client": s3}
        ) as fin:
            return fin.read().decode(src_encoding or "utf-8", errors=cfg.decode_errors)

    return resolve


def _hf_stream(cfg: StackV2Config, language: str) -> Iterable[dict]:  # pragma: no cover - needs network
    import os

    from datasets import load_dataset

    token = os.environ.get(cfg.hf_token_env)
    return load_dataset(cfg.dataset, language, split=cfg.split, streaming=cfg.streaming, token=token)


class StackV2Loader:
    """Streams The Stack v2 and yields CodeDocuments.

    `content_resolver` and `stream_factory` default to the real S3 + HF backends
    but can be injected (fixtures) for offline use.
    """

    def __init__(
        self,
        cfg: StackV2Config | None = None,
        content_resolver: ContentResolver | None = None,
        stream_factory: Callable[[StackV2Config, str], Iterable[dict]] | None = None,
    ) -> None:
        self.cfg = cfg or StackV2Config()
        self._resolver = content_resolver
        self._stream_factory = stream_factory or _hf_stream
        self.stats = {"seen": 0, "yielded": 0, "skipped_oversize": 0,
                      "skipped_no_blob": 0, "skipped_fetch_error": 0, "skipped_empty": 0}

    def _resolver_fn(self) -> ContentResolver:
        if self._resolver is None:
            self._resolver = build_s3_content_resolver(self.cfg)
        return self._resolver

    def iter_documents(self) -> Iterator[CodeDocument]:
        for language in self.cfg.languages:
            n = 0
            for row in self._stream_factory(self.cfg, language):
                if self.cfg.limit is not None and n >= self.cfg.limit:
                    break
                self.stats["seen"] += 1

                length = row.get("length_bytes") or 0
                if length and length > self.cfg.max_file_bytes:
                    self.stats["skipped_oversize"] += 1
                    continue
                blob_id = row.get("blob_id")
                if not blob_id:
                    self.stats["skipped_no_blob"] += 1
                    continue

                try:
                    content = self._resolver_fn()(str(blob_id), row.get("src_encoding"))
                except Exception:
                    self.stats["skipped_fetch_error"] += 1
                    if self.cfg.skip_on_fetch_error:
                        continue
                    raise
                if not content:
                    self.stats["skipped_empty"] += 1
                    continue

                yield row_to_document(row, content, language)
                self.stats["yielded"] += 1
                n += 1
