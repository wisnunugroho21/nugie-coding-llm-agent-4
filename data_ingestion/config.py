"""
Configuration for The Stack v2 ingestion (OpenCoder's raw code source, Sec. 2.1).

The Stack v2 (`bigcode/the-stack-v2-dedup`) ships **metadata only**; each row's
actual file bytes live in the Software Heritage S3 bucket and are fetched by
`blob_id`. So ingestion has two moving parts — an HF streaming iterator over
metadata rows, and an S3 content resolver — both configured here.

Access requirements (you provide these at run time; they are not needed to import
or unit-test this package):
  * accept the dataset terms on the Hugging Face hub and set an HF token,
  * accept the Software Heritage terms and set AWS credentials (the S3 bucket is
    requester-anonymous but boto3 still needs a session/region).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class StackV2Config:
    dataset: str = "bigcode/the-stack-v2-dedup"   # or "bigcode/the-stack-v2"
    # The Stack v2 is sharded by language: each is a separate HF config name.
    languages: tuple[str, ...] = ("Python",)
    split: str = "train"
    streaming: bool = True                 # never materialize the whole shard
    limit: int | None = None               # max docs PER language (None = all)

    # Pre-fetch size gate (paper drops >8MB files). Uses the row's length_bytes
    # so we skip the S3 fetch entirely for oversized blobs.
    max_file_bytes: int = 8 * 1024 * 1024

    # --- Software Heritage S3 content location ---
    s3_bucket: str = "softwareheritage"
    s3_key_prefix: str = "content"
    s3_compression: str = ".gz"            # SWH blobs are gzip-compressed
    decode_errors: str = "replace"         # decode with src_encoding, be lenient

    # Env var names for credentials (read at run time).
    hf_token_env: str = "HF_TOKEN"

    # Skip a blob (rather than abort) if its S3 fetch/decoding fails.
    skip_on_fetch_error: bool = True
