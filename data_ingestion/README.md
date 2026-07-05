# Dataset Ingestion — The Stack v2 (OpenCoder Sec. 2.1)

Turns OpenCoder's real raw-code source, **The Stack v2**
(`bigcode/the-stack-v2-dedup`), into the JSONL `CodeDocument` stream that
[`data_pipeline`](../data_pipeline/README.md) consumes.

The Stack v2 ships **metadata only** — the actual file bytes live in the Software
Heritage S3 bucket, fetched per row by `blob_id`. So ingestion has two parts:

```
HF streaming iterator (metadata rows)  ──┐
                                          ├─▶ row_to_document ──▶ CodeDocument JSONL ──▶ data_pipeline
Software Heritage S3 resolver (contents) ─┘
```

Both are injectable ([the_stack_v2.py](the_stack_v2.py)), so the mapping logic is
fully unit-tested offline (see [tests/test_ingestion.py](../tests/test_ingestion.py)).

## Prerequisites (run time only)

1. **Hugging Face**: accept the dataset terms on the hub, then set a token:
   ```bash
   export HF_TOKEN=hf_...
   ```
2. **Software Heritage S3**: accept the SWH terms, install the S3 extras, and
   provide AWS credentials (the bucket is public-read but boto3 needs a session):
   ```bash
   pip install boto3 "smart_open[s3]"
   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=us-east-1
   ```
   Without these, the package still imports and tests; `build_s3_content_resolver`
   raises a clear, actionable error explaining what to install.

## Usage

```bash
# stream metadata + fetch contents, write the raw corpus
python -m data_ingestion.cli ingest --languages Python,Java --limit 50000 --output raw_code.jsonl

# ingest AND run the full RefineCode pipeline in one shot
python -m data_ingestion.cli ingest --languages Python --limit 50000 \
    --output raw_code.jsonl --run-pipeline --refined-output refined.jsonl --full
```

```python
from data_ingestion import StackV2Loader, StackV2Config
from data_pipeline import run_pipeline

loader = StackV2Loader(StackV2Config(languages=("Python",), limit=50_000))
kept, stats = run_pipeline(loader.iter_documents())   # straight into RefineCode
print(loader.stats)     # seen / yielded / skipped_{oversize,no_blob,fetch_error,empty}
```

## Field mapping (Stack v2 row → CodeDocument)

| CodeDocument | Source | Notes |
|--------------|--------|-------|
| `content` | S3 blob by `blob_id`, decoded with `src_encoding` | gzip-compressed in SWH |
| `path` | `path` | |
| `language` | HF config name | Stack v2 is sharded per language |
| `repo_name` | `repo_name` | |
| `stars` | `star_events_count` | dedup tie-break |
| `commit_time` | first of `gha_updated_at`/`gha_created_at`/`revision_date`/… | dedup tie-break |
| `doc_id` | `blob_id` | |

Files whose `length_bytes` exceed 8 MB are skipped **before** the S3 fetch
(paper's size gate), so oversized blobs cost no bandwidth.

## Notes
- **Injectable backends**: pass `content_resolver=` / `stream_factory=` to
  `StackV2Loader` to run against fixtures, a local cache, or a different store.
- **Throughput**: contents are fetched one blob at a time for clarity. For large
  runs, batch the S3 reads (HF `dataset.map(download, batched=True)`) or parallelize
  the resolver — the mapping/gating logic is unchanged.
- **Other sources**: to add The Stack v1.2/dedup (inline content) or FineWeb (web
  track), add a sibling loader that yields `CodeDocument`s; the pipeline downstream
  is identical.
