"""
data_ingestion — turn a real upstream dataset into the JSONL `CodeDocument`
stream that `data_pipeline` consumes.

Primary source: **The Stack v2** (`bigcode/the-stack-v2-dedup`), OpenCoder's raw
code source (Sec. 2.1). Because Stack v2 ships metadata only, ingestion streams
rows from Hugging Face and resolves each file's bytes from Software Heritage S3.

    from data_ingestion import StackV2Loader, StackV2Config
    from data_pipeline.io_utils import write_jsonl
    from data_pipeline import run_pipeline

    loader = StackV2Loader(StackV2Config(languages=("Python",), limit=50_000))
    write_jsonl("raw_code.jsonl", loader.iter_documents())   # -> feeds data_pipeline
    # or straight through:
    kept, stats = run_pipeline(loader.iter_documents())
"""

from .config import StackV2Config
from .the_stack_v2 import (
    StackV2Loader,
    build_s3_content_resolver,
    row_to_document,
)

__all__ = ["StackV2Config", "StackV2Loader", "row_to_document", "build_s3_content_resolver"]
