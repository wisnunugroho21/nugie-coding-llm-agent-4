"""
CLI for The Stack v2 ingestion.

    # stream metadata from HF + fetch contents from S3, write the raw JSONL corpus
    python -m data_ingestion.cli ingest \
        --languages Python,Java --limit 50000 --output raw_code.jsonl

    # ingest AND run the full RefineCode pipeline in one shot
    python -m data_ingestion.cli ingest --languages Python --limit 50000 \
        --output raw_code.jsonl --run-pipeline --refined-output refined.jsonl

Requires (at run time): HF token (env HF_TOKEN) with dataset terms accepted, and
AWS credentials for the Software Heritage S3 bucket. See README.md.
"""

from __future__ import annotations

import argparse
import sys

from data_pipeline.io_utils import write_jsonl

from .config import StackV2Config
from .the_stack_v2 import StackV2Loader


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg = StackV2Config(
        dataset=args.dataset,
        languages=tuple(l.strip() for l in args.languages.split(",") if l.strip()),
        limit=args.limit,
        max_file_bytes=args.max_file_bytes,
    )
    loader = StackV2Loader(cfg)

    if args.run_pipeline:
        # Ingest straight into the RefineCode pipeline.
        from data_pipeline import PipelineConfig, run_pipeline

        pcfg = PipelineConfig() if args.full else PipelineConfig.fast_demo()
        kept, stats = run_pipeline(loader.iter_documents(), pcfg)
        n = write_jsonl(args.refined_output, kept)
        print(stats.summary(), file=sys.stderr)
        print(f"\nIngestion: {loader.stats}", file=sys.stderr)
        print(f"Wrote {n} refined documents -> {args.refined_output}", file=sys.stderr)
    else:
        n = write_jsonl(args.output, loader.iter_documents())
        print(f"Ingestion: {loader.stats}", file=sys.stderr)
        print(f"Wrote {n} raw documents -> {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="data_ingestion", description="The Stack v2 ingestion")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="stream The Stack v2 -> CodeDocument JSONL")
    ing.add_argument("--dataset", default="bigcode/the-stack-v2-dedup")
    ing.add_argument("--languages", default="Python", help="comma-separated HF config names")
    ing.add_argument("--limit", type=int, default=None, help="max docs per language")
    ing.add_argument("--max-file-bytes", type=int, default=8 * 1024 * 1024)
    ing.add_argument("--output", default="raw_code.jsonl")
    ing.add_argument("--run-pipeline", action="store_true", help="also run the RefineCode pipeline")
    ing.add_argument("--refined-output", default="refined.jsonl")
    ing.add_argument("--full", action="store_true", help="use paper 2048-perm MinHash (with --run-pipeline)")
    ing.set_defaults(func=_cmd_ingest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
