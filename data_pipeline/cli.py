"""
Command-line entry point for the RefineCode pipeline.

    python -m data_pipeline.cli run   --input raw.jsonl --output refined.jsonl [--full]
    python -m data_pipeline.cli recall --input web.jsonl --output web_code.jsonl --seed seed.jsonl

`run`    executes the full code pipeline (preprocess -> dedup -> transform ->
         filter -> downsample) and prints per-stage / per-rule stats.
`recall` trains the fastText code classifier on a labelled seed file and applies
         web code-data recall (paper Sec. 2.2).

Seed file for `recall`: JSONL where each record additionally has an integer
"label" field (1 == code-related, 0 == not).
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import PipelineConfig, WebRecallConfig
from .io_utils import read_jsonl, write_jsonl
from .models import CodeDocument
from .pipeline import run_pipeline
from .web.fasttext_recall import FastTextRecaller


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig() if args.full else PipelineConfig.fast_demo()
    kept, stats = run_pipeline(read_jsonl(args.input), cfg)
    n = write_jsonl(args.output, kept)
    print(stats.summary(), file=sys.stderr)
    print(f"\nWrote {n} documents -> {args.output}", file=sys.stderr)
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    seed = []
    with open(args.seed, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                seed.append((rec["content"], int(rec.get("label", 0))))
    cfg = WebRecallConfig()
    recaller = FastTextRecaller(cfg).train(seed)
    kept = recaller.recall(read_jsonl(args.input))
    n = write_jsonl(args.output, kept)
    print(f"Recalled {n} code-related web documents -> {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data_pipeline", description="RefineCode data pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full code data pipeline")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--output", required=True)
    p_run.add_argument("--full", action="store_true",
                       help="use the paper's 2048-perm MinHash instead of the fast 128-perm demo")
    p_run.set_defaults(func=_cmd_run)

    p_rec = sub.add_parser("recall", help="train fastText & recall code-related web data")
    p_rec.add_argument("--input", required=True)
    p_rec.add_argument("--output", required=True)
    p_rec.add_argument("--seed", required=True, help="labelled JSONL seed corpus (adds int 'label')")
    p_rec.set_defaults(func=_cmd_recall)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
