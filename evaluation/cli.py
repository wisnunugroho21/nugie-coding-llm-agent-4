"""
Evaluation CLI — Pass@k on HumanEval / MBPP.

    # harness self-test (oracle should score pass@1 == 1.000)
    python -m evaluation.cli --benchmark humaneval --oracle

    # evaluate a trained checkpoint (greedy pass@1)
    python -m evaluation.cli --benchmark humaneval --data HumanEval.jsonl \
        --model ckpt/sft.pkl

    # pass@10 from a real LLM teacher (needs anthropic + creds)
    python -m evaluation.cli --benchmark mbpp --data mbpp.jsonl \
        --claude --n 10 --k 1 10 --temperature 0.8

Without --data, a small bundled sample is used so the harness runs offline.
"""

from __future__ import annotations

import argparse
import sys

from .config import EvalConfig
from .data import load_humaneval, load_mbpp, sample_humaneval, sample_mbpp
from .generator import FunctionGenerator, OracleGenerator
from .harness import evaluate


def _load_problems(args: argparse.Namespace):
    if args.benchmark == "humaneval":
        return load_humaneval(args.data) if args.data else sample_humaneval()
    return load_mbpp(args.data) if args.data else sample_mbpp()


def _build_generator(args: argparse.Namespace, problems):
    if args.oracle:
        return OracleGenerator(problems)
    if args.model:
        import flax.nnx as nnx

        from training import KimiLinear, demo_model_config, load_model
        from training.tokenizer import BPETokenizer, ByteTokenizer
        from .generator import ModelGenerator

        tok = BPETokenizer.load(args.tokenizer) if args.tokenizer else ByteTokenizer()
        model_cfg = demo_model_config()
        if args.tokenizer:
            model_cfg.vocab_size = tok.vocab_size   # must match the checkpoint's head
        model = KimiLinear(model_cfg, rngs=nnx.Rngs(0))
        load_model(model, args.model)
        return ModelGenerator(model, tok, seed=args.seed)
    if args.claude:
        from synth_common import ClaudeTeacher

        teacher = ClaudeTeacher(model=args.claude_model)
        return FunctionGenerator(lambda p, t, n: teacher.generate(p, temperature=t, max_tokens=n))
    raise SystemExit("Pick a generator: --oracle | --model <ckpt> | --claude")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evaluation", description="HumanEval/MBPP Pass@k")
    ap.add_argument("--benchmark", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--data", default=None, help="benchmark JSONL (else bundled sample)")
    # generators
    ap.add_argument("--oracle", action="store_true", help="reference solutions (self-test)")
    ap.add_argument("--model", default=None, help="KimiLinear checkpoint to evaluate")
    ap.add_argument("--tokenizer", default=None, help="BPE tokenizer.json (match the checkpoint)")
    ap.add_argument("--claude", action="store_true", help="evaluate a Claude teacher")
    ap.add_argument("--claude-model", default="claude-opus-4-8")
    # eval params
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--k", type=int, nargs="+", default=[1], help="pass@k values")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    problems = _load_problems(args)
    generator = _build_generator(args, problems)
    cfg = EvalConfig(
        n_samples=args.n, ks=tuple(args.k), temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, timeout_s=args.timeout,
        limit=args.limit, n_workers=args.workers,
    )
    result = evaluate(problems, generator, cfg)
    print(f"[{args.benchmark}]")
    print(result.summary(), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
