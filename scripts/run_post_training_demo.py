"""
End-to-end demo of the OpenCoder post-pretraining stages, offline (MockTeacher):

    RefineCode corpus ─▶ annealing mixture (Sec. 2.3)
                       └▶ two-stage SFT data (Sec. 4): synthesize ▶ decontaminate ▶ compose ▶ format

Run:
    python scripts/make_sample_data.py                     # produces sample_data/refined.jsonl inputs
    python -m data_pipeline.cli run --input sample_data/raw_code.jsonl --output sample_data/refined.jsonl
    python scripts/run_post_training_demo.py

Everything uses the offline MockTeacher, so no API access is needed; swap in a
real TeacherModel (see synth_common/teacher.py) to generate real data.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annealing import AnnealingConfig, build_annealing_data
from data_pipeline.io_utils import read_jsonl
from data_pipeline.models import CodeDocument
from sft import (
    SFTConfig,
    TestSetReference,
    build_realuser,
    compose_two_stage,
    decontaminate,
    format_dataset,
    synthesize_diverse,
    synthesize_educational,
    synthesize_package,
    wrap_examples,
)
from synth_common import build_teacher

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFINED = os.path.join(HERE, "sample_data", "refined.jsonl")

# Seeds/dialogues that would normally come from your data loaders.
SNIPPET_SEEDS = [
    'def gcd(a, b):\n    """Greatest common divisor via Euclid\'s algorithm."""\n'
    "    while b:\n        a, b = b, a % b\n    return a\n",
    "def flatten(xss):\n    return [x for xs in xss for x in xs]  # flatten a list of lists\n",
]
WEB_SEEDS = ["A tutorial on summing even numbers in a list. Subscribe now! Learn Python."]
DIALOGUES = [
    {"instruction": "Write a Python function using def to reverse a list", "response": "ok"},
    {"instruction": "What's the best pizza topping?", "response": "pineapple"},  # non-code -> dropped
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="OpenCoder post-training demo")
    ap.add_argument("--teacher", choices=["mock", "claude", "vllm"], default="mock",
                    help="mock=offline (default); claude=Anthropic SDK; vllm=OpenAI-compatible server")
    ap.add_argument("--model", default=None, help="model id/name for claude|vllm")
    ap.add_argument("--base-url", default=None, help="vllm server base URL")
    ap.add_argument("--cache-dir", default=None, help="cache teacher responses to this dir")
    args = ap.parse_args()

    if not os.path.exists(REFINED):
        raise SystemExit(f"Missing {REFINED}. Run the data_pipeline first (see module docstring).")

    refine_docs: list[CodeDocument] = list(read_jsonl(REFINED))
    # Ensure the algorithmic keyword sampler has something to find in the demo.
    refine_docs.append(CodeDocument(
        content="def solve():\n    # leetcode dynamic programming solution\n    return 42\n",
        path="algo/leetcode_dp.py", language="Python", category="code"))

    teacher = build_teacher(args.teacher, model=args.model,
                            base_url=args.base_url, cache_dir=args.cache_dir)
    print(f"Teacher backend: {args.teacher}"
          + (f" ({args.model})" if args.model else ""))

    # ---------------------------------------------------------------- annealing
    print("=" * 70, "\nANNEALING (Sec. 2.3)\n", "=" * 70, sep="")
    _, mix_report = build_annealing_data(
        refine_docs, SNIPPET_SEEDS, total_token_budget=3000,
        cfg=AnnealingConfig(), teacher=teacher)
    print(mix_report.summary())

    # ---------------------------------------------------------------------- SFT
    print("\n" + "=" * 70, "\nTWO-STAGE SFT (Sec. 4)\n", "=" * 70, sep="")
    cfg = SFTConfig()

    diverse = list(synthesize_diverse(WEB_SEEDS, cfg.diverse, teacher, seed=cfg.seed))
    educational = list(synthesize_educational(SNIPPET_SEEDS, cfg.educational, teacher))
    package = list(synthesize_package(cfg.package, teacher))
    realuser = list(build_realuser(DIALOGUES, teacher))
    # Open-source pools (normally downloaded) — tiny stand-ins here.
    evol = list(wrap_examples([{"instruction": "Implement bubble sort", "response": "def s(a): ..."}], "evol_instruct"))
    print(f"synthesized: diverse={len(diverse)} educational={len(educational)} "
          f"package={len(package)} realuser={len(realuser)} evol(open-src)={len(evol)}")

    # Decontaminate against a (toy) HumanEval-like reference.
    ref = TestSetReference(
        texts=["def has close elements numbers threshold check if in given list any two are closer than"],
        entry_points=["has_close_elements", "gcd"])
    all_examples = diverse + educational + package + realuser + evol
    clean, dreport = decontaminate(all_examples, ref, cfg.decontam)
    print(dreport.summary())

    # Compose the two stages per Table 5 quotas.
    by_source: dict[str, list] = {}
    for ex in clean:
        by_source.setdefault(ex.source, []).append(ex)
    stages, creport = compose_two_stage(by_source, cfg)
    print(creport.summary())

    # Format a couple of training-ready records.
    print("\nSample formatted Stage-2 records:")
    for rec in list(format_dataset(stages[2]))[:2]:
        print(f"  [{rec['source']}, {rec['tokens']} tok] {rec['text'][:70].replace(chr(10), ' ')}...")

    print(f"\nTraining recipe (Sec. 4.3): stage1={cfg.stage1}  stage2={cfg.stage2}")


if __name__ == "__main__":
    main()
