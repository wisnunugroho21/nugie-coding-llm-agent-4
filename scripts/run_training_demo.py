"""
End-to-end training demo: data stages -> three training phases -> generation.

Ties the whole project together on the tiny byte-level model (CPU, seconds):

    RefineCode corpus ─▶ pretrain (WSD warmup+stable)
    annealing mixture ─▶ anneal   (WSD decay -> end_lr)   [warm-started]
    SFT instruction data ─▶ sft   (cosine)                [warm-started]
                          ─▶ greedy sample from the trained model

Prereq:  python scripts/make_sample_data.py &&
         python -m data_pipeline.cli run --input sample_data/raw_code.jsonl \
             --output sample_data/refined.jsonl
Run:     python scripts/run_training_demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from annealing import AnnealingConfig, build_annealing_data
from data_pipeline.io_utils import read_jsonl, write_jsonl
from sft import SFTConfig, synthesize_educational, synthesize_package
from sft.config import PackageSynthConfig
from training import KimiLinear, TrainConfig, demo_model_config, load_model, run_phase
from training.tokenizer import ByteTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "sample_data")
SCRATCH = os.environ.get("SYNTH_SCRATCH", os.path.join(HERE, ".train_demo"))
os.makedirs(SCRATCH, exist_ok=True)

SNIPPET_SEEDS = [
    'def gcd(a, b):\n    """Greatest common divisor."""\n    while b:\n        a, b = b, a % b\n    return a\n',
    "def flatten(xss):\n    return [x for xs in xss for x in xs]  # flatten list of lists\n",
]


def _build_phase_data() -> tuple[str, str]:
    """Produce annealing + SFT JSONL from the earlier data stages (offline)."""
    refined = os.path.join(DATA, "refined.jsonl")
    if not os.path.exists(refined):
        raise SystemExit(f"Missing {refined}. Build it first (see this file's docstring).")
    docs = list(read_jsonl(refined))

    # Annealing mixture -> content JSONL (reuses the annealing stage).
    mix, _ = build_annealing_data(docs, SNIPPET_SEEDS, total_token_budget=4000, cfg=AnnealingConfig())
    anneal_path = os.path.join(DATA, "annealing.jsonl")
    write_jsonl(anneal_path, mix)

    # SFT instruction data -> instruction/response JSONL (reuses the SFT stage).
    cfg = SFTConfig()
    examples = list(synthesize_educational(SNIPPET_SEEDS, cfg.educational))
    examples += list(synthesize_package(PackageSynthConfig(libraries=("math", "json"), max_apis_per_library=6)))
    sft_path = os.path.join(DATA, "sft_demo.jsonl")
    with open(sft_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({"instruction": ex.instruction, "response": ex.response}) + "\n")
    print(f"Built {len(mix)} annealing docs -> {anneal_path}")
    print(f"Built {len(examples)} SFT examples -> {sft_path}\n")
    return anneal_path, sft_path


def main() -> None:
    refined = os.path.join(DATA, "refined.jsonl")
    anneal_path, sft_path = _build_phase_data()
    pre_ck = os.path.join(SCRATCH, "pretrain.pkl")
    ann_ck = os.path.join(SCRATCH, "anneal.pkl")
    sft_ck = os.path.join(SCRATCH, "sft.pkl")

    print("=" * 66, "\nPHASE 1 — PRETRAIN (WSD warmup+stable)\n", "=" * 66, sep="")
    run_phase(TrainConfig(phase="pretrain", data_paths=[refined], steps=40, batch_size=4,
                          seq_len=64, warmup_steps=5, stable_steps=35, save_to=pre_ck, log_every=10))

    print("\n" + "=" * 66, "\nPHASE 2 — ANNEAL (WSD decay -> 1e-5)\n", "=" * 66, sep="")
    run_phase(TrainConfig(phase="anneal", data_paths=[anneal_path], steps=20, batch_size=4,
                          seq_len=64, warmup_steps=0, stable_steps=0, decay_steps=20,
                          init_from=pre_ck, save_to=ann_ck, log_every=10))

    print("\n" + "=" * 66, "\nPHASE 3 — SFT (cosine, response-only loss)\n", "=" * 66, sep="")
    # seq_len must fit prompt+response, or truncation masks the whole response
    # and the loss is 0 (nothing to train on). 256 = the demo model's max_seq_len.
    run_phase(TrainConfig(phase="sft", data_paths=[sft_path], steps=24, batch_size=2,
                          seq_len=256, peak_lr=2e-5, warmup_steps=2, decay_steps=24,
                          init_from=ann_ck, save_to=sft_ck, log_every=8))

    print("\n" + "=" * 66, "\nGENERATION (greedy, from the SFT model)\n", "=" * 66, sep="")
    model = KimiLinear(demo_model_config(), rngs=nnx.Rngs(0))
    load_model(model, sft_ck)
    tok = ByteTokenizer()
    prompt = ("<|system|>\nYou are a helpful programming assistant.\n"
              "<|user|>\nWrite a Python function.\n<|assistant|>\n")
    # Pad prompt to a multiple of the GDN-2 chunk size for the prefill step.
    ids = tok.encode(prompt)
    chunk = demo_model_config().gdn_chunk_size
    if len(ids) % chunk:
        ids = ids + tok.encode(" ") * (chunk - len(ids) % chunk)
    out = model.generate(jnp.asarray([ids], jnp.int32), max_new_tokens=48)
    print("PROMPT:", prompt.replace("\n", "\\n"))
    print("SAMPLE:", repr(tok.decode(list(np.asarray(out[0])))))
    print("\n(Output is gibberish — the byte-level demo model is tiny and barely "
          "trained; the point is the full pipeline runs end to end.)")


if __name__ == "__main__":
    main()
