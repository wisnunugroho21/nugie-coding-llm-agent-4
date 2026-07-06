"""
Generate notebooks/kaggle_train.ipynb — a runnable Kaggle notebook that trains the
from-scratch code LLM end to end on 2×T4 (data pipeline → BPE tokenizer →
pretrain/anneal/SFT data-parallel → generate → HumanEval/MBPP Pass@1).

    python scripts/make_kaggle_notebook.py

Kept as the source of the notebook so it can be regenerated / edited in one place.
"""

from __future__ import annotations

import json
import os

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (MD, """# Train a from-scratch Code LLM on Kaggle (2×T4)

This notebook runs the whole project end to end:

**RefineCode data pipeline → BPE tokenizer → pretrain → anneal → two-stage SFT
(data-parallel across both T4s) → generate → HumanEval/MBPP Pass@1.**

The model is *Kimi Linear* (hybrid linear/full attention, MoE FFN) with
*Gated DeltaNet-2*; the data/training recipe follows *OpenCoder* (arXiv:2411.04905).

> **Before running:** open **Settings → Accelerator → GPU T4 ×2**, and (if you need
> to fetch the code by cloning) **Settings → Internet → On**.

Everything below uses the tiny demo config + synthetic sample data so it finishes
in a few minutes. The final cell shows how to scale to a real run."""),

    (MD, "## 1. Environment & install\n\nInstall JAX (CUDA 12) + flax/optax/tokenizers. If JAX still reports CPU after this, **restart the kernel** (Run → Restart) and run the notebook again — a reinstalled JAX needs a fresh process."),
    (CODE, '''import subprocess, sys

def sh(*args, check=False):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=check)

# Lightweight pure-python deps (safe to (re)install).
sh("flax", "optax", "tokenizers")

# JAX with CUDA 12 for the T4s — only if a GPU-backed JAX isn't already importable.
def gpu_jax():
    try:
        import jax
        return jax.default_backend() == "gpu"
    except Exception:
        return False

if not gpu_jax():
    sh("-U", "jax[cuda12]")
    print("Installed jax[cuda12]. If devices below still show CPU, RESTART the kernel and re-run.")
else:
    print("GPU-backed JAX already present.")'''),

    (MD, """## 2. Get the project code

Two ways — pick one:
- **Clone from GitHub**: set `REPO_URL` below (needs Internet on).
- **Attach as a Kaggle Dataset**: upload this repo as a dataset and add it to the
  notebook (Add Input); the cell auto-detects it under `/kaggle/input/`.

The cell copies the code into the writable working dir and `cd`s there."""),
    (CODE, '''import os, glob, shutil, sys

REPO_URL = ""   # e.g. "https://github.com/<you>/nugie-coding-llm-agent.git"; leave "" to auto-detect a dataset

MARKER = "kimi_linear_gdn2.py"

def locate():
    if REPO_URL:
        dst = "/kaggle/working/project"
        if not os.path.exists(dst):
            os.system(f"git clone --depth 1 {REPO_URL} {dst}")
        return dst
    # attached Kaggle dataset (search one and two levels deep under /kaggle/input)
    for pat in ("/kaggle/input/*", "/kaggle/input/*/*"):
        for base in glob.glob(pat):
            if os.path.exists(os.path.join(base, MARKER)):
                return base
    if os.path.exists(MARKER):
        return os.getcwd()
    raise SystemExit("Project not found — set REPO_URL or attach the repo as a Dataset (Add Input).")

src = locate()
PROJECT = "/kaggle/working/project"
if os.path.abspath(src) != os.path.abspath(PROJECT):
    if os.path.exists(PROJECT):
        shutil.rmtree(PROJECT)
    shutil.copytree(src, PROJECT)
os.chdir(PROJECT)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
print("Project dir:", PROJECT)
print("Contents:", sorted(os.listdir(PROJECT))[:12])'''),

    (MD, "## 3. Verify the GPUs\n\nYou want **2 devices** for data parallelism. If you only see 1 (or CPU), check the accelerator setting / restart after the JAX install — the `t4x2` preset still runs on 1 device, just without the 2× speedup."),
    (CODE, '''import jax
print("JAX backend:", jax.default_backend())
print("devices:", jax.devices())
N_DEV = jax.local_device_count()
print("local device count:", N_DEV)'''),

    (MD, "## 4. Build the training data (RefineCode pipeline)\n\nGenerate the synthetic sample corpus and run the full cleaning pipeline (dedup → PII/copyright → heuristic filters → downsample) to produce `refined.jsonl`."),
    (CODE, '''import subprocess, sys

def run(*args):
    subprocess.run([sys.executable, *args], check=True)

run("scripts/make_sample_data.py")
run("-m", "data_pipeline.cli", "run",
    "--input", "sample_data/raw_code.jsonl",
    "--output", "sample_data/refined.jsonl")'''),

    (MD, "Now build the **annealing mixture** and **SFT instruction data** from the later stages (offline `MockTeacher`, so no API needed)."),
    (CODE, '''import json
from data_pipeline.io_utils import read_jsonl, write_jsonl
from annealing import build_annealing_data, AnnealingConfig
from sft import synthesize_educational, synthesize_package, SFTConfig
from sft.config import PackageSynthConfig

SEEDS = [
    "def gcd(a, b):\\n    while b:\\n        a, b = b, a % b\\n    return a\\n",
    "def flatten(xss):\\n    return [x for xs in xss for x in xs]\\n",
]
docs = list(read_jsonl("sample_data/refined.jsonl"))
mix, _ = build_annealing_data(docs, SEEDS, total_token_budget=6000, cfg=AnnealingConfig())
write_jsonl("sample_data/annealing.jsonl", mix)

cfg = SFTConfig()
examples = list(synthesize_educational(SEEDS, cfg.educational))
examples += list(synthesize_package(PackageSynthConfig(libraries=("math", "json", "itertools"),
                                                       max_apis_per_library=8)))
with open("sample_data/sft_demo.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps({"instruction": ex.instruction, "response": ex.response}) + "\\n")
print("annealing docs:", len(mix), "| SFT examples:", len(examples))'''),

    (MD, "## 5. Train a BPE tokenizer\n\nA byte-level BPE trained on our corpus (from scratch). The sample corpus is tiny so the vocab caps out small; use `--vocab-size 32000` (or `96640`) on real data."),
    (CODE, '''run("scripts/train_tokenizer.py",
    "--data", "sample_data/refined.jsonl", "sample_data/sft_demo.jsonl",
    "--vocab-size", "4000",
    "--output", "sample_data/tokenizer.json")'''),

    (MD, """## 6. Three-phase training on 2×T4 (data-parallel)

The `t4x2` device preset: bf16 compute, ~68M-param model, seq 1024, gradient
accumulation, and **data parallelism across both T4s** (`nnx.pmap`). Steps here are
small so the demo finishes quickly — raise `--steps` for a real run."""),
    (CODE, '''from training.devices import get_preset
p = get_preset("t4x2")
m = p.model
print(f"model: d_model={m.d_model} layers={m.n_layers} experts={m.moe_n_routed} dtype={m.compute_dtype}")
print(f"seq_len={p.seq_len} per_device_batch={p.per_device_batch} grad_accum={p.grad_accum} "
      f"data_parallel={p.data_parallel}  (global batch/step = {p.per_device_batch}×{N_DEV} devices)")'''),

    (MD, "### 6a. Pretrain (WSD warmup+stable)"),
    (CODE, '''run("-m", "training.cli", "pretrain", "--device", "t4x2",
    "--data", "sample_data/refined.jsonl",
    "--tokenizer", "sample_data/tokenizer.json",
    "--steps", "100", "--log-every", "20", "--save", "ckpt_pretrain.pkl")'''),

    (MD, "### 6b. Anneal (WSD decay → 1e-5), warm-started from pretrain"),
    (CODE, '''run("-m", "training.cli", "anneal", "--device", "t4x2",
    "--data", "sample_data/annealing.jsonl",
    "--tokenizer", "sample_data/tokenizer.json",
    "--init", "ckpt_pretrain.pkl",
    "--steps", "40", "--log-every", "10", "--save", "ckpt_anneal.pkl")'''),

    (MD, "### 6c. SFT stage 1 (cosine, response-only loss), warm-started from anneal"),
    (CODE, '''run("-m", "training.cli", "sft", "--stage", "1", "--device", "t4x2",
    "--data", "sample_data/sft_demo.jsonl",
    "--tokenizer", "sample_data/tokenizer.json",
    "--init", "ckpt_anneal.pkl",
    "--steps", "40", "--log-every", "10", "--save", "ckpt_sft.pkl")'''),

    (MD, "## 7. Generate a sample\n\n(The demo model is tiny and barely trained, so expect gibberish — the point is that the full pipeline runs.)"),
    (CODE, '''import flax.nnx as nnx
from training import KimiLinear, load_model
from training.tokenizer import BPETokenizer
from training.devices import get_preset
from evaluation.generator import ModelGenerator

tok = BPETokenizer.load("sample_data/tokenizer.json")
mcfg = get_preset("t4x2").model
mcfg.vocab_size = tok.vocab_size            # head must match the tokenizer
model = KimiLinear(mcfg, rngs=nnx.Rngs(0))
load_model(model, "ckpt_sft.pkl")

gen = ModelGenerator(model, tok, seed=0)
prompt = ("<|system|>\\nYou are a helpful programming assistant.\\n"
          "<|user|>\\nWrite a Python function that adds two numbers.\\n<|assistant|>\\n")
print(repr(gen.generate(prompt, temperature=0.8, max_new_tokens=64)))'''),

    (MD, "## 8. Evaluate — HumanEval / MBPP Pass@1\n\nThe oracle self-test must score **1.000** (proves the harness is correct); the trained tiny model will score ~0."),
    (CODE, '''from evaluation import evaluate, EvalConfig, sample_humaneval, sample_mbpp
from evaluation.generator import OracleGenerator

for name, probs in [("HumanEval", sample_humaneval()), ("MBPP", sample_mbpp())]:
    oracle = evaluate(probs, OracleGenerator(probs), EvalConfig(ks=(1,)))
    print(f"{name} oracle self-test -> pass@1 = {oracle.pass_at_k[1]:.3f}")

model_eval = evaluate(sample_humaneval(), gen, EvalConfig(ks=(1,), max_new_tokens=128))
print("Trained model on HumanEval:", model_eval.summary())'''),

    (MD, """## 9. Scaling to a real run

- **Real data**: replace the synthetic corpus with ingested code. Set up
  `data_ingestion` (The Stack v2 — needs a Hugging Face token + AWS creds), then
  `python -m data_ingestion.cli ingest --languages Python --run-pipeline`.
- **Real synthesis**: swap the offline `MockTeacher` for `ClaudeTeacher`
  (`--teacher claude`, needs `anthropic` + an API key) or a local vLLM server.
- **Bigger tokenizer**: `--vocab-size 32000` (or `96640`) on the full corpus.
- **More training**: raise `--steps` (paper is millions of steps); the `t4x2`
  preset already uses both GPUs. For larger models use a bigger box + the `h200`
  preset (or edit `training/devices.py`).
- **Save your work**: checkpoints and `tokenizer.json` are in `/kaggle/working` —
  commit the notebook (Save Version) to persist them as output.

See the per-package `README.md` files for the details of each stage."""),
]


def _cell(kind: str, text: str, idx: int) -> dict:
    # nbformat stores source as a list of lines, each keeping its trailing newline.
    lines = text.splitlines(keepends=True)
    cid = f"cell{idx:02d}"
    if kind == CODE:
        return {"cell_type": "code", "id": cid, "metadata": {},
                "execution_count": None, "outputs": [], "source": lines}
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines}


def main() -> None:
    nb = {
        "cells": [_cell(k, t, i) for i, (k, t) in enumerate(CELLS)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "kaggle_train.ipynb")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
