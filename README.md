# nugie — a code LLM built from scratch

A from-scratch code-generation LLM: a **Kimi-Linear** architecture (3:1 hybrid of
linear + full attention, NoPE MLA, MoE FFN) with **Gated DeltaNet-2** substituted
for KDA, plus a complete **OpenCoder / RefineCode** data stack
([arXiv:2411.04905](https://arxiv.org/abs/2411.04905)).

## Layout

### Model (JAX / Flax NNX)
- [`kimi_linear_gdn2.py`](kimi_linear_gdn2.py) — the decoder-only LM.
- [`gated_deltanet_2/`](gated_deltanet_2/) — the GDN-2 linear-attention token mixer.
- [`multi_latent_attention/`](multi_latent_attention/) — NoPE MLA + MoE FFN.

### Data (the OpenCoder cookbook, end to end)
The corpus flows through these stages, each a standalone package:

| Stage | Package | Paper | Output |
|-------|---------|-------|--------|
| 0. Ingestion | [`data_ingestion/`](data_ingestion/README.md) | Sec. 2.1 | streams **The Stack v2** (HF metadata + SWH S3 contents) → `CodeDocument` JSONL |
| 1. Pretraining corpus | [`data_pipeline/`](data_pipeline/README.md) | Sec. 2.1 / App. A | cleaned, deduped, filtered code (RefineCode) |
| 2. Annealing mixture | [`annealing/`](annealing/README.md) | Sec. 2.3 | ~100B-token, ~84%-RefineCode end-of-pretraining blend |
| 3. Two-stage SFT | [`sft/`](sft/README.md) | Sec. 4 | instruction data, synthesized + decontaminated + composed |

### Training ([`training/`](training/README.md))
The loop that consumes the corpus and trains the model, in the paper's three
phases (Sec. 3.2 / 4.3): **pretrain** (WSD warmup+stable) → **anneal** (WSD decay
→ 1e-5) → **two-stage SFT** (cosine, response-only loss). The `nnx.jit` train step
does AdamW on the params plus the aux-loss-free MoE router-bias update; checkpoints
hand weights from one phase to the next. Ready-made **device presets** (`--device
m1|t4|t4x2|h200`) size the model, dtype (bf16), gradient accumulation, and
multi-GPU data parallelism (`nnx.pmap`) to each target — see the table in the
[training README](training/README.md).

### Evaluation ([`evaluation/`](evaluation/README.md))
**HumanEval / MBPP Pass@k** (Sec. 5): sample completions, assemble runnable
programs, execute in the sandbox, and score with the unbiased pass@k estimator.
An oracle generator scores pass@1 = 1.000 (harness self-test); plug in the trained
checkpoint or a Claude teacher to score a real model.

[`synth_common/`](synth_common/) holds the shared synthesis primitives (pluggable
teacher model, real code-execution/test validation, token counting, n-gram
overlap) used by stages 2 and 3.

## Quick start

```bash
pip install -r requirements.txt

# Stage 1 — build the pretraining corpus from raw files
python scripts/make_sample_data.py
python -m data_pipeline.cli run --input sample_data/raw_code.jsonl --output sample_data/refined.jsonl

# Stages 2 & 3 — annealing mixture + two-stage SFT data (offline, MockTeacher)
python scripts/run_post_training_demo.py

# Training — pretrain → anneal → SFT → generate (tiny CPU model, ~20s)
python scripts/run_training_demo.py

# Real BPE tokenizer — train a vocab from the corpus, then train with it
python scripts/train_tokenizer.py --data sample_data/refined.jsonl --vocab-size 4000 --output tok.json
python -m training.cli pretrain --device t4 --data refined.jsonl --tokenizer tok.json

# Evaluation — HumanEval/MBPP Pass@1 harness self-test (oracle -> 1.000)
python -m evaluation.cli --benchmark humaneval --oracle
```

Stages 1–3 run **offline** on synthetic sample data. To go real:

```bash
# Stage 0 — ingest The Stack v2 (needs HF token + AWS creds; see data_ingestion/README.md)
pip install datasets boto3 "smart_open[s3]"
export HF_TOKEN=hf_...   AWS_ACCESS_KEY_ID=...   AWS_SECRET_ACCESS_KEY=...
python -m data_ingestion.cli ingest --languages Python --limit 50000 \
    --output raw_code.jsonl --run-pipeline --refined-output refined.jsonl
```

For real synthesized data in stages 2–3, swap the offline `MockTeacher` for a
real teacher backend ([`synth_common/clients.py`](synth_common/clients.py)):

```bash
pip install anthropic                    # ClaudeTeacher (recommended, Opus 4.8)
export ANTHROPIC_API_KEY=sk-ant-...       # or: ant auth login
python scripts/run_post_training_demo.py --teacher claude --cache-dir .teacher_cache
# or a local vLLM server:
python scripts/run_post_training_demo.py --teacher vllm --model <served-model> --base-url http://localhost:8000/v1
```

## Run on Kaggle (2×T4)

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) runs the whole
project end to end on Kaggle's free **GPU T4 ×2**: install → data pipeline → BPE
tokenizer → pretrain/anneal/SFT **data-parallel across both GPUs** → generate →
Pass@1. Upload this repo as a Kaggle Dataset (or set a clone URL in the notebook),
set the accelerator to *GPU T4 ×2*, and *Run All*. Regenerate it with
`python scripts/make_kaggle_notebook.py`.

## Tests

```bash
python -m unittest discover -s tests -v    # data + training + eval + tokenizer (70 tests)
```
