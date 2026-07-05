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
The corpus flows through three stages, each a standalone package:

| Stage | Package | Paper | Output |
|-------|---------|-------|--------|
| 1. Pretraining corpus | [`data_pipeline/`](data_pipeline/README.md) | Sec. 2.1 / App. A | cleaned, deduped, filtered code (RefineCode) |
| 2. Annealing mixture | [`annealing/`](annealing/README.md) | Sec. 2.3 | ~100B-token, ~84%-RefineCode end-of-pretraining blend |
| 3. Two-stage SFT | [`sft/`](sft/README.md) | Sec. 4 | instruction data, synthesized + decontaminated + composed |

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
```

All data stages run **offline** on synthetic sample data. To go real: point stage
1 at The Stack v2 / GitHub (an ingestion loader that emits the JSONL schema), and
swap the offline `MockTeacher` for a real client (Claude via the Anthropic SDK, or
a local vLLM server) in stages 2–3 — see [`synth_common/teacher.py`](synth_common/teacher.py).

## Tests

```bash
python -m unittest discover -s tests -v    # data_pipeline + annealing + sft
```
