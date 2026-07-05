# Training Loop (OpenCoder Sec. 3.2 / 4.3)

Trains the Kimi-Linear / GDN-2 model on the corpus produced by the data stack, in
the paper's three phases. Everything runs on CPU with the tiny byte-level demo
config; scale the model + steps for a real run.

```
RefineCode corpus ─▶ pretrain (WSD: warmup → stable)
annealing mixture ─▶ anneal   (WSD: decay → 1e-5)     [warm-started from pretrain]
SFT instruction data ─▶ sft   (cosine, response-only)  [warm-started from anneal]
```

## What the train step does ([loop.py](loop.py))

Each `nnx.jit`'d step:
1. **forward → loss** = weighted next-token cross-entropy ([loss.py](loss.py)) **+**
   the MoE load-balancing aux loss the model returns;
2. **backward → AdamW** update of the model's `nnx.Param`s (fp32 master weights,
   global-norm clipping);
3. **aux-loss-free router-bias update** — each MoE layer's `router_bias` (an
   `nnx.Variable`, *not* a trained param) is nudged toward uniform expert load
   using that layer's `group_sizes`, exactly as `multi_latent_attention/moe.py`
   intends (DeepSeek-V3 / Kimi style). This runs **outside** the gradient.

The loss mask makes one objective serve every phase: all-ones for
pretrain/annealing (predict every next token), response-only for SFT (prompt +
padding weighted 0).

## Schedules ([schedule.py](schedule.py))

| Phase | Schedule | Paper (1.5B) |
|-------|----------|--------------|
| pretrain | WSD: linear warmup → constant peak | peak LR 3e-4, warmup 2000 steps / 8B tokens |
| anneal | WSD: exponential decay → end | decay to 1e-5 over ~100B tokens |
| sft stage 1 | warmup + cosine | 1 epoch, batch 4096, LR 2e-5 |
| sft stage 2 | warmup + cosine | 3 epochs, batch 512, LR 5e-5 |

Paper values live in [config.py](config.py) (`PRETRAIN_PRESET`, `STAGE1_PRESET`,
`STAGE2_PRESET`).

## Device presets ([devices.py](devices.py))

Full configs sized to real hardware — model dims + compute dtype + batch +
gradient accumulation + data parallelism. Pass `--device` to any phase command;
it overrides the model/batch/seq-len (the phase still supplies the paper LR/schedule).

| Device | dtype | model | seq | per-dev batch × accum (× GPUs) | params |
|--------|-------|-------|-----|-------------------------------|--------|
| `m1` (M1 CPU, smoke) | fp32 | d=256, L=4, E=8 | 256 | 2 × 1 | 8.9M |
| `t4` (Colab, 16GB) | bf16 | d=512, L=8, E=8 | 1024 | 4 × 8 | 68.4M |
| `t4x2` (Kaggle, 2×16GB) | bf16 | d=512, L=8, E=8 | 1024 | 4 × 8 × **2 GPUs** (data-parallel) | 68.4M |
| `h200` (141GB) | bf16 | d=1536, L=16, E=16, top-4 | 4096 | 8 × 4 | ~1.3B |

```bash
python -m training.devices          # print the table + param counts
```

- **bf16** compute with fp32 master weights + Adam state (`compute_dtype` on the
  model); the numerically sensitive parts (GDN-2 core, RMSNorm, router, loss) stay
  fp32.
- **Gradient accumulation** (`optax.MultiSteps`) reaches a sane global batch on
  16GB cards without OOM.
- **Data parallelism** (`t4x2`) replicates params and shards the batch across all
  local GPUs via `nnx.pmap`, all-reducing gradients + expert load each step. It
  degrades to single-device automatically when only one GPU is visible.
- All presets use the **byte-level tokenizer (vocab 256)** so they run as-is; for a
  real run raise `vocab_size` to your tokenizer (e.g. OpenCoder's 96,640).
  Preset `steps` are short placeholders — scale up with `--steps`.

```bash
# example: full three-phase run on a Colab T4
python -m training.cli pretrain --device t4 --data refined.jsonl --steps 20000 --save ck/pre.pkl
python -m training.cli anneal   --device t4 --data annealing.jsonl --init ck/pre.pkl --save ck/ann.pkl
python -m training.cli sft --stage 1 --device t4 --data sft1.jsonl --init ck/ann.pkl --save ck/sft1.pkl

# Kaggle 2×T4 — same commands with --device t4x2 (data-parallel across both GPUs)
```

## Usage

```bash
# 1. pretrain on the cleaned code corpus (manual sizing, no preset)
python -m training.cli pretrain --data sample_data/refined.jsonl \
    --steps 200 --save ckpt/pretrain.pkl

# 2. anneal from the pretrained weights
python -m training.cli anneal --data sample_data/annealing.jsonl \
    --init ckpt/pretrain.pkl --steps 100 --save ckpt/anneal.pkl

# 3. two-stage SFT (response-only loss)
python -m training.cli sft --stage 1 --data sample_data/sft_stage1.jsonl \
    --init ckpt/anneal.pkl --steps 100 --save ckpt/sft1.pkl
python -m training.cli sft --stage 2 --data sample_data/sft_stage2.jsonl \
    --init ckpt/sft1.pkl --steps 100 --save ckpt/sft2.pkl
```

Or from Python:

```python
from training import TrainConfig, run_phase
run_phase(TrainConfig(phase="pretrain", data_paths=["sample_data/refined.jsonl"],
                      steps=200, save_to="ckpt/pretrain.pkl"))
```

End-to-end demo (data stages → all three phases → a greedy sample):
[scripts/run_training_demo.py](../scripts/run_training_demo.py).

## Data formats
- **pretrain / anneal**: JSONL with a `content` (or `text`) field — the cleaned
  corpus / annealing mixture. Documents are concatenated and packed into `seq_len`
  blocks.
- **sft**: JSONL with `instruction` / `response` (aliases: `prompt`/`output`).
  Formatted into the chat template; only response tokens are trained on.

## Checkpointing ([checkpoint.py](checkpoint.py))

Saves the model's **full** state — including the MoE `router_bias` variables,
which are updated during training and would otherwise reload stale and change
routing. Optimizer state is not saved (each phase uses its own optimizer).

## Constraints
- `seq_len` must be a multiple of the model's `gdn_chunk_size` (the GDN-2
  chunkwise core reshapes L into L/C) and ≤ `max_seq_len` — enforced by
  `TrainConfig`.
- For SFT, `seq_len` must be large enough to hold prompt + response, or the
  response is truncated away and the batch contributes zero loss.

## Scaling up
Use a bigger `--device` preset (or edit [devices.py](devices.py)), plug in
OpenCoder's real tokenizer (vocab 96,640) via any `encode`/`decode`/`vocab_size`
object, point the data paths at the real corpora, and raise `--steps`. The step,
schedules, checkpointing, router-bias, gradient-accumulation, and data-parallel
logic are unchanged. Beyond single-node data parallelism (many GPUs / multi-host),
initialize `jax.distributed` and keep the same `--device t4x2`-style `nnx.pmap`
step, or move to tensor/expert parallelism for models too large to replicate.

## Tests

```bash
python -m unittest tests.test_training
```
