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

## Usage

```bash
# 1. pretrain on the cleaned code corpus
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
Swap `demo_model_config()` for a full `KimiLinearConfig`, plug in OpenCoder's real
tokenizer (vocab 96,640) via any `encode`/`decode`/`vocab_size` object, point the
data paths at the real corpora, and raise `--steps` / `--batch-size`. The step,
schedules, checkpointing, and router-bias logic are unchanged. For multi-device
data parallelism, shard the batch and wrap the step with `jax` mesh/sharding.

## Tests

```bash
python -m unittest tests.test_training
```
