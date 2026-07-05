"""
training — the loop that turns the RefineCode corpus into a trained Kimi-Linear /
GDN-2 model, in the paper's three phases:

    pretrain   (WSD warmup+stable)      on the cleaned code corpus
    anneal     (WSD decay -> 1e-5)      on the annealing mixture
    sft        (two-stage, cosine)      on the instruction data

Each phase is one call to `run_phase(TrainConfig)`; checkpoints hand weights from
one phase to the next. The train step (training/loop.py) does AdamW on the params
plus the aux-loss-free MoE router-bias update the model exposes via `group_sizes`.
"""

from __future__ import annotations

import flax.nnx as nnx
import optax

from kimi_linear_gdn2 import KimiLinear

from .checkpoint import load_model, save_model
from .config import TrainConfig, demo_model_config
from .data import pretrain_batches, sft_batches
from .loop import fit, make_train_step
from .optimizer import build_tx
from .schedule import cosine_schedule, wsd_schedule
from .tokenizer import BPETokenizer, ByteTokenizer

__all__ = [
    "TrainConfig", "run_phase", "demo_model_config",
    "KimiLinear", "ByteTokenizer", "BPETokenizer",
    "fit", "make_train_step", "build_tx",
    "wsd_schedule", "cosine_schedule",
    "save_model", "load_model",
    "pretrain_batches", "sft_batches",
]


def run_phase(cfg: TrainConfig) -> KimiLinear:
    """Build/warm-start the model, run one training phase, optionally checkpoint."""
    if cfg.tokenizer_path:
        tok = BPETokenizer.load(cfg.tokenizer_path)
        if cfg.model.vocab_size != tok.vocab_size:
            print(f"[{cfg.phase}] setting model vocab_size {cfg.model.vocab_size} "
                  f"-> tokenizer vocab_size {tok.vocab_size}")
            cfg.model.vocab_size = tok.vocab_size   # model head must match the tokenizer
    else:
        tok = ByteTokenizer()
    model = KimiLinear(cfg.model, rngs=nnx.Rngs(cfg.seed))
    if cfg.init_from:
        load_model(model, cfg.init_from)
        print(f"[{cfg.phase}] warm-started from {cfg.init_from}")

    if cfg.phase in ("pretrain", "anneal"):
        schedule = wsd_schedule(
            cfg.peak_lr, cfg.end_lr, cfg.warmup_steps, cfg.stable_steps, cfg.decay_steps
        )
        batches = pretrain_batches(
            cfg.data_paths, tok, cfg.seq_len, cfg.batch_size, cfg.data_field
        )
    elif cfg.phase == "sft":
        total = cfg.decay_steps or cfg.steps
        schedule = cosine_schedule(cfg.peak_lr, cfg.warmup_steps, total)
        batches = sft_batches(
            cfg.data_paths, tok, cfg.seq_len, cfg.batch_size, cfg.system
        )
    else:
        raise ValueError(f"unknown phase {cfg.phase!r} (pretrain|anneal|sft)")

    tx = build_tx(schedule, cfg.weight_decay, cfg.b1, cfg.b2, cfg.clip_norm)
    if cfg.grad_accum > 1:                      # accumulate grads over microbatches
        tx = optax.MultiSteps(tx, cfg.grad_accum).gradient_transformation()
    fit(model, tx, batches, cfg.steps, bias_lr=cfg.bias_lr, schedule=schedule,
        log_every=cfg.log_every, label=cfg.phase,
        data_parallel=cfg.data_parallel, grad_accum=cfg.grad_accum)

    if cfg.save_to:
        save_model(model, cfg.save_to)
        print(f"[{cfg.phase}] saved checkpoint -> {cfg.save_to}")
    return model
