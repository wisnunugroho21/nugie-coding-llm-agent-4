"""
AdamW with global-norm gradient clipping — the standard LLM optimizer.

Master weights stay fp32 (the model stores params fp32; only matmuls run in the
configured compute dtype), so the optimizer state is fp32 too. The learning rate
is an `optax` schedule (WSD for pretrain/anneal, cosine for SFT).
"""

from __future__ import annotations

import optax


def build_tx(
    schedule: optax.Schedule,
    weight_decay: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
    clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, b1=b1, b2=b2, weight_decay=weight_decay),
    )
