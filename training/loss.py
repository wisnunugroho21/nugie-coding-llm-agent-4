"""
Weighted next-token cross-entropy.

Standard causal LM objective: position t's logits predict token t+1. The per-token
`weights` mask lets one loss serve every phase — all-ones for pretraining/annealing
(predict every next token), response-only for SFT (prompt + padding weighted 0).

Computed in fp32 for a numerically stable softmax under bf16 compute (the model
already upcasts its logits to fp32).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def weighted_next_token_ce(
    logits: jax.Array, input_ids: jax.Array, weights: jax.Array
) -> jax.Array:
    """logits[B,L,V], input_ids[B,L], weights[B,L] -> scalar mean CE over weighted tokens."""
    logits = logits[:, :-1, :].astype(jnp.float32)   # predictions for positions 0..L-2
    targets = input_ids[:, 1:]                        # next token at each position
    w = weights[:, 1:]                                # weight of each *target* token

    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return jnp.sum(nll * w) / (jnp.sum(w) + 1e-8)
