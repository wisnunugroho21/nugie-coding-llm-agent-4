"""
The core training step and driver.

`train_step` (jitted with `nnx.jit`) does one optimization step:
  1. forward -> weighted next-token CE + the MoE load-balancing aux loss,
  2. backward + AdamW update of the model's `nnx.Param`s,
  3. the **aux-loss-free router-bias update** (DeepSeek-V3 / Kimi style): each MoE
     layer's `router_bias` (an `nnx.Variable`, *not* a trained Param) is nudged
     toward uniform expert load using that layer's `group_sizes` — done outside the
     gradient via `update_router_bias`. The model returns `group_sizes[n_layers,E]`
     for exactly this purpose (see multi_latent_attention/moe.py).

`fit` builds a fresh `nnx.Optimizer`, runs `steps` steps over a batch iterator, and
logs CE / aux / LR. Loss goes down; expert load balances; weights persist on the
`model` object (nnx mutates it in place), ready to checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator

import flax.nnx as nnx
import jax.numpy as jnp
import optax

from multi_latent_attention.moe import update_router_bias

from .loss import weighted_next_token_ce


def make_train_step(bias_lr: float):
    """Build a jitted train step; `bias_lr` is the router-bias nudge size."""

    @nnx.jit
    def train_step(model, optimizer, input_ids, weights):
        def loss_fn(model):
            logits, aux = model(input_ids)
            ce = weighted_next_token_ce(logits, input_ids, weights)
            total = ce + aux["aux_loss"]         # CE + MoE load-balancing aux
            return total, (ce, aux)

        (_, (ce, aux)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)            # AdamW step (Params only)

        # Aux-loss-free load balancing: nudge each layer's selection bias.
        group_sizes = aux["group_sizes"]          # [n_layers, E]
        for i, layer in enumerate(model.layers):
            rb = layer.channel_mixer.router_bias  # nnx.Variable, not a Param
            rb[...] = update_router_bias(rb[...], group_sizes[i], bias_lr)

        return ce, aux["aux_loss"]

    return train_step


def fit(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    batches: Iterator,
    steps: int,
    *,
    bias_lr: float = 1e-3,
    schedule: optax.Schedule | None = None,
    log_every: int = 10,
    label: str = "train",
) -> nnx.Optimizer:
    """Run `steps` steps; returns the optimizer (for step count / further training)."""
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    step_fn = make_train_step(bias_lr)

    for step in range(1, steps + 1):
        ids_np, w_np = next(batches)
        ids = jnp.asarray(ids_np)
        w = jnp.asarray(w_np, dtype=jnp.float32)
        ce, aux_loss = step_fn(model, optimizer, ids, w)

        if step == 1 or step % log_every == 0 or step == steps:
            lr = float(schedule(step - 1)) if schedule is not None else float("nan")
            print(f"[{label}] step {step:>5}/{steps}  ce {float(ce):.4f}  "
                  f"aux {float(aux_loss):.2e}  lr {lr:.2e}")
    return optimizer
