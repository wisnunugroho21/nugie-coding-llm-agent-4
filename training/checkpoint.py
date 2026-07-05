"""
Model checkpointing — the bridge between phases (pretrain -> anneal -> SFT).

Saves the model's **full** state keyed by state path, so a checkpoint loads back
into any freshly-constructed model of the same architecture. This is what lets
`anneal` and `sft` start from the previous phase's weights.

Crucially this includes the MoE `router_bias` variables — they are *not* trained
`nnx.Param`s (the optimizer skips them), but they ARE updated during training by
the aux-loss-free balancer, so they must be checkpointed or routing changes on
reload.

Optimizer state is not saved: each phase uses its own LR schedule and optimizer,
so a fresh optimizer per phase is the intended behavior.
"""

from __future__ import annotations

import pickle

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def _keyed_leaves(state) -> list[tuple[str, object]]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    return [(jax.tree_util.keystr(kp), v) for kp, v in leaves]


def save_model(model: nnx.Module, path: str) -> None:
    state = nnx.state(model)
    flat = {key: np.asarray(v) for key, v in _keyed_leaves(state)}
    with open(path, "wb") as fh:
        pickle.dump(flat, fh)


def load_model(model: nnx.Module, path: str) -> nnx.Module:
    with open(path, "rb") as fh:
        flat = pickle.load(fh)
    state = nnx.state(model)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    new_leaves = [jnp.asarray(flat[jax.tree_util.keystr(kp)]) for kp, _ in leaves]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, new_leaves))
    return model
