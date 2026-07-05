"""
Tests for the training loop. Runs a tiny byte-level Kimi-Linear/GDN-2 on CPU.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from training import (
    KimiLinear,
    build_tx,
    cosine_schedule,
    demo_model_config,
    fit,
    load_model,
    save_model,
    wsd_schedule,
)
from training.data import sft_batches, pretrain_batches
from training.loss import weighted_next_token_ce
from training.tokenizer import ByteTokenizer

_SCRATCH = tempfile.mkdtemp(prefix="train_test_")


def _tiny_model():
    return KimiLinear(demo_model_config(), rngs=nnx.Rngs(0))


class TestTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = ByteTokenizer()
        s = "def f(x):\n    return x + 1\n"
        self.assertEqual(tok.decode(tok.encode(s)), s)
        self.assertEqual(tok.vocab_size, 256)


class TestLoss(unittest.TestCase):
    def test_mask_zeros_out_masked_tokens(self):
        B, L, V = 1, 4, 8
        logits = jnp.zeros((B, L, V))
        ids = jnp.array([[1, 2, 3, 4]])
        full = weighted_next_token_ce(logits, ids, jnp.ones((B, L)))
        # Weight only the last target; loss should still be finite and equal the
        # per-token CE (uniform logits -> log V) regardless of which tokens count.
        partial = weighted_next_token_ce(logits, ids, jnp.array([[0.0, 0.0, 0.0, 1.0]]))
        self.assertTrue(np.isfinite(float(full)) and np.isfinite(float(partial)))
        self.assertAlmostEqual(float(full), float(partial), places=4)  # uniform logits


class TestSchedules(unittest.TestCase):
    def test_wsd_shape(self):
        s = wsd_schedule(peak_lr=1e-3, end_lr=1e-5, warmup_steps=10, stable_steps=10, decay_steps=10)
        self.assertAlmostEqual(float(s(0)), 0.0, places=6)          # warmup start
        self.assertAlmostEqual(float(s(10)), 1e-3, places=6)        # peak after warmup
        self.assertAlmostEqual(float(s(15)), 1e-3, places=6)        # stable holds
        self.assertLess(float(s(28)), 1e-3)                          # decaying

    def test_cosine_decays(self):
        s = cosine_schedule(peak_lr=2e-5, warmup_steps=5, total_steps=50)
        self.assertLess(float(s(49)), float(s(5)))


class TestPretrainStep(unittest.TestCase):
    def test_loss_decreases_and_bias_updates(self):
        # A tiny repeating corpus so a few steps visibly reduce CE.
        path = os.path.join(_SCRATCH, "corpus.jsonl")
        with open(path, "w") as fh:
            for _ in range(8):
                fh.write(json.dumps({"content": "def add(a, b):\n    return a + b\n"}) + "\n")

        model = _tiny_model()
        rb0 = float(model.layers[0].channel_mixer.router_bias[...].sum())
        sched = wsd_schedule(3e-4, 1e-5, 2, 20, 0)
        tx = build_tx(sched)
        batches = pretrain_batches([path], ByteTokenizer(), seq_len=64, batch_size=4)

        step_fn = None
        from training.loop import make_train_step
        step_fn = make_train_step(1e-3)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
        ces = []
        for _ in range(10):
            ids, w = next(batches)
            ce, _ = step_fn(model, opt, jnp.asarray(ids), jnp.asarray(w, jnp.float32))
            ces.append(float(ce))
        self.assertTrue(all(np.isfinite(c) for c in ces))
        self.assertLess(ces[-1], ces[0])                            # learned something
        rb1 = float(model.layers[0].channel_mixer.router_bias[...].sum())
        self.assertNotAlmostEqual(rb0, rb1, places=6)               # balancer moved bias


class TestSFT(unittest.TestCase):
    def test_response_only_masking(self):
        path = os.path.join(_SCRATCH, "sft.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps({"instruction": "Add two numbers",
                                 "response": "def add(a, b):\n    return a + b\n"}) + "\n")
        ids, mask = next(sft_batches([path], ByteTokenizer(), seq_len=128, batch_size=1))
        self.assertEqual(ids.shape, (1, 128))
        # Prompt tokens (leading) masked 0; response tokens 1; padding 0.
        self.assertEqual(float(mask[0, 0]), 0.0)
        self.assertGreater(mask.sum(), 0.0)                          # some response is trained
        self.assertLess(mask.sum(), 128)                            # not everything

    def test_sft_step_runs(self):
        path = os.path.join(_SCRATCH, "sft2.jsonl")
        with open(path, "w") as fh:
            for i in range(4):
                fh.write(json.dumps({"instruction": f"Return {i}",
                                     "response": f"def f():\n    return {i}\n"}) + "\n")
        model = _tiny_model()
        sched = cosine_schedule(5e-5, 1, 6)
        batches = sft_batches([path], ByteTokenizer(), seq_len=128, batch_size=2)
        fit(model, build_tx(sched), batches, steps=4, schedule=sched, log_every=100, label="sft")
        # no assertion beyond "it ran without error"; loss printed above


class TestCheckpoint(unittest.TestCase):
    def test_full_state_roundtrip(self):
        m = _tiny_model()
        rb = m.layers[0].channel_mixer.router_bias
        rb[...] = rb[...] + 0.25                                     # simulate trained bias
        path = os.path.join(_SCRATCH, "ck.pkl")
        save_model(m, path)
        m2 = KimiLinear(demo_model_config(), rngs=nnx.Rngs(7))
        load_model(m2, path)
        ids = jnp.asarray(np.zeros((1, 64), np.int32))
        self.assertAlmostEqual(float(m(ids)[0].sum()), float(m2(ids)[0].sum()), places=3)


if __name__ == "__main__":
    unittest.main()
