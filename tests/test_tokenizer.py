"""
Tests for the BPE tokenizer: training, exact round-trip on code, atomic special
tokens, save/load, and end-to-end training with the model vocab set from it.
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from training.tokenizer import SPECIAL_TOKENS, BPETokenizer, ByteTokenizer

_CODE_CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "class Stack:\n    def push(self, x):\n        self.items.append(x)\n",
    "for i in range(10):\n    print(i * i)\n",
    "import math\n\ndef area(r):\n    return math.pi * r * r\n",
] * 20  # repeat so BPE has frequency to merge


class TestByteTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = ByteTokenizer()
        s = "def f(x):\n    return x + 1\n"
        self.assertEqual(tok.decode(tok.encode(s)), s)
        self.assertEqual(tok.vocab_size, 256)


class TestBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer.train(_CODE_CORPUS, vocab_size=600, min_frequency=2)

    def test_vocab_and_special_ids(self):
        self.assertGreater(self.tok.vocab_size, 256)      # merged beyond raw bytes
        self.assertEqual(self.tok.eos_id, self.tok.token_id("<|endoftext|>"))
        self.assertIsNotNone(self.tok.pad_id)

    def test_exact_roundtrip_on_code(self):
        s = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
        self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_special_tokens_are_atomic(self):
        for marker in SPECIAL_TOKENS:
            ids = self.tok.encode(marker)
            self.assertEqual(ids, [self.tok.token_id(marker)],
                             msg=f"{marker} should encode to a single id")

    def test_chat_template_tokenizes_markers_atomically(self):
        text = "<|user|>\nhi\n<|assistant|>\ndef f(): pass\n<|end|>"
        ids = self.tok.encode(text)
        self.assertIn(self.tok.token_id("<|user|>"), ids)
        self.assertIn(self.tok.token_id("<|assistant|>"), ids)
        self.assertIn(self.tok.token_id("<|end|>"), ids)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tok.json")
            self.tok.save(path)
            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.vocab_size, self.tok.vocab_size)
            s = "print('hello, world')\n"
            self.assertEqual(loaded.encode(s), self.tok.encode(s))
            self.assertEqual(loaded.eos_id, self.tok.eos_id)


class TestTrainingWithBPE(unittest.TestCase):
    def test_run_phase_sets_vocab_and_trains(self):
        import json

        import flax.nnx as nnx

        from training import BPETokenizer as BPE, run_phase
        from training.config import TrainConfig, demo_model_config

        with tempfile.TemporaryDirectory() as d:
            tokpath = os.path.join(d, "tok.json")
            BPE.train(_CODE_CORPUS, vocab_size=600).save(tokpath)
            corpus = os.path.join(d, "corpus.jsonl")
            with open(corpus, "w") as fh:
                for t in _CODE_CORPUS[:8]:
                    fh.write(json.dumps({"content": t}) + "\n")

            model_cfg = demo_model_config()               # vocab_size 256 by default
            cfg = TrainConfig(phase="pretrain", data_paths=[corpus], model=model_cfg,
                              seq_len=64, batch_size=2, steps=4, warmup_steps=1,
                              stable_steps=3, tokenizer_path=tokpath, log_every=100)
            model = run_phase(cfg)
            # Model head must now match the tokenizer vocab, not 256.
            self.assertGreater(model.cfg.vocab_size, 256)
            self.assertEqual(model.lm_head.kernel.value.shape[-1], model.cfg.vocab_size)


if __name__ == "__main__":
    unittest.main()
