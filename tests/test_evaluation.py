"""
Tests for the HumanEval/MBPP Pass@k harness. The load-bearing check is that the
ORACLE generator (reference solutions) scores pass@1 == 1.0 on both benchmarks —
that proves program assembly + execution + metric are correct, independent of any
model. A deliberately wrong generator must score 0.0.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from evaluation import EvalConfig, evaluate, pass_at_k, sample_humaneval, sample_mbpp
from evaluation.extract import extract_humaneval_completion, truncate_at_stops
from evaluation.generator import FunctionGenerator, OracleGenerator


class TestMetric(unittest.TestCase):
    def test_pass_at_k_values(self):
        self.assertEqual(pass_at_k(n=5, c=0, k=1), 0.0)
        self.assertEqual(pass_at_k(n=5, c=5, k=1), 1.0)
        self.assertAlmostEqual(pass_at_k(n=5, c=1, k=1), 0.2)      # c/n
        self.assertAlmostEqual(pass_at_k(n=10, c=1, k=10), 1.0)    # k==n, c>=1


class TestExtraction(unittest.TestCase):
    def test_truncate_at_stops(self):
        body = "    return a + b\n"
        gen = body + "\ndef other():\n    pass\n"
        self.assertEqual(truncate_at_stops(gen), body)

    def test_strip_echoed_prompt(self):
        prompt = "def f():\n"
        gen = prompt + "    return 1\n"
        self.assertEqual(extract_humaneval_completion(prompt, gen), "    return 1\n")


class TestOracleSelfTest(unittest.TestCase):
    def test_humaneval_oracle_is_perfect(self):
        probs = sample_humaneval()
        res = evaluate(probs, OracleGenerator(probs), EvalConfig(ks=(1,)))
        self.assertEqual(res.pass_at_k[1], 1.0)
        self.assertTrue(all(r.n_passed == r.n_samples for r in res.per_problem))

    def test_mbpp_oracle_is_perfect(self):
        probs = sample_mbpp()
        res = evaluate(probs, OracleGenerator(probs), EvalConfig(ks=(1,)))
        self.assertEqual(res.pass_at_k[1], 1.0)


class TestWrongGeneratorFails(unittest.TestCase):
    def test_humaneval_wrong_scores_zero(self):
        probs = sample_humaneval()
        wrong = FunctionGenerator(lambda p, t, n: "    return None\n")
        res = evaluate(probs, wrong, EvalConfig(ks=(1,)))
        self.assertEqual(res.pass_at_k[1], 0.0)

    def test_mbpp_wrong_scores_zero(self):
        probs = sample_mbpp()
        wrong = FunctionGenerator(lambda p, t, n: "def unrelated():\n    return 0\n")
        res = evaluate(probs, wrong, EvalConfig(ks=(1,)))
        self.assertEqual(res.pass_at_k[1], 0.0)


class TestConfigValidation(unittest.TestCase):
    def test_k_must_not_exceed_n(self):
        with self.assertRaises(ValueError):
            EvalConfig(n_samples=1, ks=(1, 10))


class TestModelGeneratorRuns(unittest.TestCase):
    def test_tiny_model_eval_runs(self):
        import flax.nnx as nnx

        from evaluation.generator import ModelGenerator
        from training import KimiLinear, demo_model_config
        from training.tokenizer import ByteTokenizer

        model = KimiLinear(demo_model_config(), rngs=nnx.Rngs(0))
        gen = ModelGenerator(model, ByteTokenizer(), seed=0)
        # One problem, tiny generation budget — just prove it runs and scores in [0,1].
        res = evaluate(sample_humaneval(), gen,
                       EvalConfig(ks=(1,), limit=1, max_new_tokens=16, timeout_s=10.0))
        self.assertIn(res.pass_at_k[1], (0.0, 1.0))
        self.assertEqual(len(res.per_problem), 1)


if __name__ == "__main__":
    unittest.main()
