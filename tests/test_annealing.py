"""
Tests for the annealing-data stage (OpenCoder Sec. 2.3). Pure stdlib unittest.
"""

from __future__ import annotations

import unittest

from annealing import (
    AnnealingConfig,
    build_annealing_data,
    sample_algorithmic,
    synthesize_snippets,
    synthesize_textbooks,
)
from annealing.config import TABLE3_TOKENS_B, AlgorithmicConfig, SyntheticConfig, TextbookConfig
from data_pipeline.models import CodeDocument
from synth_common.teacher import MockTeacher


def code_doc(content, path="a.py", source="github"):
    return CodeDocument(content=content, path=path, language="Python", category="code", source=source)


class TestAlgorithmic(unittest.TestCase):
    def test_keyword_sampling_and_no_mutation(self):
        original = code_doc("def solve():\n    # leetcode dynamic programming\n    return 1\n", "algo.py")
        plain = code_doc("def util():\n    return 2\n", "util.py")
        out = list(sample_algorithmic([original, plain], AlgorithmicConfig()))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "algorithmic")
        self.assertGreaterEqual(out[0].meta["algorithmic_keyword_hits"], 1)
        # The original object must NOT be mutated (copy semantics).
        self.assertEqual(original.source, "github")


class TestSynthetic(unittest.TestCase):
    def test_snippets_are_test_validated(self):
        out = list(synthesize_snippets(["def add(a,b): return a+b"], SyntheticConfig(), MockTeacher()))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].meta["validated"])
        self.assertEqual(out[0].source, "synthetic_snippet")


class TestTextbooks(unittest.TestCase):
    def test_short_snippets_skipped(self):
        cfg = TextbookConfig(min_snippet_chars=50)
        out = list(synthesize_textbooks(["x=1"], cfg, MockTeacher()))
        self.assertEqual(out, [])

    def test_long_snippet_becomes_textbook(self):
        cfg = TextbookConfig(min_snippet_chars=20)
        long_snip = "def f(nums):\n    return [x for x in nums if x % 2 == 0]  # keep evens only\n"
        out = list(synthesize_textbooks([long_snip], cfg, MockTeacher()))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].category, "text")


class TestMixture(unittest.TestCase):
    def test_table3_proportions_normalize_to_one(self):
        cfg = AnnealingConfig()
        self.assertAlmostEqual(sum(cfg.mix.proportions.values()), 1.0, places=6)
        # RefineCode should dominate (~84%).
        self.assertAlmostEqual(cfg.mix.proportions["refinecode"], 83.94 / sum(TABLE3_TOKENS_B.values()), places=4)

    def test_assemble_respects_budget_and_reports(self):
        refine = [code_doc(f"def f{i}():\n    return {i}\n") for i in range(50)]
        seeds = ["def add(a,b): return a+b"]
        mix, report = build_annealing_data(refine, seeds, total_token_budget=500, cfg=AnnealingConfig())
        self.assertGreater(report.total_tokens, 0)
        # RefineCode is the largest share of the assembled tokens.
        self.assertEqual(max(report.per_source_tokens, key=report.per_source_tokens.get), "refinecode")
        # No document appears twice (identity) — copy semantics upheld.
        self.assertEqual(len({id(d) for d in mix}), len(mix))


if __name__ == "__main__":
    unittest.main()
