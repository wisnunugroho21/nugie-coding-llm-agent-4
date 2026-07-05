"""
Tests for the two-stage SFT stage (OpenCoder Sec. 4). Pure stdlib unittest.
"""

from __future__ import annotations

import unittest

from sft import (
    SFTConfig,
    TestSetReference,
    build_realuser,
    compose_two_stage,
    decontaminate,
    format_example,
    synthesize_diverse,
    synthesize_educational,
    synthesize_package,
    wrap_examples,
)
from sft.config import STAGE1_COUNTS, STAGE2_COUNTS
from sft.models import InstructionExample
from synth_common.teacher import MockTeacher


class TestDiverse(unittest.TestCase):
    def test_cleans_ads_and_validates(self):
        cfg = SFTConfig().diverse
        out = list(synthesize_diverse(
            ["Learn Python. Subscribe now! Advertisement: buy now."], cfg, MockTeacher(), seed=3))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "diverse_instruct")
        self.assertTrue(out[0].meta["validated"])


class TestEducational(unittest.TestCase):
    def test_low_quality_seed_rejected(self):
        cfg = SFTConfig().educational
        out = list(synthesize_educational(["x=1"], cfg, MockTeacher()))  # unparseable-as-lesson, low score
        self.assertEqual(out, [])

    def test_high_quality_seed_accepted(self):
        cfg = SFTConfig().educational
        seed = 'def gcd(a, b):\n    """Greatest common divisor."""\n    while b:\n        a, b = b, a % b\n    return a\n'
        out = list(synthesize_educational([seed], cfg, MockTeacher()))
        self.assertEqual(len(out), 1)
        self.assertGreaterEqual(out[0].meta["seed_quality"], cfg.min_seed_score)


class TestPackage(unittest.TestCase):
    def test_real_pydoc_signatures(self):
        cfg = SFTConfig().package
        cfg = type(cfg)(libraries=("math",), max_apis_per_library=5)
        out = list(synthesize_package(cfg, MockTeacher()))
        self.assertTrue(out)
        self.assertTrue(all(ex.source == "package_instruct" for ex in out))
        self.assertTrue(all(ex.meta["library"] == "math" for ex in out))


class TestRealUser(unittest.TestCase):
    def test_regenerates_low_quality_code_response(self):
        dialogues = [{"instruction": "Write a Python function to sum a list using def", "response": "ok"}]
        out = list(build_realuser(dialogues, MockTeacher()))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].meta["regenerated"])
        self.assertGreater(len(out[0].response), 40)

    def test_non_code_dialogue_dropped(self):
        out = list(build_realuser([{"instruction": "What is the capital of France?", "response": "Paris"}]))
        self.assertEqual(out, [])


class TestDecontamination(unittest.TestCase):
    def test_entry_point_and_ngram_removal(self):
        exs = [
            InstructionExample("solve it", "def has_close_elements(a, t): pass", "diverse_instruct"),  # entry point
            InstructionExample(
                "q", "the quick brown fox jumps over the lazy dog again and again today", "diverse_instruct"),  # ngram
            InstructionExample("safe", "def unrelated(): return 1", "diverse_instruct"),
        ]
        ref = TestSetReference(
            texts=["the quick brown fox jumps over the lazy dog again and again today please"],
            entry_points=["has_close_elements"])
        kept, report = decontaminate(exs, ref, SFTConfig().decontam)
        self.assertEqual(report.removed_entry_point, 1)
        self.assertEqual(report.removed_ngram, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].instruction, "safe")


class TestCompose(unittest.TestCase):
    def test_two_stage_quota_and_tagging(self):
        # Build small pools; quotas are huge so all get drawn.
        diverse = [InstructionExample(f"q{i}", "a", "diverse_instruct") for i in range(5)]
        evol = list(wrap_examples([{"instruction": "x", "response": "y"}], "evol_instruct"))
        pools = {"diverse_instruct": diverse, "evol_instruct": evol}
        stages, report = compose_two_stage(pools, SFTConfig())
        self.assertEqual(report.stage_targets[1], dict(STAGE1_COUNTS))
        self.assertEqual(report.stage_targets[2], dict(STAGE2_COUNTS))
        self.assertTrue(all(ex.stage == 1 for ex in stages[1]))
        self.assertTrue(all(ex.stage == 2 for ex in stages[2]))
        self.assertEqual(len(stages[1]), 5)   # only diverse present in stage 1
        self.assertEqual(len(stages[2]), 1)   # only evol present in stage 2


class TestFormat(unittest.TestCase):
    def test_chat_template(self):
        ex = InstructionExample("do X", "here is code", "diverse_instruct")
        text = format_example(ex)
        self.assertIn("<|system|>", text)
        self.assertIn("<|user|>", text)
        self.assertIn("do X", text)
        self.assertIn("here is code", text)


if __name__ == "__main__":
    unittest.main()
