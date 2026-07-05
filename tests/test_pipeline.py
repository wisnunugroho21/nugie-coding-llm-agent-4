"""
Stage-by-stage tests for the RefineCode pipeline. Pure stdlib `unittest` (no
pytest needed):  python -m unittest discover -s tests
"""

from __future__ import annotations

import time
import unittest

from data_pipeline.config import (
    ExactDedupConfig,
    FilterConfig,
    MinHashDedupConfig,
    PipelineConfig,
    PreprocessConfig,
    TransformConfig,
    WebRecallConfig,
)
from data_pipeline.dedup.exact import exact_dedup
from data_pipeline.dedup.minhash import minhash_dedup
from data_pipeline.filtering import annotate_documents, filter_documents
from data_pipeline.filtering.quality_signals import compute_signals
from data_pipeline.models import CodeDocument
from data_pipeline.pipeline import run_pipeline
from data_pipeline.raw.preprocess import preprocess
from data_pipeline.transform.copyright import remove_copyright
from data_pipeline.transform.pii import redact_pii
from data_pipeline.web.fasttext_recall import FastTextRecaller


from data_pipeline.raw.languages import detect_language, language_category


def doc(content, path="a.py", stars=0, commit_time=0.0, **kw):
    """Build a document, resolving language/category the way preprocess would so
    stage-level tests that bypass preprocess still have category-scoped rules apply."""
    d = CodeDocument(content=content, path=path, stars=stars, commit_time=commit_time, **kw)
    if not d.language:
        # Only resolve when the extension is known; leave empty otherwise so the
        # preprocess language gate can still exercise its rejection path.
        lang = detect_language(path)
        if lang:
            d.language = lang
    if d.language and not d.category:
        d.category = language_category(d.language)
    return d


class TestPreprocess(unittest.TestCase):
    def test_size_and_language_gate(self):
        cfg = PreprocessConfig()
        big = doc("x" * (cfg.max_file_bytes + 1))
        unknown = doc("hello", path="a.unknownext")
        good = doc("print(1)\n", path="a.py")
        out = list(preprocess([big, unknown, good], cfg))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].language, "Python")
        self.assertEqual(out[0].category, "code")

    def test_newline_normalization(self):
        out = list(preprocess([doc("a\r\nb\r\n", path="a.py")], PreprocessConfig()))
        self.assertEqual(out[0].content, "a\nb\n")


class TestExactDedup(unittest.TestCase):
    def test_keeps_highest_star(self):
        a = doc("same\n", stars=5, commit_time=1)
        b = doc("same\n", stars=100, commit_time=1)
        c = doc("other\n", stars=1)
        out = list(exact_dedup([a, b, c], ExactDedupConfig()))
        self.assertEqual(len(out), 2)
        kept = next(d for d in out if d.content == "same\n")
        self.assertEqual(kept.stars, 100)


class TestFuzzyDedup(unittest.TestCase):
    def test_near_duplicates_collapse(self):
        base = (
            "def normalize(values):\n"
            "    total = sum(values)\n"
            "    if total == 0:\n"
            "        return [0.0 for _ in values]\n"
            "    scaled = [v / total for v in values]\n"
            "    # normalize the vector so its elements sum to exactly one\n"
            "    return scaled\n"
            "\n"
            "def clamp(value, low, high):\n"
            "    return max(low, min(high, value))\n"
        )
        near = base.replace("sum to exactly one", "sum to precisely one")  # 1-word edit
        far = "class Widget:\n    def render(self):\n        return '<div/>'\n" * 3
        cfg = MinHashDedupConfig(num_perm=128, bands=16, rows=8)
        out = list(minhash_dedup([doc(base, stars=1), doc(near, stars=9), doc(far)], cfg))
        # base+near collapse to one (higher-star survivor), far stays separate.
        self.assertEqual(len(out), 2)
        survivor = max((d for d in out if "normalize" in d.content), key=lambda d: d.stars)
        self.assertEqual(survivor.stars, 9)


class TestTransform(unittest.TestCase):
    def test_pii_redaction_keeps_valid_python(self):
        import ast

        src = 'password = "s3cr3t!"\nemail = "a@b.com"\nhost = "10.0.0.1"\n'
        out = redact_pii(doc(src), TransformConfig())
        self.assertNotIn("s3cr3t", out.content)
        self.assertIn("<email>", out.content)
        self.assertIn("<ip_address>", out.content)
        ast.parse(out.content)  # must remain syntactically valid

    def test_copyright_header_removed(self):
        src = "# Copyright (c) 2020 ACME\n# Licensed under MIT\n\ndef f():\n    return 1\n"
        out = remove_copyright(doc(src), TransformConfig())
        self.assertNotIn("Copyright", out.content)
        self.assertTrue(out.content.startswith("def f"))

    def test_copyright_not_removed_from_real_docstring(self):
        src = '"""This module does math."""\n\ndef f():\n    return 1\n'
        out = remove_copyright(doc(src), TransformConfig())
        self.assertEqual(out.content, src)


class TestFiltering(unittest.TestCase):
    def test_assert_heavy_removed_table11(self):
        src = "def t():\n" + "\n".join("    assert True" for _ in range(9)) + "\n"
        out = list(filter_documents([doc(src)], FilterConfig()))
        self.assertEqual(out, [])

    def test_placeholder_spam_removed_table11(self):
        src = "def f():\n    # TODO implement\n    pass\n"
        signals = compute_signals(doc(src))
        self.assertGreater(signals["placeholder_line_ratio"], 0.01)

    def test_unparseable_python_removed_table12(self):
        annotated = list(annotate_documents([doc("def broken(:\n  ???\n")], FilterConfig()))
        self.assertIn("py_not_parseable", annotated[0].filtered_by)

    def test_good_code_survives(self):
        src = ("def solve(nums):\n"
               "    total = 0\n"
               "    for n in nums:\n"
               "        if n % 2 == 0:\n"
               "            total += n\n"
               "    return total\n")
        out = list(filter_documents([doc(src)], FilterConfig()))
        self.assertEqual(len(out), 1)


class TestWebRecall(unittest.TestCase):
    def test_filename_and_classifier_recall(self):
        cfg = WebRecallConfig()
        seed = [
            ("def f(): return 1  # python function returns value", 1),
            ("import os and call os.path to join paths in code", 1),
            ("the sunny weather is lovely for a walk in the park today", 0),
            ("our restaurant serves pasta and pizza for dinner tonight", 0),
        ]
        recaller = FastTextRecaller(cfg).train(seed)
        docs = [
            doc("def quicksort(a): return sorted(a)  # sort the list", path="p.html", source="web"),
            doc("beautiful sunset over the calm ocean waves at the beach", path="q.html", source="web"),
            doc("numpy\nrequests\n", path="requirements.txt", source="web"),  # filename hit
        ]
        kept = list(recaller.recall(docs))
        paths = {d.path for d in kept}
        self.assertIn("requirements.txt", paths)   # filename recall
        self.assertIn("p.html", paths)              # classifier recall


class TestFullPipeline(unittest.TestCase):
    def test_end_to_end_monotonic_and_clean(self):
        now = time.time()
        docs = [
            doc("print('hello world')\nprint('again')\nx = 1 + 2\n", path="a.py", stars=5, commit_time=now),
            doc("print('hello world')\nprint('again')\nx = 1 + 2\n", path="b.py", stars=1, commit_time=now),  # exact dup
            doc("def broken(:\n ???\n", path="c.py"),  # unparseable -> filtered
        ]
        kept, stats = run_pipeline(docs, PipelineConfig.fast_demo())
        # exact dup collapsed, unparseable filtered -> exactly one survivor.
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].stars, 5)
        # stage counts are monotonically non-increasing.
        counts = list(stats.stage_counts.values())
        self.assertTrue(all(a >= b for a, b in zip(counts, counts[1:])))


if __name__ == "__main__":
    unittest.main()
