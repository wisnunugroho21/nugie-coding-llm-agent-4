"""
Offline tests for The Stack v2 ingestion — no network, no AWS, no HF token.

The HF stream and S3 content resolver are injected with fakes, so we exercise the
real field-mapping, size gating, error handling, and the hand-off into
data_pipeline exactly as production would, minus the external I/O.
"""

from __future__ import annotations

import unittest

from data_ingestion import StackV2Config, StackV2Loader, build_s3_content_resolver, row_to_document
from data_pipeline import PipelineConfig, run_pipeline
from data_pipeline.raw.languages import language_category

# A couple of realistic Stack v2 metadata rows (contents live in S3 by blob_id).
FAKE_ROWS = {
    "Python": [
        {"blob_id": "b1", "path": "acme/util/math.py", "repo_name": "acme/util",
         "star_events_count": 128, "src_encoding": "utf-8", "length_bytes": 60,
         "gha_updated_at": "2023-05-01T12:00:00Z", "detected_licenses": ["MIT"]},
        {"blob_id": "b2", "path": "acme/util/big.py", "repo_name": "acme/util",
         "star_events_count": 5, "src_encoding": "utf-8",
         "length_bytes": 9 * 1024 * 1024},                      # oversize -> skipped
        {"blob_id": "", "path": "acme/util/noblob.py", "length_bytes": 10},  # no blob -> skipped
        {"blob_id": "b3", "path": "acme/util/broken.py", "repo_name": "acme/util",
         "star_events_count": 1, "src_encoding": "utf-8", "length_bytes": 20},  # fetch error -> skipped
    ]
}
FAKE_CONTENTS = {
    # A substantial single function (low def/line ratio) so it survives the
    # Table-12 heuristic filters and proves the ingestion -> pipeline hand-off.
    "b1": (
        "def running_mean(values, window):\n"
        "    out = []\n"
        "    acc = 0.0\n"
        "    for i, v in enumerate(values):\n"
        "        acc += v\n"
        "        if i >= window:\n"
        "            acc -= values[i - window]\n"
        "        if i >= window - 1:\n"
        "            out.append(acc / window)\n"
        "    return out\n"
    ),
    # b3 intentionally absent -> resolver raises -> skipped
}


def fake_stream(cfg, language):
    return iter(FAKE_ROWS.get(language, []))


def fake_resolver(blob_id, src_encoding):
    return FAKE_CONTENTS[blob_id]  # KeyError for b3 -> treated as fetch error


class TestRowMapping(unittest.TestCase):
    def test_field_mapping(self):
        row = FAKE_ROWS["Python"][0]
        doc = row_to_document(row, "print(1)\n", "Python")
        self.assertEqual(doc.path, "acme/util/math.py")
        self.assertEqual(doc.language, "Python")
        self.assertEqual(doc.category, language_category("Python"))
        self.assertEqual(doc.stars, 128)
        self.assertEqual(doc.doc_id, "b1")
        self.assertGreater(doc.commit_time, 0)          # timestamp parsed
        self.assertEqual(doc.source, "the-stack-v2")
        self.assertEqual(doc.meta["src_encoding"], "utf-8")


class TestLoader(unittest.TestCase):
    def test_gating_and_error_handling(self):
        cfg = StackV2Config(languages=("Python",))
        loader = StackV2Loader(cfg, content_resolver=fake_resolver, stream_factory=fake_stream)
        docs = list(loader.iter_documents())
        self.assertEqual(len(docs), 1)                  # only b1 survives
        self.assertEqual(docs[0].doc_id, "b1")
        self.assertEqual(loader.stats["skipped_oversize"], 1)
        self.assertEqual(loader.stats["skipped_no_blob"], 1)
        self.assertEqual(loader.stats["skipped_fetch_error"], 1)
        self.assertEqual(loader.stats["yielded"], 1)

    def test_limit(self):
        rows = [{"blob_id": f"x{i}", "path": f"p{i}.py", "length_bytes": 10,
                 "src_encoding": "utf-8"} for i in range(10)]
        cfg = StackV2Config(languages=("Python",), limit=3)
        loader = StackV2Loader(cfg, content_resolver=lambda b, e: "x = 1\n",
                               stream_factory=lambda c, l: iter(rows))
        self.assertEqual(len(list(loader.iter_documents())), 3)


class TestResolverErrorMessage(unittest.TestCase):
    def test_missing_deps_raise_actionable_error(self):
        # boto3/smart_open are not installed in this env -> should raise with guidance.
        try:
            import boto3  # noqa: F401
            import smart_open  # noqa: F401
            self.skipTest("boto3/smart_open present; skip missing-deps assertion")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError) as ctx:
            build_s3_content_resolver(StackV2Config())
        self.assertIn("boto3", str(ctx.exception))


class TestIngestionFeedsPipeline(unittest.TestCase):
    def test_end_to_end_into_refinecode(self):
        cfg = StackV2Config(languages=("Python",))
        loader = StackV2Loader(cfg, content_resolver=fake_resolver, stream_factory=fake_stream)
        kept, stats = run_pipeline(loader.iter_documents(), PipelineConfig.fast_demo())
        # b1 is clean Python and should survive the whole pipeline.
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.stage_counts["0_input"], 1)


if __name__ == "__main__":
    unittest.main()
