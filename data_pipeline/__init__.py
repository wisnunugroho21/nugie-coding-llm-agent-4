"""
data_pipeline — a faithful, from-scratch implementation of the **RefineCode** data
processing pipeline from *OpenCoder: The Open Cookbook For Top-Tier Code Large
Language Models* (arXiv:2411.04905).

It turns raw source files (GitHub / The Stack v2 / web) into clean pretraining
data for a code LLM, through the paper's stages: raw preprocessing, exact +
fuzzy deduplication, PII/copyright transformation, ~130 heuristic filtering rules
(App. A, Tables 11/12), high-resource-language downsampling, and code-related web
recall via a fastText classifier (Sec. 2.2 / App. C).

Quick start:

    from data_pipeline import run_pipeline, PipelineConfig
    from data_pipeline.io_utils import read_jsonl, write_jsonl

    docs = read_jsonl("raw.jsonl")
    kept, stats = run_pipeline(docs, PipelineConfig.fast_demo())
    print(stats.summary())
    write_jsonl("refined.jsonl", kept)
"""

from .config import PipelineConfig
from .models import CodeDocument
from .pipeline import PipelineStats, run_pipeline

__all__ = ["run_pipeline", "PipelineConfig", "PipelineStats", "CodeDocument"]
__version__ = "0.1.0"
