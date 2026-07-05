"""
synth_common — shared building blocks for OpenCoder's data-synthesis stages
(annealing, Sec. 2.3; and post-training SFT, Sec. 4):

  * teacher.py     — TeacherModel protocol + offline MockTeacher + code-block parsing
  * execution.py   — real subprocess code execution / test validation
  * token_count.py — pluggable token estimator for mixture budgeting
  * ngram.py       — word-level n-gram overlap for decontamination
  * prompts.py     — shared, paper-faithful prompt templates
"""

from .execution import ExecResult, run_python, validate_with_tests
from .ngram import build_banned_ngrams, has_overlap, ngrams
from .teacher import MockTeacher, TeacherModel, extract_code_blocks
from .token_count import TokenCounter, estimate_tokens

__all__ = [
    "TeacherModel", "MockTeacher", "extract_code_blocks",
    "ExecResult", "run_python", "validate_with_tests",
    "estimate_tokens", "TokenCounter",
    "ngrams", "build_banned_ngrams", "has_overlap",
]
