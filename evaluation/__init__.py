"""
evaluation — HumanEval / MBPP Pass@k harness for the trained code model.

    from evaluation import evaluate, EvalConfig, sample_humaneval
    from evaluation.generator import OracleGenerator

    problems = sample_humaneval()
    result = evaluate(problems, OracleGenerator(problems), EvalConfig(ks=(1,)))
    print(result.summary())          # oracle -> pass@1 1.000 (harness self-test)

Pieces: data.py (loaders + bundled sample), problems.py (program assembly),
extract.py (completion/code-block extraction), generator.py (model / function /
oracle), harness.py (driver), metrics.py (unbiased pass@k). Execution reuses the
timeout-guarded subprocess sandbox in synth_common.execution.
"""

from .config import EvalConfig
from .data import load_humaneval, load_mbpp, sample_humaneval, sample_mbpp
from .harness import EvalResult, ProblemResult, evaluate
from .metrics import aggregate_pass_at_k, pass_at_k

__all__ = [
    "EvalConfig", "evaluate", "EvalResult", "ProblemResult",
    "load_humaneval", "load_mbpp", "sample_humaneval", "sample_mbpp",
    "pass_at_k", "aggregate_pass_at_k",
]
