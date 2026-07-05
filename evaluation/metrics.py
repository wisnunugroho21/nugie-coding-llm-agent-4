"""
The unbiased pass@k estimator from the HumanEval paper (Chen et al., 2021).

For a problem where `n` samples were drawn and `c` of them passed, the unbiased
estimate of the probability that at least one of `k` samples passes is:

    pass@k = 1 - C(n-c, k) / C(n, k)

computed in the numerically stable product form. pass@1 with n samples reduces to
c/n. The benchmark score is the mean of pass@k over all problems.
"""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k given `c` of `n` samples passed."""
    if k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate_pass_at_k(per_problem: list[tuple[int, int]], ks: tuple[int, ...]) -> dict[int, float]:
    """`per_problem` = list of (n_samples, n_passed). Returns {k: mean pass@k}."""
    if not per_problem:
        return {k: 0.0 for k in ks}
    out: dict[int, float] = {}
    for k in ks:
        out[k] = float(np.mean([pass_at_k(n, c, k) for n, c in per_problem]))
    return out
