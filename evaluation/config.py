"""Evaluation configuration."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EvalConfig:
    n_samples: int = 1              # samples per problem (n>1 needed for pass@k, k>1)
    ks: tuple[int, ...] = (1,)      # which pass@k to report
    temperature: float = 0.0        # 0 = greedy; raise (e.g. 0.2/0.8) for n_samples>1
    max_new_tokens: int = 256
    timeout_s: float = 10.0
    limit: int | None = None        # cap #problems (quick runs)
    n_workers: int = 1              # parallel problems (stateless generators only)

    def __post_init__(self) -> None:
        if any(k > self.n_samples for k in self.ks):
            raise ValueError(
                f"pass@k needs k <= n_samples; got ks={self.ks}, n_samples={self.n_samples}"
            )
