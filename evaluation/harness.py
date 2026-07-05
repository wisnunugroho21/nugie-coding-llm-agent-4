"""
The evaluation driver.

For each problem: draw `n_samples` completions from the generator, assemble each
into a runnable program, execute it in the timeout-guarded subprocess sandbox
(reusing `synth_common.execution.run_python` — the same harness the SFT synthesis
uses), count how many pass, and report pass@k via the unbiased estimator.

Note: pass@k for k>1 needs n_samples>1 with temperature>0. Pass@1 with a single
greedy sample (n_samples=1, temperature=0) is the common headline number.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

from synth_common.execution import run_python

from .config import EvalConfig
from .metrics import aggregate_pass_at_k


@dataclasses.dataclass
class ProblemResult:
    task_id: str
    n_samples: int
    n_passed: int


@dataclasses.dataclass
class EvalResult:
    per_problem: list[ProblemResult]
    pass_at_k: dict[int, float]

    def summary(self) -> str:
        lines = [f"Evaluated {len(self.per_problem)} problems:"]
        for k in sorted(self.pass_at_k):
            lines.append(f"  pass@{k}: {self.pass_at_k[k]:.3f}")
        n_solved = sum(1 for r in self.per_problem if r.n_passed > 0)
        lines.append(f"  problems with >=1 passing sample: {n_solved}/{len(self.per_problem)}")
        return "\n".join(lines)


def _eval_one(problem, generator, cfg: EvalConfig) -> ProblemResult:
    prompt = problem.prompt_for_model()
    passed = 0
    for _ in range(cfg.n_samples):
        gen = generator.generate(prompt, temperature=cfg.temperature,
                                 max_new_tokens=cfg.max_new_tokens)
        program = problem.build_program(gen)
        if run_python(program, timeout=cfg.timeout_s).passed:
            passed += 1
    return ProblemResult(problem.task_id, cfg.n_samples, passed)


def evaluate(problems, generator, cfg: EvalConfig | None = None) -> EvalResult:
    cfg = cfg or EvalConfig()
    problems = list(problems)[: cfg.limit] if cfg.limit else list(problems)

    if cfg.n_workers > 1:
        # Parallelize across problems (execution is subprocess/IO-bound). Use with
        # stateless generators (Function/Oracle); ModelGenerator holds a PRNG key.
        with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
            results = list(pool.map(lambda p: _eval_one(p, generator, cfg), problems))
    else:
        results = [_eval_one(p, generator, cfg) for p in problems]

    agg = aggregate_pass_at_k([(r.n_samples, r.n_passed) for r in results], cfg.ks)
    return EvalResult(per_problem=results, pass_at_k=agg)
