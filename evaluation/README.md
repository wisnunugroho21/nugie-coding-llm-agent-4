# Evaluation Harness — HumanEval / MBPP Pass@k

Measures the trained model's code-generation ability with the standard
**Pass@k** metric (Chen et al., 2021) on **HumanEval** and **MBPP** — the same
benchmarks OpenCoder reports (Sec. 5).

## How it works

For each problem: draw `n` completions from a generator → assemble each into a
runnable program → execute it in the timeout-guarded subprocess sandbox (reusing
[`synth_common.execution`](../synth_common/execution.py)) → count passes → report
`pass@k` via the unbiased estimator `1 - C(n-c,k)/C(n,k)` ([metrics.py](metrics.py)).

- **HumanEval** ([problems.py](problems.py)): the model completes a function
  signature+docstring; the program is `prompt + completion` (truncated at the next
  top-level construct), plus the problem's `check(candidate)` test. Full fenced
  functions from instruct models are used directly.
- **MBPP**: the model writes a function; the program is setup + code + the
  `assert` `test_list`.

## Harness self-test

The design guarantee: an **oracle** generator (reference solutions) must score
`pass@1 == 1.000`. That validates program assembly + execution + metric
independent of any model.

```bash
python -m evaluation.cli --benchmark humaneval --oracle   # -> pass@1: 1.000
python -m evaluation.cli --benchmark mbpp --oracle        # -> pass@1: 1.000
```

## Usage

```bash
# evaluate a trained checkpoint (greedy pass@1) on the real benchmark
python -m evaluation.cli --benchmark humaneval --data HumanEval.jsonl --model ckpt/sft.pkl

# pass@10 from a real LLM teacher (needs `anthropic` + credentials)
python -m evaluation.cli --benchmark mbpp --data mbpp.jsonl \
    --claude --n 10 --k 1 10 --temperature 0.8 --workers 8
```

```python
from evaluation import evaluate, EvalConfig, load_humaneval
from evaluation.generator import ModelGenerator
from training import KimiLinear, demo_model_config, load_model
from training.tokenizer import ByteTokenizer

model = load_model(KimiLinear(demo_model_config(), rngs=nnx.Rngs(0)), "ckpt/sft.pkl")
gen = ModelGenerator(model, ByteTokenizer())
res = evaluate(load_humaneval("HumanEval.jsonl"), gen, EvalConfig(ks=(1,)))
print(res.summary())
```

Without `--data`, a small bundled sample runs offline (canonical solutions pass,
so the oracle scores 1.0).

## Generators ([generator.py](generator.py))
| Generator | Use |
|-----------|-----|
| `ModelGenerator` | samples from the trained Kimi-Linear/GDN-2 via its `step` API (temperature sampling; greedy at T=0) |
| `FunctionGenerator` | wraps any `callable(prompt, temperature, max_new_tokens)` — e.g. a `ClaudeTeacher` |
| `OracleGenerator` | reference solutions — the harness self-test |

## Data formats
- **HumanEval** JSONL: `task_id`, `prompt`, `entry_point`, `test`,
  `canonical_solution` (openai/human-eval).
- **MBPP** JSONL: `task_id`, `text`/`prompt`, `test_list`, `test_setup_code`,
  `code` (google-research/mbpp).

## Notes
- pass@k for k>1 needs `--n > 1` and `--temperature > 0`; pass@1 with one greedy
  sample is the headline number.
- **Security:** candidate code runs in a timeout-guarded subprocess, not a hardened
  sandbox — run large benchmark sweeps inside a container / nsjail.
- The byte-level demo model won't actually solve problems; scale up the model +
  tokenizer (see [training/README.md](../training/README.md)) for real scores.

## Tests
```bash
python -m unittest tests.test_evaluation
```
