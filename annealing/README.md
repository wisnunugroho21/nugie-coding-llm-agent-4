# Annealing Data Stage (OpenCoder Sec. 2.3)

Builds the ~100B-token **annealing mixture** used at the end of pretraining — a
high-quality data blend that is ~84% original RefineCode (to avoid catastrophic
forgetting) plus higher-signal sources. Under the WSD schedule the peak LR decays
to 1e-5 across exactly this data.

## Table 3 mixture (what this stage assembles)

| Source (`source` tag)          | Paper tokens | How it's built here |
|--------------------------------|--------------|---------------------|
| Original Data (`refinecode`)   | 83.94 B (84%) | the cleaned corpus from `data_pipeline`, passed through |
| Algorithmic Corpus (`algorithmic`) | 12.44 B | [algorithmic.py](algorithmic.py) — keyword sampling ("leetcode", "solution", …) |
| HQ Code Snippet (`synthetic_snippet`) | 2.71 B | [synthetic.py](synthetic.py) — teacher-synthesized **+ test-validated** (only passing code kept) |
| Code Textbooks (`code_textbook`) | 0.91 B | [textbooks.py](textbooks.py) — teacher rewrites HQ snippets into educational prose |

[mixer.py](mixer.py) draws from each source up to its Table-3 share of a token
budget and reports target-vs-actual proportions. [config.py](config.py) holds the
proportions and the WSD schedule (peak LR 3e-4 → 1e-5, 2000-step / 8B-token
warm-up, global batch 1024).

## Usage

```python
from annealing import build_annealing_data, AnnealingConfig
from data_pipeline.io_utils import read_jsonl

refine = list(read_jsonl("refined.jsonl"))          # data_pipeline output
seeds  = ["def gcd(a,b):\n    while b: a,b=b,a%b\n    return a\n", ...]
mix, report = build_annealing_data(refine, seeds, total_token_budget=100_000_000_000)
print(report.summary())
```

The synthesis steps take a `TeacherModel` (default: offline `MockTeacher`, which
emits runnable code so test-validation actually executes). Swap in a real Claude
/ vLLM client for production — see [synth_common/teacher.py](../synth_common/teacher.py).

## Notes / extension
- **Token counts** use a heuristic estimator; pass the real tokenizer as
  `counter=` to `assemble_mixture` for exact budgeting.
- **Test validation** runs generated code in a timeout-guarded subprocess
  ([synth_common/execution.py](../synth_common/execution.py)); sandbox it (container
  / nsjail) before running untrusted synthesis at scale.
