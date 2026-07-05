# Two-Stage Instruction Tuning / SFT (OpenCoder Sec. 4)

Builds the post-training instruction data and the two-stage recipe: **Stage 1**
(theory + breadth) then **Stage 2** (practical coding). Four synthesis pipelines
(Fig. 5) produce data; open-source corpora are loaded and tagged; everything is
decontaminated against the eval sets, composed per Table 5, and chat-formatted.

## Synthesis pipelines (Sec. 4.1, Fig. 5)

| Pipeline | Module | What it does |
|----------|--------|--------------|
| Large-scale Diverse (a) | [synth_diverse.py](synth_diverse.py) | clean web seed → task-spec (lang/difficulty/type) → question @ T=1.0 → answer → **test-validate** → refine (comments) |
| Educational (b) | [synth_educational.py](synth_educational.py) | **scorer** selects HQ code seeds → teacher QA + tests → keep only test-passing |
| Package-related (c) | [synth_package.py](synth_package.py) | **real PyDoc/`inspect`** pulls current API signatures → teacher writes up-to-date QA |
| RealUser | [realuser.py](realuser.py) | filter dialogues to code-related → clean → regenerate low-quality responses |

## Assembly

- **Decontamination** ([decontaminate.py](decontaminate.py), Sec. 4.4): drop any
  example containing a test-set **entry-point** name, or sharing a **10-gram** with
  HumanEval / MBPP.
- **Two-stage composition** ([compose.py](compose.py), Table 5):

  | Stage 1 (target) | Stage 2 (target) |
  |------------------|------------------|
  | RealUser-Instruct 0.7M · Diverse 2.3M · Filtered Infinity 1.0M | McEval 36K · Evol 111K · Educational 110K · Package 110K |

- **Chat formatting** ([format.py](format.py)) → training-ready text records.
- **Training recipe** ([config.py](config.py), Sec. 4.3): Stage 1 = 1 epoch,
  batch 4096, LR 2e-5; Stage 2 = 3 epochs, batch 512, LR 5e-5; 100 warm-up,
  cosine.

## Usage

```python
from sft import (synthesize_diverse, synthesize_educational, synthesize_package,
                 build_realuser, wrap_examples, decontaminate, TestSetReference,
                 compose_two_stage, format_dataset, SFTConfig)

cfg = SFTConfig()
pkg  = list(synthesize_package(cfg.package))                      # real, offline
edu  = list(synthesize_educational(code_seeds, cfg.educational))  # test-validated
evol = list(wrap_examples(load_evol_instruct(), "evol_instruct")) # open-source pool

clean, drep = decontaminate(pkg + edu + evol,
    TestSetReference(texts=humaneval_prompts, entry_points=humaneval_entrypoints), cfg.decontam)
stages, crep = compose_two_stage(group_by_source(clean), cfg)
train_records = list(format_dataset(stages[1]))                  # -> tokenize & train
```

## Teacher backends

Synthesizers take a `TeacherModel`. The default offline `MockTeacher` emits
runnable code so the full synthesize → execute → validate flow works with no API.
For real data, swap in a production backend from
[synth_common/clients.py](../synth_common/clients.py):

| Backend | Use | Notes |
|---------|-----|-------|
| `ClaudeTeacher` | **Recommended** — Anthropic SDK, default `claude-opus-4-8` | Adaptive thinking on; does **not** forward `temperature` (Opus 4.8 rejects sampling params); SDK auto-retries. Creds via `ANTHROPIC_API_KEY` or `ant auth login`. |
| `OpenAICompatibleTeacher` | Local **vLLM** / any OpenAI-compatible server | `base_url=http://localhost:8000/v1`; forwards `temperature` (open models accept it). |
| `CachingTeacher` | Wrap either one | On-disk cache keyed by (model, prompt, temp, max_tokens) — dedup + resumability for large runs. |

```python
from synth_common import build_teacher, ClaudeTeacher
teacher = build_teacher("claude", cache_dir=".teacher_cache")   # or ClaudeTeacher(...)
edu = list(synthesize_educational(code_seeds, cfg.educational, teacher))
```

The end-to-end demo takes a backend flag:

```bash
python scripts/run_post_training_demo.py --teacher claude --cache-dir .teacher_cache
python scripts/run_post_training_demo.py --teacher vllm --model Qwen2.5-Coder-7B --base-url http://localhost:8000/v1
```

See [scripts/run_post_training_demo.py](../scripts/run_post_training_demo.py).
