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

Synthesizers take a `TeacherModel`; the default offline `MockTeacher` emits
runnable code so the full synthesize → execute → validate flow works with no API.
Swap in a real client (see [synth_common/teacher.py](../synth_common/teacher.py)).
See [scripts/run_post_training_demo.py](../scripts/run_post_training_demo.py) for
an end-to-end run.
