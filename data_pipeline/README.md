# RefineCode Data Processing Pipeline

A faithful, from-scratch implementation of the **RefineCode** data pipeline from
[*OpenCoder: The Open Cookbook For Top-Tier Code Large Language Models*](https://arxiv.org/abs/2411.04905)
(arXiv:2411.04905). It turns raw source files into clean pretraining data for a
code LLM — here, the Kimi-Linear/GDN-2 model in the repo root.

Every threshold is traceable to a section or table of the paper; see the module
docstrings and [`config.py`](config.py) for exact citations.

## Pipeline at a glance

```
raw files (JSONL)
     │
     ▼
1. preprocess        raw/preprocess.py     Sec. 2.1   size ≤ 8MB, known extension → language + category
2. exact dedup       dedup/exact.py        Sec. 2.1   SHA256; keep highest-star / latest commit (~75% are dups)
3. fuzzy dedup       dedup/minhash.py      Sec. 2.1   5-gram MinHash, 2048 perms, LSH 16×128 bands (file-level)
4. transform         transform/            Sec. 2.1   PII redaction + copyright/licence-header removal
5. heuristic filter  filtering/            App. A     quality-signal computation → rule execution (~130 rules)
6. downsample        sampling/downsample.py Sec. 2.1  cap high-resource langs (Java 409→200GB, HTML 213→64GB)
     │
     ▼
refined code corpus (JSONL)

  (separate track)
   web pages ──► web/fasttext_recall.py    Sec. 2.2 / App. C   filename recall + fastText code classifier
```

**Ordering rationale (paper App. A.1):** heuristic filtering runs as *late* as
possible, because it is the most-frequently-retuned stage — putting it last
avoids re-running expensive dedup after every threshold tweak. Filtering itself is
split into two steps (RedPajama-style): **quality-signal computation** then
**filtering execution**.

**Why file-level dedup?** The paper (App. B, Table 13) compares file-, repo-, and
chunk-level dedup; file-level removed the most redundancy and gave the best
HumanEval / MBPP Pass@1, so this pipeline deduplicates individual files.

## Install

```bash
pip install -r requirements.txt   # numpy (fastText optional)
```

## Quick start

```bash
# 1. generate the sample corpora
python scripts/make_sample_data.py

# 2. run the full code pipeline (fast 128-perm MinHash demo; add --full for 2048-perm)
python -m data_pipeline.cli run --input sample_data/raw_code.jsonl --output sample_data/refined.jsonl

# 3. recall code-related web data (trains the fastText classifier on a labelled seed)
python -m data_pipeline.cli recall \
    --input sample_data/web_docs.jsonl --output sample_data/web_code.jsonl \
    --seed  sample_data/web_seed.jsonl
```

Or from Python:

```python
from data_pipeline import run_pipeline, PipelineConfig
from data_pipeline.io_utils import read_jsonl, write_jsonl

kept, stats = run_pipeline(read_jsonl("raw_code.jsonl"), PipelineConfig.fast_demo())
print(stats.summary())          # per-stage survivor counts + per-rule hit counts
write_jsonl("refined.jsonl", kept)
```

## Input format

One JSON object per line (JSONL). Only `content` is required; the rest carry the
repo metadata used for dedup tie-breaking and language-aware filtering:

```json
{"content": "def f(): ...", "path": "owner/repo/util.py",
 "stars": 128, "commit_time": 1699900000.0, "source": "github"}
```

Fields map to [`CodeDocument`](models.py): `content`, `path`, `language`,
`category` (`code`/`data`/`text`), `repo_name`, `stars`, `commit_time`, `source`.
`language`/`category` are auto-resolved from the extension if omitted.

## Heuristic rules (exact paper thresholds)

General-code rules — **Table 11** — and Python-specific rules — **Table 12** — are
implemented verbatim in [`filtering/rules.py`](filtering/rules.py), with thresholds
in [`config.py`](config.py):

| Rule (signal)                                   | Fires when      | Source   |
|-------------------------------------------------|-----------------|----------|
| long-string line ratio                          | `> 0.2`         | Table 11 |
| long in-string word (>20 char) char ratio       | `> 0.4`         | Table 11 |
| hexadecimal char ratio                          | `> 0.4`         | Table 11 |
| placeholder lines (TODO/FIXME/"your code here") | `> 0.01`        | Table 11 |
| assert-statement line ratio                     | `> 0.4`         | Table 11 |
| Python: #functions / #lines                     | `> 0.2`         | Table 12 |
| Python: not AST-parseable                        | `== False`      | Table 12 |
| Python: import-line ratio                       | `> 0.3`         | Table 12 |

Natural-language rules (for `text`-category files) and general line-count/length
guards follow the paper's described intent and its RedPajama/Gopher lineage.
Tune any of them by editing `FilterConfig` — no code change needed.

## Extending toward the full paper

- **Languages:** [`raw/languages.py`](raw/languages.py) ships a curated subset of
  the paper's 607 languages (the 8 with language-specific rules + common
  long-tail). Add rows to `EXT_TO_LANG` / `LANGUAGE_CATEGORY` to widen coverage.
- **Language-specific rules:** the paper defines them for Python, C, C++, C#,
  Java, JavaScript, Go, HTML. Python is implemented; add analogous `Rule`s
  (e.g. C `goto` frequency) scoped by `languages=(...)`.
- **Scale:** every stage is a generator over `CodeDocument`; dedup/downsample are
  whole-corpus ops that currently materialize in memory. For out-of-core scale,
  shard by hash prefix (dedup) / language (downsample) and run per shard.
- **Annealing & SFT:** the paper's later stages build on this pretraining corpus
  — an *annealing* mix (paper Sec. 2.3: ~84% RefineCode + algorithmic corpus +
  synthetic high-quality snippets + code textbooks, ~100B tokens) and *two-stage*
  instruction tuning (Sec. 3: ~4M diverse examples, then ~367K code-specific).
  Both are downstream of this module and are the natural next thing to add.

## Tests

```bash
python -m unittest discover -s tests -v
```
