"""
Central configuration for the RefineCode data-processing pipeline.

Every threshold here is traceable to the OpenCoder paper ("OpenCoder: The Open
Cookbook For Top-Tier Code Large Language Models", arXiv:2411.04905) — the
pipeline that builds the *RefineCode* corpus (960B tokens, ~130 rules, 607
languages). Section / Table references are noted inline so the code stays
auditable against the paper.

Design note (paper App. A.1): heuristic filtering is decomposed into two steps —
**quality-signal computation** then **filtering execution** — and is deliberately
placed as LATE as possible in the pipeline. The stage ordering in `PipelineConfig`
reflects that: preprocess -> exact dedup -> fuzzy dedup -> transform -> filter ->
downsample.
"""

from __future__ import annotations

import dataclasses


# --------------------------------------------------------------------------- #
#  Stage 1 — Raw preprocessing / file-level admission (paper Sec. 2.1)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PreprocessConfig:
    # Files larger than 8 MB are "predominantly non-text" and dropped (Sec. 2.1).
    max_file_bytes: int = 8 * 1024 * 1024
    # Empty / whitespace-only files are dropped.
    min_content_chars: int = 1
    # Normalize line endings to '\n' and strip a UTF-8 BOM.
    normalize_newlines: bool = True
    # Only admit files whose extension maps to a known language (linguist-style).
    require_known_language: bool = True


# --------------------------------------------------------------------------- #
#  Stage 2 — Exact deduplication (paper Sec. 2.1: ~75% of files are exact dups)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ExactDedupConfig:
    # SHA256 over the (optionally normalized) file content.
    hash_algorithm: str = "sha256"
    # When collapsing an exact-duplicate group, keep the copy with the highest
    # star count, breaking ties by the most recent commit time (Sec. 2.1).
    keep_by: tuple[str, ...] = ("stars", "commit_time")


# --------------------------------------------------------------------------- #
#  Stage 3 — Fuzzy (near-)deduplication via MinHash + LSH (paper Sec. 2.1)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class MinHashDedupConfig:
    # Paper config: 5-gram shingles, 2048 MinHash permutations, LSH banded into
    # 16 bands x 128 rows (16 * 128 == 2048). Implied Jaccard threshold ~0.98,
    # i.e. only very near-identical files collapse.
    ngram: int = 5
    num_perm: int = 2048
    bands: int = 16
    rows: int = 128
    # Same tie-break as exact dedup when choosing a cluster representative.
    keep_by: tuple[str, ...] = ("stars", "commit_time")

    def __post_init__(self) -> None:
        if self.bands * self.rows != self.num_perm:
            raise ValueError(
                f"bands*rows ({self.bands}*{self.rows}) must equal num_perm ({self.num_perm})"
            )


# --------------------------------------------------------------------------- #
#  Stage 4 — Transformation: PII reduction + copyright removal (paper Sec. 2.1)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TransformConfig:
    redact_pii: bool = True
    remove_copyright: bool = True
    # Placeholders used when redacting (paper mentions e.g. "<name>", "<password>").
    email_placeholder: str = "<email>"
    ip_placeholder: str = "<ip_address>"
    secret_placeholder: str = "<password>"
    key_placeholder: str = "<key>"


# --------------------------------------------------------------------------- #
#  Stage 5 — Heuristic filtering thresholds (paper App. A, Tables 11 & 12)
#
#  All values are "remove the file if <signal> <op> <threshold>" — see
#  filtering/rules.py for the operators. Where the paper gives an exact number
#  (Tables 11/12) it is used verbatim; the rest follow the paper's described
#  intent and the RedPajama/Gopher lineage it cites.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FilterConfig:
    # --- Natural-language rules (applied to 'text' category files) -------------
    nl_min_alpha_frac: float = 0.25       # remove if fraction_alphabetic < 0.25
    nl_max_numeric_frac: float = 0.50     # remove if fraction_numeric   > 0.50
    nl_min_lines: int = 3                 # remove if num_lines          < 3
    nl_max_mean_line_len: float = 200.0   # remove if mean_line_length   > 200
    nl_max_line_len: int = 2000           # remove if max_line_length    > 2000

    # --- General code rules — Table 11 (exact thresholds from the paper) -------
    # A "long string line" = a line whose word count exceeds this bound.
    long_string_word_count: int = 12
    code_max_long_string_line_ratio: float = 0.20   # Table 11: score > 0.2
    code_long_word_char_len: int = 20               # "char count exceeding 20"
    code_max_long_word_char_ratio: float = 0.40     # Table 11: score > 0.4
    code_max_hex_char_ratio: float = 0.40           # Table 11: score > 0.4
    code_max_placeholder_line_ratio: float = 0.01   # Table 11: score > 0.01
    code_max_assert_line_ratio: float = 0.40        # Table 11: score > 0.4
    # Extra general guards (paper describes "number of lines / line length" etc.)
    code_min_lines: int = 3
    code_max_mean_line_len: float = 100.0
    code_max_line_len: int = 1000
    code_min_alpha_frac: float = 0.25

    # --- Python-specific rules — Table 12 (exact thresholds from the paper) -----
    py_max_func_line_ratio: float = 0.20   # Table 12: #funcs / #lines  > 0.2
    py_require_ast_parse: bool = True      # Table 12: drop if not AST-parseable
    py_max_import_line_ratio: float = 0.30 # Table 12: import-line ratio > 0.3


# --------------------------------------------------------------------------- #
#  Stage 6 — High-resource language downsampling (paper Sec. 2.1)
#
#  The paper downsamples e.g. Java 409GB -> 200GB and HTML 213GB -> 64GB. We
#  express this as a per-language *keep fraction*; anything not listed is kept
#  in full. Random subsampling with a fixed seed for reproducibility.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DownsampleConfig:
    seed: int = 42
    keep_fraction: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "Java": 200.0 / 409.0,   # ~0.489  (paper: 409GB -> 200GB)
            "HTML": 64.0 / 213.0,    # ~0.300  (paper: 213GB -> 64GB)
        }
    )


# --------------------------------------------------------------------------- #
#  Web code-data recall (paper Sec. 2.2 / App. C.2)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class WebRecallConfig:
    # StarCoder-style filename recall (App. C.2): keep if 'requirement' in the
    # lowercased name, or the stem is one of these.
    filename_stems: tuple[str, ...] = (
        "readme",
        "notes",
        "todo",
        "description",
        "cmakelists",
    )
    filename_substring: str = "requirement"
    # fastText recall threshold: keep a web doc if P(code) >= this.
    fasttext_threshold: float = 0.5
    # A domain is flagged "code-related" if >= this fraction of its sampled
    # pages are classified as code (paper Sec. 2.2).
    domain_code_ratio: float = 0.10


# --------------------------------------------------------------------------- #
#  Master config — one object threaded through the whole pipeline.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = dataclasses.field(default_factory=PreprocessConfig)
    exact_dedup: ExactDedupConfig = dataclasses.field(default_factory=ExactDedupConfig)
    minhash_dedup: MinHashDedupConfig = dataclasses.field(default_factory=MinHashDedupConfig)
    transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)
    filtering: FilterConfig = dataclasses.field(default_factory=FilterConfig)
    downsample: DownsampleConfig = dataclasses.field(default_factory=DownsampleConfig)
    web_recall: WebRecallConfig = dataclasses.field(default_factory=WebRecallConfig)

    @classmethod
    def fast_demo(cls) -> "PipelineConfig":
        """A lighter config for local/demo runs: a 128-perm MinHash (16x8) instead
        of the paper's 2048-perm (16x128). Semantics identical, just far cheaper."""
        cfg = cls()
        cfg.minhash_dedup = MinHashDedupConfig(num_perm=128, bands=16, rows=8)
        return cfg
