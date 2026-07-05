"""
Configuration for the annealing-data stage (OpenCoder Sec. 2.3, Table 3).

Annealing is a short, high-quality *end-of-pretraining* phase: ~100B tokens whose
mixture is dominated (~84%) by the original RefineCode distribution to avoid
catastrophic forgetting, sprinkled with higher-signal data (algorithmic corpus +
LLM-synthesized snippets + code textbooks). Under the WSD schedule the peak LR is
held during stable training then decayed to 1e-5 across exactly this data.
"""

from __future__ import annotations

import dataclasses

# Table 3 — annealing data mixture, in *billions* of tokens.
TABLE3_TOKENS_B: dict[str, float] = {
    "refinecode": 83.94,          # Original Data
    "algorithmic": 12.44,         # Algorithmic Corpus (keyword-sampled)
    "synthetic_snippet": 2.71,    # Synthetic Data: High Quality Code Snippet
    "code_textbook": 0.91,        # Synthetic Data: Code Textbooks
}
# -> total 100.00B; original fraction 83.94 / 100 == 0.8394.


@dataclasses.dataclass
class AnnealingMixConfig:
    # Target mixture as token *proportions* (normalized from Table 3). Scale-free:
    # the mixer hits these ratios up to a total token budget you choose.
    proportions: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            k: v / sum(TABLE3_TOKENS_B.values()) for k, v in TABLE3_TOKENS_B.items()
        }
    )
    seed: int = 42


@dataclasses.dataclass
class AlgorithmicConfig:
    # Paper: sampled with algorithm/competitive-programming keywords ("leetcode",
    # "solution", ...). Case-insensitive substring match over path + content.
    keywords: tuple[str, ...] = (
        "leetcode", "solution", "competitive programming", "codeforces",
        "dynamic programming", "binary search", "algorithm", "dijkstra",
        "backtracking", "greedy", "time complexity", "def solve",
    )
    min_keyword_hits: int = 1


@dataclasses.dataclass
class SyntheticConfig:
    # High-quality snippet synthesis with test validation (kept only if tests pass).
    tests_per_item: int = 1
    exec_timeout_s: float = 10.0
    teacher_temperature: float = 0.7


@dataclasses.dataclass
class TextbookConfig:
    # Educational rewrite of high-quality snippets ("Code Textbooks", hqcode-style).
    teacher_temperature: float = 0.7
    # Only turn sufficiently substantial snippets into textbook passages.
    min_snippet_chars: int = 120


@dataclasses.dataclass
class WSDScheduleConfig:
    """Warmup-Stable-Decay schedule params for the annealing phase (Sec. 3.2)."""
    annealing_tokens_b: float = 100.0     # additional tokens in the decay phase
    warmup_steps: int = 2000
    warmup_tokens_b: float = 8.0
    peak_lr: float = 3e-4
    end_lr: float = 1e-5                  # decayed to this over annealing
    micro_batch_size: int = 4
    global_batch_size: int = 1024


@dataclasses.dataclass
class AnnealingConfig:
    mix: AnnealingMixConfig = dataclasses.field(default_factory=AnnealingMixConfig)
    algorithmic: AlgorithmicConfig = dataclasses.field(default_factory=AlgorithmicConfig)
    synthetic: SyntheticConfig = dataclasses.field(default_factory=SyntheticConfig)
    textbook: TextbookConfig = dataclasses.field(default_factory=TextbookConfig)
    schedule: WSDScheduleConfig = dataclasses.field(default_factory=WSDScheduleConfig)
