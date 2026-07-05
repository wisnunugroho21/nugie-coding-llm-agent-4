"""
Configuration for OpenCoder post-training / two-stage instruction tuning
(Sec. 4, Table 5 + Sec. 4.3 hyperparameters + Sec. 4.4 decontamination).
"""

from __future__ import annotations

import dataclasses

# Table 5 — target example counts per source, split across the two stages.
STAGE1_COUNTS: dict[str, int] = {
    "realuser_instruct": 700_000,          # 0.7 M
    "diverse_instruct": 2_300_000,         # 2.3 M
    "filtered_infinity_instruct": 1_000_000,  # 1.0 M
}
STAGE2_COUNTS: dict[str, int] = {
    "mceval_instruct": 36_000,             # 36 K
    "evol_instruct": 111_000,              # 111 K
    "educational_instruct": 110_000,       # 110 K
    "package_instruct": 110_000,           # 110 K
}


@dataclasses.dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int = 100
    scheduler: str = "cosine"


@dataclasses.dataclass
class DecontamConfig:
    # Sec. 4.4: 10-gram overlap removal + drop anything containing a test-set
    # entry point (function name) from HumanEval / MBPP.
    ngram: int = 10
    remove_entry_point_matches: bool = True


@dataclasses.dataclass
class DiverseSynthConfig:
    # Sec. 4.1 "Large-scale Diverse Instruction Synthesis".
    languages: tuple[str, ...] = ("Python", "JavaScript", "Java", "C++", "Go")
    difficulties: tuple[str, ...] = ("easy", "medium", "hard")
    task_types: tuple[str, ...] = (
        "function implementation", "bug fix", "algorithm", "data processing",
        "API usage",
    )
    question_temperature: float = 1.0     # paper: T = 1.0 for diverse questions
    validate_with_tests: bool = True
    exec_timeout_s: float = 10.0


@dataclasses.dataclass
class EducationalSynthConfig:
    # Sec. 4.1 "Educational Instruction Synthesis": score seeds, keep high-quality,
    # generate QA + tests, keep only test-passing samples.
    min_seed_score: float = 0.5
    exec_timeout_s: float = 10.0


@dataclasses.dataclass
class PackageSynthConfig:
    # Sec. 4.1 "Package-related Instruction Synthesis": up-to-date API QA via PyDoc.
    libraries: tuple[str, ...] = ("math", "json", "itertools", "collections", "statistics")
    max_apis_per_library: int = 8


@dataclasses.dataclass
class SFTConfig:
    stage1: TrainingConfig = dataclasses.field(
        default_factory=lambda: TrainingConfig(epochs=1, batch_size=4096, learning_rate=2e-5)
    )
    stage2: TrainingConfig = dataclasses.field(
        default_factory=lambda: TrainingConfig(epochs=3, batch_size=512, learning_rate=5e-5)
    )
    decontam: DecontamConfig = dataclasses.field(default_factory=DecontamConfig)
    diverse: DiverseSynthConfig = dataclasses.field(default_factory=DiverseSynthConfig)
    educational: EducationalSynthConfig = dataclasses.field(default_factory=EducationalSynthConfig)
    package: PackageSynthConfig = dataclasses.field(default_factory=PackageSynthConfig)
    seed: int = 42
