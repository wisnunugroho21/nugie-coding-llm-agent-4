"""
Training configuration + paper-faithful phase presets (OpenCoder Sec. 3.2 / 4.3).

`TrainConfig` is what the driver consumes. The `*_PRESET` objects record the
paper's real hyperparameters for reference and as defaults you can scale down for
a laptop run. `demo_model_config()` returns a tiny `KimiLinearConfig` (byte-level
vocab) so the full pretrain -> anneal -> SFT flow runs on CPU in seconds.
"""

from __future__ import annotations

import dataclasses

from kimi_linear_gdn2 import KimiLinearConfig


# --------------------------------------------------------------------------- #
#  Paper hyperparameters (1.5B run) — reference values.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class WSDPreset:
    peak_lr: float = 3e-4          # Sec. 3.2
    end_lr: float = 1e-5           # decays to this during annealing
    warmup_steps: int = 2000       # over 8B tokens
    global_batch_size: int = 1024
    weight_decay: float = 0.1
    clip_norm: float = 1.0


@dataclasses.dataclass
class SFTPreset:
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int = 100
    scheduler: str = "cosine"


PRETRAIN_PRESET = WSDPreset()
STAGE1_PRESET = SFTPreset(epochs=1, batch_size=4096, learning_rate=2e-5)   # Table/Sec. 4.3
STAGE2_PRESET = SFTPreset(epochs=3, batch_size=512, learning_rate=5e-5)


# --------------------------------------------------------------------------- #
#  Runnable defaults.
# --------------------------------------------------------------------------- #
def demo_model_config() -> KimiLinearConfig:
    """Tiny Kimi-Linear/GDN-2 that trains on CPU (byte-level vocab 256)."""
    return KimiLinearConfig(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        full_attn_period=4,        # -> layer 3 is MLA, rest GDN-2 (3:1)
        gdn_chunk_size=32,
        max_seq_len=256,
        moe_n_routed=8,
        moe_top_k=2,
        moe_d_ff=256,
    )


@dataclasses.dataclass
class TrainConfig:
    phase: str                     # "pretrain" | "anneal" | "sft"
    data_paths: list[str]
    model: KimiLinearConfig = dataclasses.field(default_factory=demo_model_config)

    seq_len: int = 128             # must be a multiple of model.gdn_chunk_size
    batch_size: int = 8            # per-step global batch fed to fit (before accumulation)
    steps: int = 50
    grad_accum: int = 1            # microbatches per optimizer step (optax.MultiSteps)
    data_parallel: bool = False    # replicate params, shard batch across local devices (nnx.pmap)

    # Optimizer.
    weight_decay: float = 0.1
    b1: float = 0.9
    b2: float = 0.95
    clip_norm: float = 1.0
    bias_lr: float = 1e-3          # router-bias nudge (aux-loss-free balancing)

    # Schedule (WSD for pretrain/anneal, cosine for sft).
    peak_lr: float = 3e-4
    end_lr: float = 1e-5
    warmup_steps: int = 20
    stable_steps: int = 0          # pretrain fills this; anneal sets decay_steps
    decay_steps: int = 0           # >0 for annealing (and cosine total for sft)

    # IO.
    init_from: str | None = None   # checkpoint to warm-start from
    save_to: str | None = None
    seed: int = 0
    data_field: str = "content"    # pretrain/anneal doc text field
    tokenizer_path: str | None = None  # BPE tokenizer.json; None -> byte tokenizer
    system: str = "You are a helpful programming assistant."
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.seq_len % self.model.gdn_chunk_size != 0:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be a multiple of the model's "
                f"gdn_chunk_size ({self.model.gdn_chunk_size})."
            )
        if self.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) exceeds model.max_seq_len ({self.model.max_seq_len})."
            )
