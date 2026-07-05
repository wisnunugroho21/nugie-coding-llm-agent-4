"""
Full per-device training configs.

Four presets sized to real hardware, each giving a model config (dims + compute
dtype) and the runtime knobs (seq_len, per-device batch, gradient accumulation,
data parallelism) that make it fit and train efficiently on that device:

  * "m1"   — Apple M1 CPU: tiny, fp32, a minutes-long smoke test.
  * "t4"   — one Nvidia T4 16GB (Colab): bf16, ~55M params, grad accumulation.
  * "t4x2" — two T4s (Kaggle): the T4 model, data-parallel across both GPUs.
  * "h200" — one Nvidia H200 141GB: bf16, ~1.3B-param MoE, long context.

All use the byte-level tokenizer (vocab 256) so they run with the code as-is; for
a real run raise `vocab_size` to your tokenizer's size (e.g. OpenCoder's 96,640).
The phase LR/schedule/steps come from the paper defaults (config.py / the CLI);
a device preset only fixes *what fits* — model size, dtype, batch, parallelism.

`build_config(device, phase, ...)` returns a ready `TrainConfig`.
"""

from __future__ import annotations

import dataclasses

from kimi_linear_gdn2 import KimiLinearConfig

from .config import STAGE1_PRESET, STAGE2_PRESET, TrainConfig


@dataclasses.dataclass
class DevicePreset:
    name: str
    model: KimiLinearConfig
    seq_len: int
    per_device_batch: int
    grad_accum: int
    data_parallel: bool
    # Suggested steps per phase (short, so a demo finishes; scale up for real runs).
    steps: dict[str, int]


def _m1() -> DevicePreset:
    return DevicePreset(
        name="m1",
        model=KimiLinearConfig(
            vocab_size=256, d_model=256, n_layers=4, full_attn_period=4,
            gdn_chunk_size=64, max_seq_len=512,
            moe_n_routed=8, moe_top_k=2, moe_d_ff=256,
            compute_dtype="float32",
        ),
        seq_len=256, per_device_batch=2, grad_accum=1, data_parallel=False,
        steps={"pretrain": 30, "anneal": 15, "sft": 15},
    )


def _t4() -> DevicePreset:
    # ~55M params. bf16 compute, fp32 master weights + Adam state fit 16GB with room.
    model = KimiLinearConfig(
        vocab_size=256, d_model=512, n_layers=8, full_attn_period=4,
        gdn_num_heads=8, gdn_head_k_dim=64, gdn_head_v_dim=64,
        gdn_chunk_size=64, max_seq_len=1024,
        mla_num_q_heads=8, mla_num_kv_heads=2, mla_head_dim=64,
        moe_n_routed=8, moe_top_k=2, moe_n_shared=1, moe_d_ff=512,
        compute_dtype="bfloat16",
    )
    return DevicePreset(
        name="t4", model=model,
        seq_len=1024, per_device_batch=4, grad_accum=8, data_parallel=False,
        steps={"pretrain": 2000, "anneal": 500, "sft": 500},
    )


def _t4x2() -> DevicePreset:
    p = _t4()
    # Same model, data-parallel across the two Kaggle T4s. per_device_batch stays
    # 4; build_config multiplies batch_size by the device count at build time.
    return DevicePreset(
        name="t4x2", model=p.model,
        seq_len=1024, per_device_batch=4, grad_accum=8, data_parallel=True,
        steps=p.steps,
    )


def _h200() -> DevicePreset:
    # ~1.3B-param MoE (top-4 of 16 experts active). bf16 compute; long 4096 context.
    model = KimiLinearConfig(
        vocab_size=256, d_model=1536, n_layers=16, full_attn_period=4,
        gdn_num_heads=12, gdn_head_k_dim=128, gdn_head_v_dim=128,
        gdn_chunk_size=64, max_seq_len=4096,
        mla_num_q_heads=16, mla_num_kv_heads=2, mla_head_dim=128,
        moe_n_routed=16, moe_top_k=4, moe_n_shared=1, moe_d_ff=1024,
        compute_dtype="bfloat16",
    )
    return DevicePreset(
        name="h200", model=model,
        seq_len=4096, per_device_batch=8, grad_accum=4, data_parallel=False,
        steps={"pretrain": 20000, "anneal": 2000, "sft": 1000},
    )


DEVICES = {"m1": _m1, "t4": _t4, "t4x2": _t4x2, "h200": _h200}


def get_preset(device: str) -> DevicePreset:
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; choose from {sorted(DEVICES)}")
    return DEVICES[device]()


def build_config(
    device: str,
    phase: str,
    data_paths: list[str],
    *,
    init_from: str | None = None,
    save_to: str | None = None,
    steps: int | None = None,
    stage: int = 1,
    log_every: int = 20,
    tokenizer_path: str | None = None,
) -> TrainConfig:
    """Assemble a full TrainConfig for a device + phase (paper LR/schedule)."""
    p = get_preset(device)
    n_steps = steps if steps is not None else p.steps[phase]

    # Data parallelism multiplies the per-step batch by the local device count.
    batch_size = p.per_device_batch
    if p.data_parallel:
        import jax

        batch_size = p.per_device_batch * max(jax.local_device_count(), 1)

    # Per-phase schedule (mirrors training/cli.py).
    if phase == "pretrain":
        warmup = max(min(2000, n_steps // 10), 1)
        sched = dict(peak_lr=3e-4, warmup_steps=warmup,
                     stable_steps=max(n_steps - warmup, 0), decay_steps=0)
    elif phase == "anneal":
        sched = dict(peak_lr=3e-4, end_lr=1e-5, warmup_steps=0, stable_steps=0,
                     decay_steps=n_steps)
    elif phase == "sft":
        preset = STAGE1_PRESET if stage == 1 else STAGE2_PRESET
        sched = dict(peak_lr=preset.learning_rate,
                     warmup_steps=max(min(preset.warmup_steps, n_steps // 10), 1),
                     decay_steps=n_steps)
    else:
        raise ValueError(f"unknown phase {phase!r}")

    return TrainConfig(
        phase=phase, data_paths=data_paths, model=p.model,
        seq_len=p.seq_len, batch_size=batch_size, steps=n_steps,
        grad_accum=p.grad_accum, data_parallel=p.data_parallel,
        init_from=init_from, save_to=save_to, log_every=log_every,
        tokenizer_path=tokenizer_path,
        **sched,
    )


def _describe() -> None:
    """`python -m training.devices` — print each device config + param count."""
    import flax.nnx as nnx

    from kimi_linear_gdn2 import KimiLinear, count_params

    for name in ("m1", "t4", "t4x2", "h200"):
        p = get_preset(name)
        m = p.model
        eff = p.per_device_batch * p.grad_accum * (2 if p.data_parallel else 1)
        line = (f"{name:5s} | d_model={m.d_model} layers={m.n_layers} "
                f"experts={m.moe_n_routed} top_k={m.moe_top_k} d_ff={m.moe_d_ff} | "
                f"dtype={m.compute_dtype} seq={p.seq_len} "
                f"per_dev_batch={p.per_device_batch} accum={p.grad_accum} "
                f"dp={p.data_parallel} | ~global_batch={eff}")
        # Building the ~1.3B h200 model on CPU is heavy; only count the small ones.
        if name in ("m1", "t4"):
            n = count_params(KimiLinear(m, rngs=nnx.Rngs(0)))
            line += f" | params={n/1e6:.1f}M"
        print(line)


if __name__ == "__main__":
    _describe()
