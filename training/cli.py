"""
Training CLI — one subcommand per phase.

    # 1. pretrain on the cleaned code corpus (WSD warmup+stable)
    python -m training.cli pretrain --data sample_data/refined.jsonl \
        --steps 200 --save ckpt/pretrain.pkl

    # 2. anneal from the pretrained weights (WSD decay -> end-lr)
    python -m training.cli anneal --data sample_data/annealing.jsonl \
        --init ckpt/pretrain.pkl --steps 100 --save ckpt/anneal.pkl

    # 3a. SFT stage 1 (diverse/theory), then 3b. stage 2 (code-specific)
    python -m training.cli sft --stage 1 --data sample_data/sft_stage1.jsonl \
        --init ckpt/anneal.pkl --steps 100 --save ckpt/sft1.pkl
    python -m training.cli sft --stage 2 --data sample_data/sft_stage2.jsonl \
        --init ckpt/sft1.pkl --steps 100 --save ckpt/sft2.pkl

Defaults are tiny (byte-level model) so this runs on CPU; scale up the model
config and steps for a real run. Paper hyperparameters live in training/config.py.
"""

from __future__ import annotations

import argparse

from .config import STAGE1_PRESET, STAGE2_PRESET, TrainConfig
from . import run_phase


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", nargs="+", required=True, help="JSONL path(s)")
    ap.add_argument("--init", default=None, help="warm-start checkpoint")
    ap.add_argument("--save", default=None, help="output checkpoint path")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)


def _cmd_pretrain(a: argparse.Namespace) -> int:
    warmup = min(a.warmup, max(a.steps // 10, 1))
    cfg = TrainConfig(
        phase="pretrain", data_paths=a.data, steps=a.steps, batch_size=a.batch_size,
        seq_len=a.seq_len, init_from=a.init, save_to=a.save, seed=a.seed,
        peak_lr=a.peak_lr, warmup_steps=warmup, stable_steps=max(a.steps - warmup, 0),
        decay_steps=0, log_every=a.log_every,
    )
    run_phase(cfg)
    return 0


def _cmd_anneal(a: argparse.Namespace) -> int:
    # WSD decay region: no warmup/stable, decay peak_lr -> end_lr over all steps.
    cfg = TrainConfig(
        phase="anneal", data_paths=a.data, steps=a.steps, batch_size=a.batch_size,
        seq_len=a.seq_len, init_from=a.init, save_to=a.save, seed=a.seed,
        peak_lr=a.peak_lr, end_lr=a.end_lr, warmup_steps=0, stable_steps=0,
        decay_steps=a.steps, log_every=a.log_every,
    )
    run_phase(cfg)
    return 0


def _cmd_sft(a: argparse.Namespace) -> int:
    preset = STAGE1_PRESET if a.stage == 1 else STAGE2_PRESET
    lr = a.lr if a.lr is not None else preset.learning_rate
    cfg = TrainConfig(
        phase="sft", data_paths=a.data, steps=a.steps, batch_size=a.batch_size,
        seq_len=a.seq_len, init_from=a.init, save_to=a.save, seed=a.seed,
        peak_lr=lr, warmup_steps=min(preset.warmup_steps, max(a.steps // 10, 1)),
        decay_steps=a.steps, log_every=a.log_every,
    )
    print(f"[sft] stage {a.stage}: lr={lr} (paper: {preset.epochs} epochs, "
          f"batch {preset.batch_size})")
    run_phase(cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="training", description="Kimi-Linear/GDN-2 training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain", help="WSD warmup+stable on the code corpus")
    _common(p)
    p.add_argument("--peak-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.set_defaults(func=_cmd_pretrain)

    an = sub.add_parser("anneal", help="WSD decay on the annealing mixture")
    _common(an)
    an.add_argument("--peak-lr", type=float, default=3e-4)
    an.add_argument("--end-lr", type=float, default=1e-5)
    an.set_defaults(func=_cmd_anneal)

    s = sub.add_parser("sft", help="two-stage instruction tuning (cosine)")
    _common(s)
    s.add_argument("--stage", type=int, choices=[1, 2], default=1)
    s.add_argument("--lr", type=float, default=None, help="override the stage's paper LR")
    s.set_defaults(func=_cmd_sft)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
