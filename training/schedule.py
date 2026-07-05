"""
Learning-rate schedules.

* WSD (Warmup-Stable-Decay) — the pretraining + annealing schedule from the paper
  (Sec. 3.2, following MiniCPM): linear warmup to the peak LR, hold it constant
  during stable training, then decay exponentially to `end_lr` across the
  annealing tokens. Splitting pretraining vs annealing is just a matter of which
  window you train in: pretraining runs the warmup+stable region, annealing runs
  the decay region (`decay_steps > 0`).

* Cosine — the SFT schedule (Sec. 4.3): short warmup then cosine decay to ~0.

Both return an `optax` schedule (step -> lr) usable directly as the AdamW LR.
"""

from __future__ import annotations

import optax


def wsd_schedule(
    peak_lr: float,
    end_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
) -> optax.Schedule:
    warmup = optax.linear_schedule(0.0, peak_lr, max(warmup_steps, 1))
    stable = optax.constant_schedule(peak_lr)
    if decay_steps > 0:
        # lr(t) = peak * (end/peak)^(t/decay_steps), clamped at end_lr.
        decay = optax.exponential_decay(
            init_value=peak_lr,
            transition_steps=decay_steps,
            decay_rate=max(end_lr / peak_lr, 1e-8),
            end_value=end_lr,
        )
    else:
        decay = optax.constant_schedule(peak_lr)
    return optax.join_schedules(
        [warmup, stable, decay],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )


def cosine_schedule(
    peak_lr: float, warmup_steps: int, total_steps: int, end_lr: float = 0.0
) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=max(warmup_steps, 1),
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=end_lr,
    )
