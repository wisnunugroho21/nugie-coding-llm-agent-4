"""
Tests for the per-device training presets, gradient accumulation, and the
data-parallel path (verified on 2 fake CPU devices via a subprocess).
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from training.config import TrainConfig
from training.devices import DEVICES, build_config, get_preset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPresets(unittest.TestCase):
    def test_all_presets_build_valid_configs(self):
        for device in DEVICES:
            for phase in ("pretrain", "anneal", "sft"):
                cfg = build_config(device, phase, ["data.jsonl"], steps=10)
                self.assertIsInstance(cfg, TrainConfig)
                # TrainConfig.__post_init__ enforces seq_len % chunk == 0 and <= max.
                self.assertEqual(cfg.seq_len % cfg.model.gdn_chunk_size, 0)
                self.assertLessEqual(cfg.seq_len, cfg.model.max_seq_len)

    def test_dtypes_and_parallelism(self):
        self.assertEqual(get_preset("m1").model.compute_dtype, "float32")
        self.assertEqual(get_preset("t4").model.compute_dtype, "bfloat16")
        self.assertEqual(get_preset("h200").model.compute_dtype, "bfloat16")
        self.assertFalse(get_preset("t4").data_parallel)
        self.assertTrue(get_preset("t4x2").data_parallel)

    def test_sft_stage_lr(self):
        s1 = build_config("t4", "sft", ["d"], steps=10, stage=1)
        s2 = build_config("t4", "sft", ["d"], steps=10, stage=2)
        self.assertAlmostEqual(s1.peak_lr, 2e-5)     # Sec. 4.3 stage 1
        self.assertAlmostEqual(s2.peak_lr, 5e-5)     # Sec. 4.3 stage 2

    def test_mla_head_divisibility(self):
        for device in DEVICES:
            m = get_preset(device).model
            self.assertEqual(m.mla_num_q_heads % m.mla_num_kv_heads, 0)

    def test_unknown_device_raises(self):
        with self.assertRaises(ValueError):
            get_preset("v100")


class TestGradAccum(unittest.TestCase):
    def test_multisteps_run(self):
        from training import run_phase
        from training.config import demo_model_config

        cfg = TrainConfig(phase="pretrain", data_paths=[os.path.join(REPO, "sample_data/refined.jsonl")],
                          model=demo_model_config(), seq_len=64, batch_size=2, steps=6,
                          warmup_steps=2, stable_steps=4, grad_accum=2, log_every=100)
        run_phase(cfg)   # runs without error under optax.MultiSteps


class TestDataParallel(unittest.TestCase):
    def test_dp_runs_on_two_fake_devices(self):
        script = (
            "import jax; assert jax.local_device_count()==2, jax.local_device_count();"
            "from training import run_phase;"
            "from training.config import TrainConfig, demo_model_config;"
            "cfg=TrainConfig(phase='pretrain', data_paths=['sample_data/refined.jsonl'],"
            " model=demo_model_config(), seq_len=64, batch_size=4, steps=4,"
            " warmup_steps=1, stable_steps=3, data_parallel=True, grad_accum=2, log_every=100);"
            "run_phase(cfg); print('DP_RAN_OK')"
        )
        env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2",
                   JAX_PLATFORMS="cpu")
        proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=300)
        self.assertIn("DP_RAN_OK", proc.stdout, msg=proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
