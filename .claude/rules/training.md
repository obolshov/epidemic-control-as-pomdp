---
globs: ["src/train.py", "src/evaluation.py"]
---

# SB3 Pipeline Invariants

When modifying training or evaluation code, ALL of the following must hold:

1. **VecEnv stack order:** `DummyVecEnv → VecMonitor → VecNormalize → [VecFrameStack]`
   - Train and eval env MUST have identical wrapper structure (`sync_envs_normalization` walks both in parallel; mismatch → AssertionError).
   - `VecMonitor` must come **before** `VecNormalize`.

2. **Monitor dir must be per-seed:** `tensorboard_dir / f"{agent_name}_seed{seed}"`
   - Shared dir corrupts per-seed learning curves.

3. **`EvalCallback` eval_freq must account for n_envs:**
   ```python
   adjusted_eval_freq = max(1, eval_freq // n_envs)
   ```
   Counts per-env steps, not total timesteps.

4. **VecNormalize lifecycle:**
   - Training: `norm_obs=True, norm_reward=True`.
   - Save: `env.save(vecnormalize_path)`.
   - Eval: `VecNormalize.load(path, venv)` then `training=False, norm_reward=False`.

5. **After training, load the best checkpoint** (from `EvalCallback`), not final model state.

6. **Do NOT remove `_save_episode_logs()`** — saves per-episode action logs to `logs/{agent_name}/`.
