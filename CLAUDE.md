# Project Context
- **Domain:** Epidemic control via NPIs, SEIR model with stochastic Binomial transitions.
- **RL Framework:** Stable Baselines 3 (PPO, RecurrentPPO), Gymnasium.
- **Research question:** How do memory mechanisms (FrameStack vs LSTM) compare under increasing partial observability (MDP → POMDP)?
- **Key boundary:** `src/config.py` owns SEIR/reward/RL hyperparameters. POMDP observation parameters (`include_exposed`, `detection_rate`, `noise_stds`, `lag`) belong in `src/scenarios.py` — never in `Config`.
- **LR schedule invariant:** `lr_decay_steps` controls LR decay horizon (steps to reach 10% of peak), independent of `total_timesteps`. Changing `total_timesteps` must NOT change LR at any given training step. See `docs/17`.

# Rules
- **No backward compatibility.** When changing behavior, replace the old code path entirely. No `if-else` fallbacks preserving old logic.
- **Never delete `experiments/` data.** Timestamped results are not reproducible cheaply (~25 min per full run).
- **Use venv Python directly:** `venv/Scripts/python.exe` (not `source venv/Scripts/activate`).
- **Smoke tests:** use `--scenario <name> -t 1000 --num-seeds 2`. Scenario folder includes timesteps (`mdp_t1000`), so no collision with real experiment weights.
- **Before committing**, check whether `README.md` or `CLAUDE.md` need updating. Include doc changes in the same commit.
- **Google-style docstrings** for classes and major functions. Document input/output shapes for tensors/arrays.
- **NumPy vectorization** over `for` loops.
- **Every stochastic code path** must accept a `seed` argument. No unseeded randomness.

# Experiment Data
- Results live in `experiments/{scenario_name}/{YYYY-MM-DD_HH-MM-SS}/`.
- Each run produces `evaluation.json` (raw per-episode metrics by seed) and `summary.json` (cross-seed mean ± SE).
- `analyses.json` (project root) maps analysis names → experiment paths. Manually maintained.
- `analysis/data.py`: `load_analysis(name)` → `dict[str, AnalysisRun]` with config + summary + evaluation.
- Analysis scripts: `python -m analysis.pomdp_gap`, `python -m analysis.significance_tests`, `python -m analysis.framestack_ablation`, `python -m analysis.distortion_ablation`, `python -m analysis.reward_grid <grid_name>` (reward-only 2D table across scenarios × agents; grid config in `analyses.json["reward_grid"]`).
- **Training diagnostics:** `python -m analysis.training_summary <experiment_path>` — per-seed best eval reward, best step, final step, early stopping status. Reads `evaluations.npz` directly (no manifest needed). Example: `python -m analysis.training_summary pomdp_t3000000/default`.
- **Publication figures:** `analysis/` plot scripts route styling through `analysis/style.py` — call `apply_style()`, save via `save_figure()` (PDF), use `FIG_WIDTH`; don't hardcode figsize/fontsize/savefig. (`seir` is the exception: its own larger `SEIR_*` scale, no `apply_style()`.)

# Invariants: Observation Space
- Base obs shape `(6,)`: `[S, E, I, R, prev_action_idx, day_frac]`.
- With `include_exposed=False`: shape `(5,)` → `[S+E, I, R, prev_action_idx, day_frac]`. E is folded into S (not dropped) so `S_obs + I_obs + R_obs = N` and the agent cannot recover E via SEIR conservation.
- Obs bounds are **per-element**: high = `[N, N, N, N, 3.0, 1.0]` (6-el) or `[N, N, N, 3.0, 1.0]` (5-el). `S+E ≤ N` always, so `high[0] = N` stays valid under folding.
- `MultiplicativeNoiseWrapper`: `len(noise_stds) == obs_size - 2` (compartments only). Trailing `prev_action_idx` and `day_frac` pass through unchanged.
- Wrapper chain: `EpidemicObservationWrapper → UnderReportingWrapper → MultiplicativeNoiseWrapper → TemporalLagWrapper`.

# Invariants: Temporal Resolution
- **1 step = `action_interval` days** (default 5). Episode = `days / action_interval` steps (default 60).
- **FrameStack `n_stack`** is in **steps**, not days. `n_stack=30` = 150 days of history.
- **`TemporalLagWrapper`**: steps internally, but `create_environment()` accepts days and converts:
  ```python
  min_lag_steps = max(1, round(min_lag_days / action_interval))
  max_lag_steps = max(min_lag_steps, round(max_lag_days / action_interval))
  ```
  Always specify lag in **days** in scenarios. Never pass raw step counts.
