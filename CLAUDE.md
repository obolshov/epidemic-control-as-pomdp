# Project Context
- **Domain:** Epidemic control via NPIs, SEIR model with stochastic Binomial transitions.
- **RL Framework:** Stable Baselines 3 (PPO, RecurrentPPO), Gymnasium.
- **Research question:** How do memory mechanisms (FrameStack vs LSTM) compare under increasing partial observability (MDP → POMDP)?
- **Key boundary:** `src/config.py` owns SEIR/reward/RL hyperparameters. POMDP observation parameters (`include_exposed`, `detection_rate`, `noise_stds`, `lag`) belong in `src/scenarios.py` — never in `Config`.

# Rules
- **No backward compatibility.** When changing behavior, replace the old code path entirely. No `if-else` fallbacks preserving old logic.
- **Never delete `experiments/` data.** Timestamped results are not reproducible cheaply (~25 min per full run).
- **Always activate venv** (`source venv/Scripts/activate`) before running Python.
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
- Analysis scripts: `python -m analysis.pomdp_gap`, `python -m analysis.significance_tests`, `python -m analysis.framestack_ablation`, `python -m analysis.distortion_ablation`.
- **Resume training:** `--resume-from <scenario_folder>` loads weights from `experiments/{folder}/weights/` and continues training. Agents without matching weights train from scratch. New weights go to the new scenario folder. `config.json` records `resumed_from` for provenance.

# Invariants: Observation Space
- Base obs shape `(6,)`: `[S, E, I, R, prev_action_idx, day_frac]`.
- With `include_exposed=False`: shape `(5,)` → `[S, I, R, prev_action_idx, day_frac]`.
- Obs bounds are **per-element**: high = `[N, N, N, N, 3.0, 1.0]` (6-el) or `[N, N, N, 3.0, 1.0]` (5-el).
- `MultiplicativeNoiseWrapper`: `len(noise_stds) == obs_size - 2` (compartments only). Trailing `prev_action_idx` and `day_frac` pass through unchanged.
- Wrapper chain: `EpidemicObservationWrapper → UnderReportingWrapper → TemporalLagWrapper → MultiplicativeNoiseWrapper`.

# Invariants: Temporal Resolution
- **1 step = `action_interval` days** (default 5). Episode = `days / action_interval` steps (default 60).
- **FrameStack `n_stack`** is in **steps**, not days. `n_stack=20` = 100 days of history.
- **`TemporalLagWrapper`**: steps internally, but `create_environment()` accepts days and converts:
  ```python
  min_lag_steps = max(1, round(min_lag_days / action_interval))
  max_lag_steps = max(min_lag_steps, round(max_lag_days / action_interval))
  ```
  Always specify lag in **days** in scenarios. Never pass raw step counts.
- **`action_delay`**: days in scenario params, converted to steps by `create_environment()`. Env uses FIFO queue; `prev_action_idx` reflects the **applied** (delayed) action.
