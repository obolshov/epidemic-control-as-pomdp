# Project Context
- **Domain:** Epidemic control via NPIs (Non-Pharmaceutical Interventions).
- **Core Model:** SEIR (Susceptible-Exposed-Infected-Recovered) with stochastic Binomial transitions.
- **RL Framework:** Stable Baselines 3 (SB3), Gymnasium.
- **Main idea:** Performance comparison of RL algorithms with and without memory (FrameStack, Recurrent) in MDP and POMDP scenarios.
- **Architecture:**
    - `src/seir.py`: SEIR dynamics. Stochastic Binomial mode (`rng=np.random.Generator`) is the default used by the environment. Deterministic mode (`rng=None`) is also available.
    - `src/env.py`: Gymnasium environment (`EpidemicEnv`) and `InterventionAction` enum (the domain action type).
    - `src/agents.py`: Agent wrappers and baseline logic. Imports `InterventionAction` from `env.py`.
    - `src/results.py`: Data containers — `SimulationResult` (single-episode trajectory) and `AggregatedResult` (multi-episode mean ± SD).
    - `src/wrappers.py`: `ObservationWrapper` subclasses for POMDP distortions. Also contains `create_environment()` factory used by `train.py`. Wrapper chain order: `EpidemicObservationWrapper → UnderReportingWrapper → TemporalLagWrapper → MultiplicativeNoiseWrapper`.
    - `src/train.py`: Training pipeline. Builds `DummyVecEnv → VecMonitor → VecNormalize → [VecFrameStack]`, configures `EvalCallback` + `StopTrainingOnNoModelImprovement`, and trains PPO / RecurrentPPO with per-seed weight saving.
    - `src/evaluation.py`: Post-training evaluation. `evaluate_agent()` runs multi-episode evaluation (mean ± SD) for ANY agent type on fixed eval seeds; `select_best_model()` selects the best training seed via reward-only eval; `run_evaluation()` is the unified pipeline for baselines and RL agents.
    - `src/experiment.py`: `ExperimentConfig` dataclass — manages paths, seeds, and per-seed weight/VecNormalize file locations.
    - `src/scenarios.py`: Predefined scenario registry (`PREDEFINED_SCENARIOS`) and `create_custom_scenario_name()`.
    - `main.py`: Entry point using `typer`.

# Coding Standards & Style
- **Python:** Use Python 3.10+. Enforce strict type hinting (`from typing import ...`).
- **Style:** Follow PEP 8. Use concise, descriptive variable names (e.g., `infected_count` not `i`, `lockdown_level` not `l`).
- **Docstrings:** Use Google-style docstrings for all classes and major functions. Focus on input/output shapes for tensors/arrays.
- **Vectorization:** PREFER NumPy vectorization over `for` loops. The SEIR simulation is the bottleneck.

# RL & Scientific Rigor
1.  **Reproducibility:** ALWAYS handle random seeds (`seed`) in environments and agents. Never write stochastic code without a seed argument.
2.  **Gymnasium:** Use `gymnasium` (not `gym`). Ensure `observation_space` and `action_space` are strictly defined.
3.  **POMDP Awareness:**
    - When modifying the environment, assume partial observability is the goal.
    - Use `ObservationWrapper` for modifying inputs (masking Exposed, adding noise).
    - Differentiate clearly between "Ground Truth" (state) and "Observation" (obs).
4.  **Evaluation:**
    - Evaluation metrics must include Mean Reward AND Standard Deviation over multiple episodes (5-10 seeds).
    - "Oracle" (MDP) is the upper bound baseline. All new agents must be compared against it.

# Specific Libraries
- `stable_baselines3`: Use `PPO` and `VecFrameStack`.
- `sb3_contrib`: Use `RecurrentPPO` for stateful agents.
- `numpy`: Use for all math operations.
- `typer`: Use for CLI commands in `main.py`.

# Plotting & Visualization
- Always visualize the *Standard Deviation* (shaded area) when plotting RL training curves or evaluation results.
- Label axes clearly with units (e.g., "Days", "Infected Population").

# Workflow constraints
- If suggesting a major architectural change (e.g., switching from FrameStack to RNN), explain the *scientific* motivation first.
- `src/config.py` (`@dataclass Config`) is the single source of truth for SEIR model, reward, and RL hyperparameters. **Do NOT add POMDP observation parameters** (e.g. `include_exposed`, `detection_rate`) to `Config` — those belong exclusively in `PREDEFINED_SCENARIOS` (src/scenarios.py) and CLI arguments.
- **After completing any non-trivial change**, always check:
  - `README.md`: is the output structure, CLI options, or agent descriptions still accurate?
  - `CLAUDE.md`: does the Architecture section still describe the actual module responsibilities? Are any invariants stale?

# Observation Space Invariants

- Base `EpidemicEnv._get_obs()` returns shape `(6,)`: `[S, E, I, R, prev_action_idx, day_frac]`.
- After `EpidemicObservationWrapper(include_exposed=False)`: shape `(5,)` → `[S, I, R, prev_action_idx, day_frac]`.
- Observation space bounds are **per-element** (not uniform): high = `[N, N, N, N, 3.0, 1.0]` (6-element) or `[N, N, N, 3.0, 1.0]` (5-element).
- `MultiplicativeNoiseWrapper` expects `len(noise_stds) == obs_size - 2` (compartments only). Trailing `prev_action_idx` and `day_frac` **pass through unchanged** and must NOT be included in `noise_stds`.
- `MultiplicativeNoiseWrapper` supports AR(1) autocorrelated noise via `noise_rho` ∈ [0, 1). `noise_rho=0` → iid (backward compatible). `noise_rho=0.7` → persistent measurement bias (decorrelation half-life ≈ 2 steps). The `√(1 - ρ²)` innovation scaling preserves marginal variance = σ² regardless of ρ.

# Temporal Resolution Invariants

- **1 step = `action_interval` days** (default 5). One episode = `days / action_interval` steps (default 300/5 = 60 steps).
- **FrameStack `n_stack`** is in **steps**, not days. `n_stack=10` → agent sees the last 10 decision points (50 days of history). This is the intended scope — calibrated to decision periods, not calendar days.
- **`TemporalLagWrapper` min/max lag** is in **steps** internally. However, `create_environment()` accepts `pomdp_params["lag"] = [min_days, max_days]` and converts to steps automatically:
  ```python
  min_lag_steps = max(1, round(min_lag_days / action_interval))   # e.g. 5/5 = 1
  max_lag_steps = max(min_lag_steps, round(max_lag_days / action_interval))  # e.g. 14/5 ≈ 3
  ```
  Always specify lag in **days** in `PREDEFINED_SCENARIOS` and CLI (`--lag`). Never pass raw step counts.
- **`action_delay`** is in **days** in `pomdp_params` and CLI (`--action-delay`). `create_environment()` converts to steps:
  ```python
  action_delay_steps = max(0, round(action_delay_days / config.action_interval))  # e.g. 5/5 = 1
  ```
  The env uses a FIFO queue: `reset()` pre-fills queue with `action_delay` default actions (idx 0); each `step()` enqueues the new action and dequeues the oldest. `prev_action_idx` in obs reflects the **applied** action.

# SB3 Pipeline Invariants
When modifying any training or evaluation code, ALL of the following must hold:

1. **VecEnv stack order:** `DummyVecEnv → VecMonitor → VecNormalize → [VecFrameStack]`
   - Train env and eval env MUST have identical wrapper structure (SB3 `sync_envs_normalization` walks both stacks in parallel and will raise `AssertionError` on mismatch).
   - `VecMonitor` must come **before** `VecNormalize`.

2. **Monitor dir must be per-seed:** `tensorboard_dir / f"{agent_name}_seed{seed}"`
   - Using only `agent_name` causes all seeds to append to the same `monitor.csv`, corrupting per-seed learning curves.

3. **`EvalCallback` eval_freq must account for n_envs:**
   ```python
   adjusted_eval_freq = max(1, eval_freq // n_envs)
   ```
   `EvalCallback` counts per-environment steps, not total timesteps. Without this correction the callback may never fire when `total_timesteps` is small.

4. **VecNormalize lifecycle:**
   - Training: `norm_obs=True, norm_reward=True` (stats update continuously).
   - Save after training: `env.save(vecnormalize_path)`.
   - Eval: load frozen — `VecNormalize.load(path, venv)` then set `training=False, norm_reward=False`.

5. **After training, load the best checkpoint** (saved by `EvalCallback`), not the final model state.

# Running Experiments

## Virtual environment
ALWAYS activate the venv before running any Python script:
```bash
source venv/Scripts/activate
python main.py ...
```
Without this, dependencies will not resolve correctly and imports will fail.

## experiments/ directory — DO NOT DELETE
Each scenario run saves results into a **separate timestamped subfolder**:
```
experiments/{scenario_name}/{YYYY-MM-DD_HH-MM-SS}/
```

## Use custom scenarios for smoke tests — NOT predefined ones
Predefined scenarios (`--scenario mdp`, `--scenario underreporting`, etc.) share a **single weights directory** per scenario name (`experiments/{scenario}/weights/`). Running a predefined scenario will **overwrite existing trained weights**.

For smoke tests and validation, always use the **custom scenario flags** instead:
```bash
# Equivalent to --scenario underreporting, but writes to a unique dir
python main.py --no-exposed --detection-rate 0.3 -t 10000 --num-seeds 1

# Equivalent to --scenario pomdp, but writes to a unique dir
python main.py --no-exposed --detection-rate 0.3 --noise-stds 0.05,0.3,0.15 --lag 5,14 -t 10000 --num-seeds 1
```
Custom scenarios generate a unique `scenario_name` (e.g. `custom_no_exposed_k0.3_lag5_14_t10000`) and never collide with the user's predefined experiment weights.

## Keep -t small for smoke tests
One full training run for RecurrentPPO takes ~25 minutes. Use `-t 10000` for any validation or smoke test — it is sufficient to confirm the pipeline works end-to-end:
```bash
python main.py --no-exposed --detection-rate 0.3 -t 10000 --num-seeds 1
```
