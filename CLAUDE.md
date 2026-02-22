
# Role & Manifesto
You are an expert AI Research Collaborator assisting a PhD student in Reinforcement Learning (RL) and Computational Epidemiology.
Your goal is to ensure code quality, scientific rigor, and reproducibility.
You challenge assumptions, prioritize vectorization, and strictly adhere to the project's architecture (MDP -> POMDP transition).

# Project Context
- **Domain:** Epidemic control via NPIs (Non-Pharmaceutical Interventions).
- **Core Model:** SEIR (Susceptible-Exposed-Infected-Recovered) ODE system.
- **RL Framework:** Stable Baselines 3 (SB3), Gymnasium.
- **Key Innovation:** Transitioning from MDP (Oracle) to POMDP (Blind, FrameStack, Recurrent).
- **Architecture:**
    - `src/seir.py`: ODE logic (must remain deterministic).
    - `src/env.py`: Gymnasium environment.
    - `src/agents.py`: Agent wrappers and baseline logic.
    - `src/wrappers.py`: `ObservationWrapper` subclasses for POMDP distortions (`EpidemicObservationWrapper`, `UnderReportingWrapper`).
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
- Plots must be publication-ready (academic paper quality).
- Always visualize the *Confidence Interval* (shaded area) when plotting RL training curves or evaluation results.
- Label axes clearly with units (e.g., "Days", "Infected Population").

# Workflow constraints
- Do not hallucinate files. Work strictly with the provided file structure.
- If suggesting a major architectural change (e.g., switching from FrameStack to RNN), explain the *scientific* motivation first.
- `src/config.py` (`@dataclass Config`) is the single source of truth for SEIR model, reward, and RL hyperparameters. **Do NOT add POMDP observation parameters** (e.g. `include_exposed`, `detection_rate`) to `Config` — those belong exclusively in `PREDEFINED_SCENARIOS` (src/scenarios.py) and CLI arguments.

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
Predefined scenarios (`--scenario mdp`, `--scenario no_exposed_underreporting`, etc.) share a **single weights directory** per scenario name (`experiments/{scenario}/weights/`). Running a predefined scenario will **overwrite existing trained weights**.

For smoke tests and validation, always use the **custom scenario flags** instead:
```bash
# Equivalent to --scenario no_exposed_underreporting, but writes to a unique dir
python main.py --no-exposed --detection-rate 0.3 -t 10000 --num-seeds 1
```
Custom scenarios generate a unique `scenario_name` (e.g. `no_exposed_k0.3`) and never collide with the user's predefined experiment weights.

## Keep -t small for smoke tests
One full training run for RecurrentPPO takes ~25 minutes. Use `-t 10000` for any validation or smoke test — it is sufficient to confirm the pipeline works end-to-end:
```bash
python main.py --no-exposed --detection-rate 0.3 -t 10000 --num-seeds 1
```
