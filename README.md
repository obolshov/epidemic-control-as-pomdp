# Epidemic Control as Partially Observable Domain

Reinforcement learning framework for epidemic control using Non-Pharmaceutical Interventions (NPIs). Implements MDP and POMDP scenarios for studying partial observability effects on optimal control policies.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Verify epidemic model with static policies
python support/run_static_agents.py

# Run MDP experiment (trains RL agents by default)
python main.py --scenario mdp

# Run POMDP experiment (incomplete surveillance)
python main.py --scenario incompleteness

# Skip training, use existing weights
python main.py --scenario mdp --skip-training all
```

Results are saved to `experiments/{scenario}/{timestamp_or_run_name}/` with plots and logs.
Model weights are shared at `experiments/{scenario}/weights/` for reuse across runs.

## Scenarios

The system supports predefined scenarios for reproducibility:

### **mdp** (Baseline)
Full observability MDP where all SEIR compartments are visible.
```bash
python main.py --scenario mdp
```

### **incompleteness** (POMDP Experiment 1)
Incomplete surveillance: masked Exposed (E) compartment + per-episode stochastic under-reporting
(detection rate k ~ U[0.15, 0.40]) with testing-capacity saturation (r ~ U[0.5%, 2.0%] of population/day).
Detection parameters are resampled at the start of every episode and held constant throughout it,
so the true detection regime is a latent variable the agent must infer from trajectory dynamics.
Masked E is folded into the observed S (pre-symptomatic exposed are not tested and not symptomatic,
so they appear as healthy population), and undetected I and R are likewise absorbed into S.
This keeps `S_obs + I_obs + R_obs = N` identically, preventing the agent from recovering E via
SEIR conservation algebra. Detection rate further drops during surges via Michaelis-Menten saturation.
```bash
python main.py --scenario incompleteness
```

### **incompleteness_and_noise** (POMDP Experiment 2)
Stochastic incomplete surveillance + AR(1) autocorrelated multiplicative noise (ρ=0.7).
Simulates persistent measurement bias from false-positive/negative testing (I: σ=0.30)
and incomplete recovery statistics (R: σ=0.15). The autocorrelated noise creates
measurement drift that rewards memory-based agents.
```bash
python main.py --scenario incompleteness_and_noise
```

### **pomdp** (POMDP Experiment 3)
Stochastic incomplete surveillance + AR(1) noise (ρ=0.7) + temporal lag (5–14 days).
The agent receives observations from a random number of days in the past, simulating
bureaucratic and laboratory reporting delays. The most challenging and realistic scenario.
```bash
python main.py --scenario pomdp
```

## Agents

Seven agents evaluated in each experiment:

- **no_action**: Always applies NO intervention — upper bound on infections, zero cost
- **severe**: Always applies SEVERE lockdown — lower bound on infections, maximum cost
- **RandomAgent**: Selects random interventions at each step
- **ThresholdAgent**: Rule-based policy calibrated to match PPO performance (thresholds: 1%, 5%, 9%). In POMDP scenarios with underreporting, compensates by dividing observed I by a fixed detection-rate estimate before threshold comparison — the prior mean of the sampling range when `detection_rate` is stochastic per episode.
- **PPO (Baseline)**: Trained with standard PPO (single-step observations)
- **PPO (FrameStack)**: Uses stacked observations for temporal awareness (sees last 10 decision points)
- **PPO (Recurrent)**: LSTM-based policy that compresses temporal history into hidden state

## CLI Options

```bash
python main.py --help
```

Key options:
- `--scenario, -s` **(required)**: Predefined scenario. Available: `mdp`, `incompleteness`, `incompleteness_and_noise`, `pomdp`, `only_noise`, `only_temporal`
- `--skip-training`: Skip training for agents (comma-separated list or `all`)
- `--timesteps, -t`: Training timesteps per seed (default: 1 000 000)
- `--num-seeds, -n`: Number of independent training seeds (default: 5)
- `--deterministic`: Use deterministic ODE dynamics instead of stochastic Binomial transitions (adds `_det` suffix to scenario name)
- `--lstm-hidden-size`: LSTM hidden size for RecurrentPPO (default: 32). Non-default values are encoded in the agent name (e.g. `ppo_recurrent_lstm64`), so variants coexist in the same weights directory.
- `--n-stack`: FrameStack depth for ppo_framestack (default: 20). Non-default values are encoded in the agent name (e.g. `ppo_framestack_nstack5`), so variants coexist in the same weights directory.
- `--ent-coef`: Entropy bonus for all RL agents (default: 0.2). Non-default values are encoded in the agent name (e.g. `ppo_baseline_ent0.05`).
- `--recurrent-n-steps`: Rollout length per env for RecurrentPPO (default: 256). Non-default values are encoded in the agent name (e.g. `ppo_recurrent_nsteps512`).
- `--run-name`: Custom name for the results subfolder (default: auto-generated timestamp). Useful for labelling runs semantically (e.g. `--run-name baseline_v2`). Raises an error if the folder already exists.
- `--resume-from`: Scenario folder name to resume training from (e.g. `pomdp_t500000`). Loads weights from `experiments/{name}/weights/`. Agents without matching weights train from scratch. New weights are saved to the new scenario folder.

**Training behavior:**
- By default, trains all RL agents from scratch
- Use `--skip-training all` to load existing weights for all agents
- Use `--skip-training ppo_baseline,ppo_recurrent` to skip specific agents — accepts base names, matches variants (e.g. `ppo_baseline` matches `ppo_baseline_ent0.05`)
- Use `--resume-from pomdp_t500000` to continue training from existing weights (e.g. extend from 500k to 1M timesteps)

## Output Structure

Each experiment creates a timestamped directory (or a custom `--run-name` subfolder), while model weights are shared at the scenario level:

```
experiments/
  mdp/
    weights/                                    # Shared weights (reused across runs)
      ppo_baseline_seed42.zip                   # Per-seed model checkpoint
      ppo_baseline_seed42_vecnormalize.pkl      # VecNormalize running stats (frozen at eval)
      best_ppo_baseline_seed42/
        best_model.zip                          # Best checkpoint saved by EvalCallback
    2026-02-07_14-30-00/                        # Timestamped experiment results
      config.json                               # Full experiment configuration
      summary.json                              # Cross-seed aggregated stats (mean ± SE per agent)
      evaluation.json                            # Raw per-episode metrics grouped by seed
      plots/
        comparison_all_agents.png               # Side-by-side SEIR curves (cross-seed mean ± SD shaded)
        evaluation_curves.png                   # Mean ± SD reward across training seeds
        no_action_seir.png                      # Per-agent SEIR (cross-seed mean ± SD shaded)
        severe_seir.png
        random_seir.png
        threshold_seir.png
        ppo_baseline_seir.png
        ppo_baseline_seed42_learning_timesteps.png  # Per-seed monitor-based learning curve
      logs/
        no_action/                              # Per-agent action logs
          seed_2024.txt                         # One log per eval episode
          seed_2025.txt
          ...
        severe/
        random/
        threshold/
        ppo_baseline/
        tensorboard/
          ppo_baseline_seed42/                  # VecMonitor logs (per-seed)
          ppo_baseline_seed42_1/                # TensorBoard logs (per-seed)
```

**Note:** Weights are stored at `experiments/{scenario}/weights/` to enable reuse across multiple runs. Each seed produces its own weight file and VecNormalize stats.


## Cross-Scenario Analysis

Analysis scripts live in the `analysis/` package. They load experiment data via a manifest file (`analyses.json`) that maps analysis names to specific experiment runs.

### Manifest (`analyses.json`)

The manifest explicitly declares which experiment run to use for each analysis. Copy the example and fill in timestamps:

```bash
cp analyses.json.example analyses.json
```

```json
{
  "pomdp_gap": {
    "mdp": "mdp_t200000/2026-03-21_01-42-45",
    "incompleteness": "incompleteness_t200000/2026-03-21_01-43-42",
    ...
  },
  "framestack_ablation": {
    "ppo_baseline": "pomdp_t300000/<timestamp>",
    "n_stack=1": "pomdp_t300000/<timestamp>",
    ...
  },
  "comparisons": {
    "my_comparison": {
      "Label A": {"path": "scenario_t500000/<timestamp>", "agent": "ppo_baseline"},
      "Label B": {"path": "other_scenario/<timestamp>", "agent": "ppo_recurrent"}
    }
  }
}
```

Values are relative paths under `experiments/`. Update this file when you run new experiments that should be included in analysis. The file is gitignored (local to each machine); `analyses.json.example` is tracked.

### Validate manifest

```bash
python -m analysis.validate
```

Checks that all manifest entries point to existing directories with `config.json`, `summary.json`, and `evaluation.json`.

### POMDP Gap Plot

```bash
python -m analysis.pomdp_gap
```

Produces `analysis_output/pomdp_gap_plot.png` — a 3-panel figure (Reward, Total Infected, Total Stringency) with error bars.

### Statistical Significance Tests

```bash
python -m analysis.significance_tests
```

Wilcoxon signed-rank tests with Holm-Bonferroni correction for pairwise agent comparisons across all 4 scenarios. Saves to `analysis_output/significance_tests.csv`.

### Distortion Ablation Study

```bash
python -m analysis.distortion_ablation
```

Prints a summary table and saves a heatmap (`analysis_output/distortion_ablation.png`) showing how much each isolated distortion type (incompleteness, noise, temporal) degrades each agent's reward relative to the MDP baseline. Requires experiments for the isolated distortion scenarios (`incompleteness`, `only_noise`, `only_temporal`) plus `mdp`; map these to analysis keys in `analyses.json` under `distortion_ablation`.

### FrameStack Window Size Ablation

```bash
python -m analysis.framestack_ablation
```

Prints a summary table of all n_stack metrics, saves a line plot of reward vs. `n_stack` with RecurrentPPO and PPO baseline reference lines (`analysis_output/framestack_ablation.png`), and plots training learning curves for selected window sizes (`analysis_output/framestack_learning_curves.png`).

To run an ablation without retraining baseline/recurrent for each variant:
```bash
# Run 1: train all agents (first n_stack value)
python main.py --scenario incompleteness_and_noise --n-stack 5 -t 300000 --num-seeds 5

# Subsequent runs: only train the new framestack variant
python main.py --scenario incompleteness_and_noise --n-stack 10 -t 300000 --num-seeds 5 --skip-training ppo_baseline,ppo_recurrent
python main.py --scenario incompleteness_and_noise --n-stack 20 -t 300000 --num-seeds 5 --skip-training ppo_baseline,ppo_recurrent
```
All variants share `experiments/incompleteness_and_noise_t300000/weights/`, so baseline and recurrent are trained only once.

### Ad-hoc Comparison Table

```bash
python -m analysis.compare <comparison_name>
python -m analysis.compare --list
```

Prints a summary table (Reward, SE, Stringency, Peak Inf, Total Inf, Seed Std) for arbitrary experiment entries defined in `analyses.json["comparisons"]`. Each entry maps a display label to `{"path": "...", "agent": "..."}`, allowing you to compare results from completely different scenarios and runs side by side. If `agent` is omitted, all agents from the run are shown as separate rows.

### Reward Grid (Scenarios × Agents)

```bash
python -m analysis.reward_grid <grid_name>
python -m analysis.reward_grid --list
```

Prints a 2D table of `cross_seed_mean_reward ± SE` with scenarios as rows and agents as columns. Each cell is configured independently in `analyses.json["reward_grid"][<grid_name>]` as `scenario -> agent_label -> {"path": "...", "agent": "..."}`, so a single scenario can pull from a run with non-default hyperparameters (e.g. `ppo_recurrent` with `lstm_hidden_size=64` under `pomdp`) while the rest point at default runs. Missing cells render as `—`.

### Using the data loading library

```python
from analysis.data import load_analysis

# Load runs for any analysis defined in analyses.json
runs = load_analysis("pomdp_gap")
for label, run in runs.items():
    metrics = run.agent_metrics("ppo_baseline")
    rewards = run.agent_episode_rewards("ppo_baseline")
```

## Model Verification

For quick epidemic model sanity checks:

```bash
python support/run_static_agents.py
```

Runs all four static policies (NO, MILD, MODERATE, SEVERE interventions) and generates a comparison plot.

## Examples

```bash
# Train all agents (default behavior)
python main.py --scenario mdp

# Skip training, use existing weights
python main.py --scenario mdp --skip-training all

# Train only new agents, skip ppo_baseline
python main.py --scenario mdp --skip-training ppo_baseline

# POMDP: full distortions including temporal lag
python main.py --scenario pomdp

# Adjust training duration
python main.py --scenario mdp --timesteps 100000
```
