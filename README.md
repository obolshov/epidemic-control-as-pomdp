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

# Run POMDP experiment (masked exposed compartment)
python main.py --scenario no_exposed

# Skip training, use existing weights
python main.py --scenario mdp --skip-training all
```

Results are saved to `experiments/{scenario}/{timestamp}/` with plots and logs.
Model weights are shared at `experiments/{scenario}/weights/` for reuse across runs.

## Scenarios

The system supports predefined scenarios for reproducibility:

### **mdp** (Baseline)
Full observability MDP where all SEIR compartments are visible.
```bash
python main.py --scenario mdp
```

### **no_exposed** (POMDP Experiment 1)
Partial observability with masked Exposed (E) compartment.
```bash
python main.py --scenario no_exposed
```

### **underreporting** (POMDP Experiment 2)
Masked E compartment + under-reporting (detection rate k=0.3) + testing capacity saturation (1.5%/day).
The agent observes 30% of true I and R; undetected cases are absorbed into S,
matching real-world surveillance where unconfirmed infections appear as healthy population.
Detection rate further drops during surges via Michaelis-Menten saturation.
```bash
python main.py --scenario underreporting
```

### **noisy_pomdp** (POMDP Experiment 3)
Masked E + under-reporting (k=0.3) + testing saturation (1.5%/day) + AR(1) autocorrelated multiplicative noise (ρ=0.7).
Simulates persistent measurement bias from false-positive/negative testing (I: σ=0.30)
and incomplete recovery statistics (R: σ=0.15). The autocorrelated noise creates
measurement drift that rewards memory-based agents.
```bash
python main.py --scenario noisy_pomdp
```

### **pomdp** (POMDP Experiment 4)
Masked E + under-reporting (k=0.3) + testing saturation (1.5%/day) + AR(1) noise (ρ=0.7) + temporal lag (5–14 days) + action delay (5 days).
The agent receives observations from a random number of days in the past, simulating
bureaucratic and laboratory reporting delays. Additionally, enacted interventions take
5 days to come into effect. The most challenging and realistic scenario.
```bash
python main.py --scenario pomdp
```

### Custom Scenarios
Specify POMDP parameters directly via CLI:
```bash
# Mask E compartment only
python main.py --no-exposed

# Mask E compartment + under-reporting
python main.py --no-exposed --detection-rate 0.3
```

## Agents

Seven agents evaluated in each experiment:

- **no_action**: Always applies NO intervention — upper bound on infections, zero cost
- **severe**: Always applies SEVERE lockdown — lower bound on infections, maximum cost
- **RandomAgent**: Selects random interventions at each step
- **ThresholdAgent**: Rule-based policy calibrated to match PPO performance (thresholds: 1%, 5%, 9%). In POMDP scenarios with underreporting, compensates by dividing observed I by the nominal `detection_rate` before threshold comparison.
- **PPO (Baseline)**: Trained with standard PPO (single-step observations)
- **PPO (FrameStack)**: Uses stacked observations for temporal awareness (sees last 10 decision points)
- **PPO (Recurrent)**: LSTM-based policy that compresses temporal history into hidden state

## CLI Options

```bash
python main.py --help
```

Key options:
- `--scenario, -s`: Predefined scenario (`mdp`, `no_exposed`, `underreporting`, `noisy_pomdp`, `pomdp`)
- `--skip-training`: Skip training for agents (comma-separated list or `all`)
- `--timesteps, -t`: Training timesteps per seed (default: 200 000)
- `--num-seeds, -n`: Number of independent training seeds (default: 5)
- `--no-exposed`: Mask E compartment (custom mode)
- `--detection-rate`: Fraction of true I and R observed, e.g. `0.3` (custom mode)
- `--testing-capacity`: Fraction of population testable per day. When set, detection rate drops during surges via Michaelis-Menten saturation (e.g. `--testing-capacity 0.015` = 1.5%/day)
- `--noise-stds`: Per-compartment multiplicative noise stds (comma-separated, e.g. `--noise-stds 0.05,0.3,0.15` for [S, I, R])
- `--noise-rho`: AR(1) autocorrelation coefficient for multiplicative noise in [0, 1). 0.0 = iid noise (default), 0.7 = persistent measurement bias (decorrelation half-life ≈ 2 steps / 10 days)
- `--lag`: Temporal lag range in days (comma-separated, e.g. `--lag 5,14`). Disabled if omitted.
- `--action-delay`: Action implementation delay in days (e.g. `--action-delay 5`). Enacted interventions take this many days to come into effect.
- `--deterministic`: Use deterministic ODE dynamics instead of stochastic Binomial transitions (adds `_det` suffix to scenario name)
- `--lstm-hidden-size`: LSTM hidden size for RecurrentPPO (default: 32). Non-default values are encoded in the agent name (e.g. `ppo_recurrent_lstm64`), so variants coexist in the same weights directory.
- `--n-stack`: FrameStack depth for ppo_framestack (default: 10). Non-default values are encoded in the agent name (e.g. `ppo_framestack_nstack5`), so variants coexist in the same weights directory.

**Training behavior:**
- By default, trains all RL agents from scratch
- Use `--skip-training all` to load existing weights for all agents
- Use `--skip-training ppo_baseline,ppo_recurrent` to skip specific agents — accepts base names, matches variants (e.g. `ppo_baseline` matches `ppo_baseline_ent0.05`)

## Output Structure

Each experiment creates a timestamped directory, while model weights are shared at the scenario level:

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
    "no_exposed": "no_exposed_t200000/2026-03-21_01-43-42",
    ...
  },
  "framestack_ablation": {
    "ppo_baseline": "pomdp_t300000/<timestamp>",
    "ppo_recurrent": "pomdp_t300000/<timestamp>",
    "n_stack=1": "pomdp_t300000/<timestamp>",
    ...
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

Wilcoxon signed-rank tests with Holm-Bonferroni correction for pairwise agent comparisons across all 5 scenarios. Saves to `analysis_output/significance_tests.csv`.

### FrameStack Window Size Ablation

```bash
python -m analysis.framestack_ablation
```

Line plot of FrameStack reward vs. `n_stack` window size, with RecurrentPPO and PPO baseline as horizontal reference lines. Saves to `analysis_output/framestack_ablation.png`.

To run an ablation without retraining baseline/recurrent for each variant:
```bash
# Run 1: train all agents (first n_stack value)
python main.py --scenario noisy_pomdp --n-stack 5 -t 300000 --num-seeds 5

# Subsequent runs: only train the new framestack variant
python main.py --scenario noisy_pomdp --n-stack 10 -t 300000 --num-seeds 5 --skip-training ppo_baseline,ppo_recurrent
python main.py --scenario noisy_pomdp --n-stack 20 -t 300000 --num-seeds 5 --skip-training ppo_baseline,ppo_recurrent
```
All variants share `experiments/noisy_pomdp_t300000/weights/`, so baseline and recurrent are trained only once.

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

# POMDP: under-reporting with testing capacity saturation
python main.py --no-exposed --detection-rate 0.3 --testing-capacity 0.015

# POMDP: custom noise levels
python main.py --no-exposed --detection-rate 0.3 --noise-stds 0.05,0.3,0.15

# Adjust training duration
python main.py --scenario mdp --timesteps 100000
```
