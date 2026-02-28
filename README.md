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
python run_static_agents.py

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
Masked E compartment + under-reporting of active cases (detection rate k=0.3).
The agent observes 30% of true I and R — consistent with COVID-19 surveillance estimates.
```bash
python main.py --scenario underreporting
```

### **noisy_pomdp** (POMDP Experiment 3)
Masked E + under-reporting (k=0.3) + per-compartment multiplicative noise.
Simulates false-positive/negative testing (I, E: σ=0.30) and incomplete recovery
statistics (R: σ=0.15).
```bash
python main.py --scenario noisy_pomdp
```

### **pomdp** (POMDP Experiment 4)
Masked E + under-reporting (k=0.3) + multiplicative noise + temporal lag (5–14 days).
The agent receives observations from a random number of days in the past, simulating
bureaucratic and laboratory reporting delays. The most challenging and realistic scenario.
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

Agents evaluated in each experiment:

- **RandomAgent**: Selects random interventions
- **ThresholdAgent**: Rule-based policy using infection thresholds
- **PPO (Baseline)**: Trained with standard PPO (single-step observations)
- **PPO (FrameStack)**: Uses stacked observations for temporal awareness (sees last 10 time steps)
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
- `--noise-stds`: Per-compartment multiplicative noise stds (pass once per value, e.g. `--noise-stds 0.05 --noise-stds 0.3 --noise-stds 0.15` for [S, I, R])
- `--lag`: Temporal lag range in days (pass twice: `--lag 5 --lag 14`). Disabled if omitted.
- `--deterministic`: Use deterministic ODE dynamics instead of stochastic Binomial transitions (adds `_det` suffix to scenario name)

**Training behavior:**
- By default, trains all RL agents from scratch
- Use `--skip-training all` to load existing weights for all agents
- Use `--skip-training ppo_baseline,ppo_framestack` to skip specific agents

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
      summary.json                              # Key metrics for all agents
      plots/
        comparison_all_agents.png               # Side-by-side SEIR curves
        evaluation_curves.png                   # Mean ± 95% CI reward across seeds
        random_agent_seir.png                   # Individual agent SEIR plots
        threshold_agent_seir.png
        ppo_baseline_seir.png
        ppo_baseline_seed42_learning.png        # Per-seed monitor-based learning curve
      logs/
        random_agent.txt                        # Detailed action logs
        threshold_agent.txt
        ppo_baseline.txt
        tensorboard/
          ppo_baseline_seed42/                  # VecMonitor logs (per-seed)
          ppo_baseline_seed42_1/                # TensorBoard logs (per-seed)
```

**Note:** Weights are stored at `experiments/{scenario}/weights/` to enable reuse across multiple runs. Each seed produces its own weight file and VecNormalize stats.


## Model Verification

For quick epidemic model sanity checks:

```bash
python run_static_agents.py
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

# POMDP: custom detection rate
python main.py --no-exposed --detection-rate 0.5

# POMDP: custom noise levels
python main.py --no-exposed --detection-rate 0.3 \
  --noise-stds 0.05 --noise-stds 0.3 --noise-stds 0.15

# Adjust training duration
python main.py --scenario mdp --timesteps 100000
```

## Extending the System

To add new POMDP parameters, RL agents, or scenarios, see [EXTENDING.md](EXTENDING.md).

