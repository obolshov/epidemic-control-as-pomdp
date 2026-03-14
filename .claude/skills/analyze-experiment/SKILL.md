---
name: analyze-experiment
description: Analyze experiment results from the experiments/ directory. Use when the user asks to compare experiments, look at results, check agent performance, review action logs, or understand what happened in a training run.
argument-hint: [scenario or path] [focus area]
allowed-tools: Read, Bash, Glob, Grep
---

# Analyze Experiment Results

You are analyzing results from RL epidemic control experiments. All results live under `experiments/`.

## Arguments

`$ARGUMENTS` may specify:
- A scenario name (e.g. `pomdp`, `mdp`) or full path to an experiment run
- A focus area (e.g. "actions", "rewards", "compare all")

If no arguments given, list available experiments and ask what to analyze.

## Directory Structure

```
experiments/
  {scenario_name}/
    weights/                          # Shared trained weights (ignore for analysis)
    {YYYY-MM-DD_HH-MM-SS}/           # Timestamped run — THIS is what you analyze
      config.json                     # Experiment config: base_config, pomdp_params, seeds, timesteps
      summary.json                    # Evaluation metrics: per-agent mean_reward, std_reward, peak/total infected
      plots/                          # Generated plots (PNG images)
        comparison_all_agents.png     # Side-by-side SEIR curves
        evaluation_curves.png         # Reward comparison across agents
        {agent}_seir.png              # Individual SEIR trajectory plots
        {agent}_seed{N}_learning_timesteps.png  # Training learning curves
      logs/
        {agent_name}/                 # Action logs from evaluation episodes
          seed_2024.txt               # One file per eval episode seed
          seed_2025.txt
          ...
        {agent_name}_seed{N}_eval/    # EvalCallback data (evaluations.npz) — use for learning curves
          evaluations.npz             # Keys: timesteps, results (n_evals × n_episodes)
        tensorboard/                  # Training monitor CSVs and TF events
          {agent}_seed{N}/monitor.csv
```

## Key Files and What They Contain

### summary.json
The primary source for comparing agents. Contains per-agent:
- `mean_reward`, `std_reward` — evaluation reward (higher = better)
- `mean_peak_infected`, `std_peak_infected` — epidemic peak (lower = better)
- `mean_total_infected`, `std_total_infected` — cumulative infections (lower = better)
- `per_seed` with individual `episode_rewards` for RL agents

### config.json
Experiment parameters:
- `base_config` — SEIR model params, RL hyperparams (ent_coef, lstm_hidden_size, n_stack, etc.)
- `pomdp_params` — wrapper config (include_exposed, detection_rate, noise_stds, lag, etc.)
- `total_timesteps`, `training_seeds`, `eval_seeds`, `target_agents`

### logs/{agent_name}/seed_{N}.txt
Tab-separated action logs with columns: Day, S, E, I, R, Reward, Action.
- Actions: NO, MILD, MODERATE, SEVERE
- Footer has Summary Statistics: Peak Infected, Total Infected, Total Reward
- **IMPORTANT:** These are in `logs/{agent_name}/`, NOT `logs/{agent_name}_seed{N}_eval/` (that's EvalCallback npz data, a different thing)

### logs/{agent_name}_seed{N}_eval/evaluations.npz
NumPy archive from EvalCallback with training-time eval curve:
```python
data = np.load(path)
timesteps = data['timesteps']       # shape (n_evals,)
results = data['results']           # shape (n_evals, n_episodes)
means = results.mean(axis=1)        # mean reward per eval checkpoint
```

## Analysis Workflow

1. **Start with summary.json** — build a comparison table of all agents
2. **Read config.json** — note key parameters (pomdp_params, timesteps, hyperparams)
3. **Compare action strategies** — read one action log per agent (e.g. seed_2024.txt), extract the action sequence compactly (Day → Action)
4. **If requested, check learning curves** — load evaluations.npz to see training progress
5. **If comparing runs** — read summary.json from each run, align by agent name

## Output Format

Always present results as:
1. **Reward comparison table** (all agents, sorted best→worst)
2. **Key metrics table** (peak infected, total infected)
3. **Action strategy summary** (compact: which actions each agent uses and when)
4. **Notable observations** (which agent is best/worst, interesting patterns, anomalies)

When comparing multiple experiments, use side-by-side tables with clear labels for each run.

Keep analysis concise. Lead with the numbers, then interpret.
