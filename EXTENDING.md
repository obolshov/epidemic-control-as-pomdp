# Extending the Experiment System

This guide explains how to extend the codebase with new POMDP parameters, RL agents, and scenarios. Designed for developers and researchers adding new functionality.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Adding POMDP Parameters](#adding-pomdp-parameters)
- [Adding RL Agents](#adding-rl-agents)
- [Adding Predefined Scenarios](#adding-predefined-scenarios)
- [File Naming Conventions](#file-naming-conventions)
- [Code Structure](#code-structure)

## Architecture Overview

The experiment system consists of three core components:

### 1. **ExperimentConfig** (`src/experiment.py`)
Stores complete experiment configuration:
- Base SEIR model parameters (`DefaultConfig`)
- POMDP modifications (`pomdp_params` dict)
- Target agents to evaluate
- Training settings

Serializes to `config.json` for reproducibility.

### 2. **ExperimentDirectory** (`src/experiment.py`)
Manages directory structure and file paths:
- Creates `experiments/{scenario}/{timestamp}/` hierarchy for results
- **Weights stored at scenario level**: `experiments/{scenario}/weights/` (shared across runs)
- Provides methods: `get_weight_path()`, `get_plot_path()`, `get_log_path()`
- Handles `config.json` and `summary.json` saving

**Weight Management:**
- Model weights are stored at `experiments/{scenario}/weights/` to enable reuse
- Training agents overwrites existing weights at this location
- Use `--skip-training` to load existing weights instead of training

### 3. **Scenarios** (`src/scenarios.py`)
Defines predefined POMDP configurations:
- `PREDEFINED_SCENARIOS` dict
- `create_custom_scenario_name()` for ad-hoc configurations

## Adding POMDP Parameters

Adding a new POMDP parameter (e.g., observation delay) requires 4 simple steps:

### Step 1: Add CLI Option

Edit `main.py`, add parameter to `main()` function:

```python
@app.command()
def main(
    # ... existing options ...
    delay: int = typer.Option(
        0,
        "--delay",
        help="Observation delay in days"
    ),
):
```

### Step 2: Include in pomdp_params

Add to the `pomdp_params` dictionary:

```python
# In main() function, around line 280
pomdp_params = {
    "include_exposed": not no_exposed,
    "delay": delay,  # ← Add here
    # "noise_std": noise,  # Future parameter
}
```

### Step 3: Implement Wrapper

Create wrapper in `src/wrappers.py`:

```python
class DelayWrapper(gym.ObservationWrapper):
    """Delays observations by specified number of days."""
    
    def __init__(self, env: gym.Env, delay: int):
        super().__init__(env)
        self.delay = delay
        self.observation_buffer = []
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        self.observation_buffer.append(obs)
        if len(self.observation_buffer) > self.delay:
            return self.observation_buffer.pop(0)
        return obs  # Return current obs if buffer not full
```

Apply wrapper in `main.py` `create_environment()` function:

```python
def create_environment(config: DefaultConfig, pomdp_params: dict) -> EpidemicEnv:
    env = EpidemicEnv(config)
    
    if not pomdp_params.get("include_exposed", True):
        env = EpidemicObservationWrapper(env, include_exposed=False)
    
    if pomdp_params.get("delay", 0) > 0:  # ← Add here
        env = DelayWrapper(env, delay=pomdp_params["delay"])
    
    return env
```

### Step 4: Update Scenario Naming

Edit `src/scenarios.py` `create_custom_scenario_name()`:

```python
def create_custom_scenario_name(pomdp_params: Dict[str, Any]) -> str:
    parts = []
    
    if not pomdp_params.get("include_exposed", True):
        parts.append("no_exposed")
    
    if "delay" in pomdp_params and pomdp_params["delay"] > 0:  # ← Add here
        parts.append(f"delay{pomdp_params['delay']}")
    
    if "noise_std" in pomdp_params and pomdp_params["noise_std"] > 0:
        parts.append(f"noise{pomdp_params['noise_std']}")
    
    return "_".join(parts) if parts else "custom"
```

**Done!** The parameter is now:
- Available via CLI: `python main.py --delay 5 --train-ppo`
- Saved to `config.json`
- Included in experiment directory name
- Applied to environment via wrapper

**Example usage:**
```bash
python main.py --no-exposed --delay 5
# Saves to: experiments/no_exposed_delay5/{timestamp}/
# Weights to: experiments/no_exposed_delay5/weights/

# Skip training, use existing weights
python main.py --no-exposed --delay 5 --skip-training all
```

## Adding RL Agents

To add a new RL agent type (e.g., PPO with frame stacking):

### Step 1: Add to Target Agents

Edit `src/scenarios.py`, add to scenario configurations:

```python
PREDEFINED_SCENARIOS = {
    "mdp": {
        "description": "Baseline MDP (full observability, all target agents)",
        "pomdp_params": {"include_exposed": True},
        "target_agents": [
            "random",
            "threshold",
            "ppo_baseline",
            "ppo_framestack",  # ← Add here
        ],
    },
    # ... other scenarios
}
```

### Step 2: Implement Training Logic (if needed)

For standard PPO variants, no changes needed - the system handles it automatically.

For custom architectures requiring special environment wrappers, modify `src/train.py`.

**Example: PPO FrameStack (already implemented)**

The `ppo_framestack` agent uses VecFrameStack to stack observations from multiple time steps:

```python
# In train_ppo_agent()
if agent_name == "ppo_framestack":
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=config.n_stack)
```

**Important:** The `n_stack` parameter is configured in `DefaultConfig` (currently set to 5).

When adding agents that use VecEnv wrappers, you must also update `run_evaluation()` in `main.py` to create the appropriate environment for evaluation.

### Step 3: Test

```bash
python main.py --scenario mdp
```

The system automatically:
- Trains the new agent
- Saves weights to `experiments/mdp/weights/{agent_name}.zip`
- Generates plots: `plots/{agent_name}_seir.png`, `plots/{agent_name}_learning.png`
- Creates logs: `logs/{agent_name}.txt`

**To train only the new agent** (skip existing agents):
```bash
# Skip ppo_baseline, only train ppo_framestack
python main.py --scenario mdp --skip-training ppo_baseline

# Skip multiple agents
python main.py --scenario mdp --skip-training ppo_baseline,ppo_framestack
```

## Adding Predefined Scenarios

Predefined scenarios ensure reproducibility and make common configurations easy to run.

### Add to src/scenarios.py

```python
PREDEFINED_SCENARIOS = {
    # ... existing scenarios ...
    
    "noisy_observations": {
        "description": "POMDP Experiment 2: Gaussian observation noise",
        "pomdp_params": {
            "include_exposed": True,
            "noise_std": 0.1,
        },
        "target_agents": ["random", "threshold", "ppo_baseline"],
    },
    
    "delayed_partial": {
        "description": "POMDP Experiment 3: Partial obs + delay",
        "pomdp_params": {
            "include_exposed": False,
            "delay": 5,
        },
        "target_agents": ["random", "threshold", "ppo_baseline", "ppo_recurrent"],
    },
}
```

### Usage

```bash
python main.py --scenario noisy_observations
# Results in: experiments/noisy_observations/{timestamp}/
# Weights in: experiments/noisy_observations/weights/

python main.py --scenario delayed_partial
# Results in: experiments/delayed_partial/{timestamp}/
# Weights in: experiments/delayed_partial/weights/

# Use existing weights
python main.py --scenario noisy_observations --skip-training all
```

## File Naming Conventions

The system uses **strict snake_case naming** for all outputs:

### Agents
- Use lowercase with underscores: `random_agent`, `threshold_agent`, `ppo_baseline`
- Avoid spaces, hyphens, or mixed case

### Plots (`plots/`)
- `comparison_all_agents.png` - Multi-agent comparison
- `{agent_name}_seir.png` - Individual SEIR curves
- `{agent_name}_learning_episodes.png` - Learning curve (episodes)
- `{agent_name}_learning_timesteps.png` - Learning curve (timesteps)

### Logs (`logs/`)
- `{agent_name}.txt` - Action-by-action log with state and rewards

### Weights (`weights/`)
- `{agent_name}.zip` - SB3 model weights

### Scenarios
- Use lowercase with underscores: `mdp`, `no_exposed`, `noisy_observations`
- Avoid camelCase, spaces, or hyphens

## Code Structure

### Key Files

```
src/
  experiment.py       # ExperimentConfig, ExperimentDirectory
  scenarios.py        # PREDEFINED_SCENARIOS, scenario utilities
  config.py           # DefaultConfig (SEIR parameters)
  env.py              # EpidemicEnv (Gymnasium environment)
  wrappers.py         # POMDP wrappers (ObservationWrapper, etc.)
  train.py            # RL training logic
  evaluation.py       # Agent evaluation and logging
  utils.py            # Plotting and logging utilities
  agents.py           # Agent implementations
  seir.py             # SEIR ODE solver

main.py               # CLI entry point
run_static_agents.py  # Static policy verification
```

### Data Flow

```
CLI (main.py)
  ↓ Creates
ExperimentConfig
  ↓ Passed to
ExperimentDirectory (creates dirs)
  ↓ Used by
train_ppo_agent() → run_evaluation() → plot_all_results()
  ↓ Saves to
experiments/{scenario}/{timestamp}/
```

### Important Classes

**ExperimentConfig** (dataclass):
- `base_config`: DefaultConfig
- `pomdp_params`: Dict[str, Any]
- `scenario_name`: str
- `target_agents`: List[str]
- `train_rl`: bool
- `total_timesteps`: int

**ExperimentDirectory**:
- `__init__(exp_config)`: Creates directory structure
- `save_config()`: Writes config.json
- `save_summary(results)`: Writes summary.json
- `get_weight_path(agent_name)`: Returns Path
- `get_plot_path(filename)`: Returns Path
- `get_log_path(agent_name)`: Returns Path

## Variable Naming Standards

**Use full, descriptive names:**
- ✅ `experiment_dir` (NOT `exp_dir`)
- ✅ `experiment_config` (NOT `exp_config`)
- ✅ `pomdp_params` (OK, widely understood)
- ✅ `agent_name` (NOT `name` or `ag`)

**Avoid ambiguous abbreviations:**
- ❌ `exp` (exponent? experiment? exposure?)
- ❌ `obs` in variable names (use `observation`)
- ❌ `cfg` (use `config`)

## Testing New Features

After adding features:

### 1. Syntax Check
```bash
python -m py_compile main.py src/train.py src/wrappers.py
```

### 2. Run Unit Tests
```bash
python test_refactoring.py
```

### 3. Quick Experiment Test
```bash
# Test without training (fast)
python main.py --scenario mdp

# Test with training (slow but thorough)
python main.py --scenario mdp --timesteps 1000

# Test loading existing weights
python main.py --scenario mdp --skip-training all
```

### 4. Check Output Structure
Verify files are created with correct naming:
```
experiments/
  {scenario}/
    weights/                    # Shared weights
      {agent}.zip
    {timestamp}/                # Results
      config.json
      summary.json
      plots/{agent}_*.png
      logs/{agent}.txt
```

## Best Practices

1. **Always test with multiple agents** - Ensure your changes work for all agent types
2. **Check config.json serialization** - New parameters should appear in saved config
3. **Maintain backward compatibility** - Old experiments should still load
4. **Follow naming conventions** - Use snake_case consistently
5. **Document POMDP parameters** - Add clear docstrings for new wrappers
6. **Keep it simple** - If adding a parameter requires >10 lines, consider refactoring

## Common Patterns

### Adding Observation Wrapper
```python
# src/wrappers.py
class YourWrapper(gym.ObservationWrapper):
    def __init__(self, env, your_param):
        super().__init__(env)
        self.your_param = your_param
        # Adjust observation_space if needed
    
    def observation(self, obs):
        # Modify observation
        return modified_obs

# main.py - create_environment()
if pomdp_params.get("your_param"):
    env = YourWrapper(env, pomdp_params["your_param"])
```

### Adding Reward Wrapper
```python
# src/wrappers.py
class YourRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # Modify reward
        return modified_reward
```

### Custom Agent Architecture
```python
# src/agents.py or new file
class CustomPolicyAgent(Agent):
    def __init__(self, model_path=None):
        # Load or initialize custom model
        pass
    
    def predict(self, observation, deterministic=True):
        # Return (action_idx, state)
        return action_idx, None
```

## Questions?

If you encounter issues or have questions:

1. Check existing examples in `src/scenarios.py` and `src/wrappers.py`
2. Review `experiments/` output to verify behavior
3. Run `test_refactoring.py` to ensure core system works
4. Check `UPDATES.md` for recent changes

Happy extending! 🚀
