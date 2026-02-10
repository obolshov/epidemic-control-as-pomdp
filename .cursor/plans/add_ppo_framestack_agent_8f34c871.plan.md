---
name: Add PPO FrameStack Agent
overview: Add a new PPO FrameStack agent that uses VecFrameStack to provide temporal awareness by stacking observations from multiple consecutive days, enabling the agent to learn temporal patterns in the epidemic dynamics.
todos:
  - id: add-n-stack-config
    content: Add n_stack=5 parameter to DefaultConfig in src/config.py
    status: completed
  - id: update-scenarios
    content: Add ppo_framestack to target_agents in mdp and no_exposed scenarios
    status: completed
  - id: implement-framestack-training
    content: Modify train_ppo_agent() to apply DummyVecEnv + VecFrameStack for ppo_framestack agent
    status: completed
  - id: verify-evaluation
    content: Check and update run_agent() in evaluation.py for VecEnv compatibility
    status: completed
  - id: update-docs
    content: Update README.md and EXTENDING.md to reflect ppo_framestack as active agent
    status: completed
  - id: test-training
    content: Run test training with --timesteps 5000 to verify implementation
    status: completed
  - id: full-evaluation
    content: Execute full experiment to generate comparison plots and validate performance
    status: completed
isProject: false
---

# Add PPO FrameStack Agent

## Overview

Implement a new RL agent `ppo_framestack` that uses Stable Baselines3's `VecFrameStack` wrapper to stack observations from multiple consecutive days, providing temporal awareness for better policy learning in epidemic control.

## Key Architecture Decisions

### Frame Stacking Approach

The PPO FrameStack agent will use SB3's vectorized environment wrappers:

```python
DummyVecEnv([lambda: env]) -> VecFrameStack(env, n_stack=5)
```

This differs from `ppo_baseline` which sees only the current observation `[S, E, I, R]`. The `ppo_framestack` agent will see:

```
[S_t-4, E_t-4, I_t-4, R_t-4,  # 5 days ago
 S_t-3, E_t-3, I_t-3, R_t-3,  # 4 days ago
 ...
 S_t, E_t, I_t, R_t]          # today
```

This gives the agent temporal context to recognize trends (increasing infections, effect of past interventions, etc.).

### Why VecFrameStack Requires Special Handling

Unlike observation wrappers (e.g., `[EpidemicObservationWrapper](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\wrappers.py)`), `VecFrameStack`:

- **Operates on vectorized environments** (VecEnv), not single environments
- **Cannot be applied in `create_environment()**` since that returns a single Env
- **Must be applied in training** after wrapping with `DummyVecEnv`
- **Changes observation space dimensionality** from `(4,)` to `(4*n_stack,)` or `(n_stack, 4)` depending on channel order

## Implementation Steps

### 1. Add `n_stack` Parameter to Config

**File:** `[src/config.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\config.py)`

Add to `DefaultConfig`:

```python
# Frame stacking for temporal awareness
self.n_stack = 5  # Number of consecutive observations to stack
```

This makes `n_stack` globally configurable and consistent across experiments.

### 2. Add `ppo_framestack` to Scenarios

**File:** `[src/scenarios.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\scenarios.py)`

Update `PREDEFINED_SCENARIOS`:

```python
"mdp": {
    "description": "Baseline MDP (full observability, all target agents)",
    "pomdp_params": {"include_exposed": True},
    "target_agents": ["random", "threshold", "ppo_baseline", "ppo_framestack"],
},
"no_exposed": {
    "description": "POMDP Experiment 1: Masked E compartment",
    "pomdp_params": {"include_exposed": False},
    "target_agents": ["random", "threshold", "ppo_baseline", "ppo_framestack"],
},
```

### 3. Implement FrameStack Training Logic

**File:** `[src/train.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\train.py)`

Modify `train_ppo_agent()` to detect `ppo_framestack` and apply vectorization + frame stacking:

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def train_ppo_agent(...):
    # Existing environment creation...
    env = env_cls(config)
    if not pomdp_params.get("include_exposed", True):
        env = EpidemicObservationWrapper(env, include_exposed=False)
    
    # Apply frame stacking for ppo_framestack agent
    if agent_name == "ppo_framestack":
        # Wrap in DummyVecEnv first (required by VecFrameStack)
        env = DummyVecEnv([lambda: env])
        # Apply frame stacking
        env = VecFrameStack(env, n_stack=config.n_stack)
        print(f"Applied VecFrameStack with n_stack={config.n_stack}")
    else:
        # Standard Monitor wrapper for non-framestack agents
        monitor_dir = experiment_dir.tensorboard_dir / agent_name
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_dir))
    
    # Training continues as normal...
```

**Critical Implementation Details:**

- VecFrameStack must wrap a VecEnv, not a regular Env
- Monitor wrapper conflicts with VecEnv - only use Monitor for non-vectorized envs
- The observation space automatically changes from `(4,)` to `(20,)` with n_stack=5

### 4. Handle Evaluation for VecEnv

**File:** `[src/evaluation.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\evaluation.py)`

Check if evaluation logic in `run_agent()` needs updates:

- VecEnv has different reset/step APIs: returns arrays, not single values
- May need to extract single environment from vectorized wrapper for evaluation
- Alternative: Keep evaluation on single env, but ensure model can predict on both

**Investigation needed:** Check current `run_agent()` implementation to see if it already handles VecEnv or needs modification.

### 5. Update Documentation

**Files:** `[README.md](c:\Users\apathy\Projects\epidemic-control-as-pomdp\README.md)`, `[EXTENDING.md](c:\Users\apathy\Projects\epidemic-control-as-pomdp\EXTENDING.md)`

Update agent list:

- Change `*(Future)* **PPO (FrameStack)**` to active agent description
- Add notes about n_stack configuration in DefaultConfig
- Update example outputs to include ppo_framestack results

## Technical Considerations

### Observation Space Changes


| Agent                        | Observation Space | Example                                             |
| ---------------------------- | ----------------- | --------------------------------------------------- |
| `ppo_baseline`               | `Box(4,)`         | `[S, E, I, R]`                                      |
| `ppo_framestack` (n_stack=5) | `Box(20,)`        | `[S_-4, E_-4, I_-4, R_-4, ..., S_0, E_0, I_0, R_0]` |


### Why 5 Frames?

With `action_interval=5` days and `n_stack=5`:

- Agent sees 5 × 5 = 25 days of history
- Captures full cycle of one intervention period
- Sufficient to detect trends without excessive memory

### POMDP Compatibility

Frame stacking works with partial observability:

- If `include_exposed=False`, each frame is `[S, I, R]` instead of `[S, E, I, R]`
- VecFrameStack applied **after** `EpidemicObservationWrapper`
- Observation space becomes `(3*n_stack,)` instead of `(4*n_stack,)`

## Expected Outcomes

### Training

- Same timesteps as `ppo_baseline` (50,000 default)
- Comparable training time (slightly slower due to larger observation space)
- Model saved to: `experiments/{scenario}/weights/ppo_framestack.zip`

### Evaluation

- Learning curves: `plots/ppo_framestack_learning.png`
- SEIR trajectory: `plots/ppo_framestack_seir.png`
- Comparison with baseline in: `plots/comparison_all_agents.png`

### Performance Hypothesis

`ppo_framestack` should outperform `ppo_baseline` in scenarios where temporal patterns matter:

- Detecting rising infection trends earlier
- Learning optimal timing for intervention changes
- Better handling of delayed effects of NPIs

## Testing Strategy

1. **Syntax check:** Verify imports and VecFrameStack API usage
2. **Observation space:** Print observation shapes during training to confirm stacking
3. **Training run:** Execute `python main.py --scenario mdp --timesteps 5000` (quick test)
4. **Full evaluation:** Run `python main.py --scenario mdp` to train and evaluate
5. **POMDP scenario:** Test with `python main.py --scenario no_exposed` to verify compatibility

## Files to Modify

1. `[src/config.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\config.py)` - Add `n_stack` parameter
2. `[src/scenarios.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\scenarios.py)` - Add `ppo_framestack` to target_agents
3. `[src/train.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\train.py)` - Implement frame stacking logic
4. `[src/evaluation.py](c:\Users\apathy\Projects\epidemic-control-as-pomdp\src\evaluation.py)` - Verify VecEnv compatibility (investigation)
5. `[README.md](c:\Users\apathy\Projects\epidemic-control-as-pomdp\README.md)` - Update agent documentation
6. `[EXTENDING.md](c:\Users\apathy\Projects\epidemic-control-as-pomdp\EXTENDING.md)` - Update examples

## Scientific Justification

From RL perspective, frame stacking addresses the **temporal credit assignment problem** in epidemic control:

- Intervention effects are delayed (incubation period, recovery time)
- Single-step observations are insufficient to distinguish "rising" vs "declining" infection trends
- Frame stacking provides Markovian approximation for partially observable temporal dynamics

This is analogous to Atari RL where frame stacking helps agents perceive velocity and acceleration from raw pixel observations.