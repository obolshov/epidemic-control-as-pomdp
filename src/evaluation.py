import numpy as np
import os
from typing import Optional

from src.agents import Agent
from src.env import EpidemicEnv, SimulationResult
from src.utils import log_results, plot_single_result


def run_agent(
    agent: Agent,
    env: EpidemicEnv,
    experiment_dir: Optional["ExperimentDirectory"] = None,
    agent_name: Optional[str] = None,
) -> SimulationResult:
    """
    Runs a simulation for a single agent.

    :param agent: The agent to evaluate.
    :param env: The environment to run the simulation in (can be VecEnv or regular Env).
    :param experiment_dir: ExperimentDirectory for saving outputs. If None, saves to default locations.
    :param agent_name: Override name for saving files. If None, uses agent's class name.
    :return: A SimulationResult object containing the results of the simulation.
    """
    # Check if this is a VecEnv (used for framestack)
    from stable_baselines3.common.vec_env import VecEnv
    is_vec_env = isinstance(env, VecEnv)
    
    obs = env.reset()
    
    # VecEnv returns obs without info dict in reset
    if is_vec_env:
        # VecEnv returns array with shape (n_envs, *obs_shape)
        # We use single environment, so extract first element
        obs_flat = obs[0] if len(obs.shape) > 1 else obs
    else:
        obs, _ = obs if isinstance(obs, tuple) else (obs, {})
        obs_flat = obs
    
    done = False

    # Get unwrapped environment to access true state
    if is_vec_env:
        # For VecEnv, get the underlying environment from the first env
        unwrapped_env = env.envs[0]
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
    else:
        unwrapped_env = env.unwrapped
    
    # Handle partial observability: get full state from unwrapped env if needed
    # Note: obs_flat might be stacked frames for framestack agent
    if hasattr(unwrapped_env, 'current_state'):
        state = unwrapped_env.current_state
        S_init = state.S
        E_init = state.E
        I_init = state.I
        R_init = state.R
    else:
        # Fallback: try to extract from observation
        # For framestack, obs_flat will be stacked, take last frame
        if len(obs_flat) == 3:
            S_init, I_init, R_init = obs_flat[-3:]
            E_init = 0.0
        elif len(obs_flat) == 4:
            S_init, E_init, I_init, R_init = obs_flat[-4:]
        else:
            # Likely stacked observations, take last 4 or 3 elements
            if len(obs_flat) % 4 == 0:
                S_init, E_init, I_init, R_init = obs_flat[-4:]
            else:
                S_init, I_init, R_init = obs_flat[-3:]
                E_init = 0.0
    
    all_S = [S_init]
    all_E = [E_init]
    all_I = [I_init]
    all_R = [R_init]

    actions_taken = []
    timesteps = []
    rewards = []
    observations = []

    current_timestep = 0

    while not done:
        # Store observation (for framestack, this is the stacked observation)
        observations.append(obs_flat if is_vec_env else obs)
        timesteps.append(current_timestep)

        # Predict action (agent handles both VecEnv and regular env observations)
        if is_vec_env:
            action_idx, _ = agent.predict(obs, deterministic=True)
            # VecEnv predict returns array, extract scalar
            action_idx = int(action_idx[0]) if hasattr(action_idx, '__len__') else int(action_idx)
        else:
            action_idx, _ = agent.predict(obs, deterministic=True)

        # Step environment
        if is_vec_env:
            obs, reward, done_array, info_array = env.step([action_idx])
            # Extract from vectorized format
            obs_flat = obs[0]
            reward = float(reward[0])
            done = bool(done_array[0])
            info = info_array[0]
        else:
            obs, reward, done, truncated, info = env.step(action_idx)

        S = info.get("S", [])
        E = info.get("E", [])
        I = info.get("I", [])
        R = info.get("R", [])

        if len(S) > 0:
            all_S.extend(S)
            all_E.extend(E)
            all_I.extend(I)
            all_R.extend(R)

        current_timestep += len(S)

        # Access action_map from unwrapped env (wrapper might not have it)
        action_enum = unwrapped_env.action_map[action_idx]
        actions_taken.append(action_enum)
        rewards.append(reward)

    t = np.arange(len(all_S))
    
    # Use custom agent name if provided
    save_name = agent_name if agent_name else None

    result = SimulationResult(
        agent=agent,
        t=t,
        S=np.array(all_S),
        E=np.array(all_E),
        I=np.array(all_I),
        R=np.array(all_R),
        actions=actions_taken,
        timesteps=timesteps,
        rewards=rewards,
        observations=observations,
        custom_name=save_name,
    )
    
    # Generate file-safe name for saving
    if save_name is None:
        save_name = result.agent_name.lower().replace(" ", "_").replace("-", "_")
    
    # Save logs and plots
    if experiment_dir is not None:
        # Save to experiment directory structure
        log_path = experiment_dir.get_log_path(save_name)
        plot_path = experiment_dir.get_plot_path(f"{save_name}_seir.png")
    else:
        # Fallback to default locations (for backwards compatibility)
        log_path = os.path.join("logs", f"{save_name}.txt")
        os.makedirs("results", exist_ok=True)
        plot_path = os.path.join("results", f"{save_name}_seir.png")
    
    log_results(result, log_path=str(log_path))
    plot_single_result(result, save_path=str(plot_path))

    return result
