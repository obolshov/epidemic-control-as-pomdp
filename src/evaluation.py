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
    :param env: The environment to run the simulation in.
    :param experiment_dir: ExperimentDirectory for saving outputs. If None, saves to default locations.
    :param agent_name: Override name for saving files. If None, uses agent's class name.
    :return: A SimulationResult object containing the results of the simulation.
    """
    obs, _ = env.reset()
    done = False

    # Handle partial observability: get full state from unwrapped env if needed
    # If wrapper is applied, obs might be [S, I, R] instead of [S, E, I, R]
    if len(obs) == 3:
        # E is masked, get full state from unwrapped environment
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, 'current_state'):
            state = unwrapped_env.current_state
            S_init = state.S
            E_init = state.E
            I_init = state.I
            R_init = state.R
        else:
            # Fallback: assume E=0 if we can't access it
            S_init, I_init, R_init = obs
            E_init = 0.0
    else:
        # Full observability: obs is [S, E, I, R]
        S_init, E_init, I_init, R_init = obs
    
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
        observations.append(obs)
        timesteps.append(current_timestep)

        action_idx, _ = agent.predict(obs, deterministic=True)

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
        unwrapped_env = env.unwrapped
        action_enum = unwrapped_env.action_map[action_idx]
        actions_taken.append(action_enum)
        rewards.append(reward)

    t = np.arange(len(all_S))

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
    )

    # Use custom agent name if provided
    save_name = agent_name if agent_name else result.agent_name.lower().replace(" ", "_").replace("-", "_")
    
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
