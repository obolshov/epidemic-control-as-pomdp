import numpy as np

from src.agents import Agent
from src.env import EpidemicEnv, SimulationResult
from src.utils import log_results, plot_single_result


def run_agent(agent: Agent, env: EpidemicEnv) -> SimulationResult:
    """
    Runs a simulation for a single agent.

    :param agent: The agent to evaluate.
    :param env: The environment to run the simulation in.
    :return: A SimulationResult object containing the results of the simulation.
    """
    obs, _ = env.reset()
    done = False

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

        action_enum = env.action_map[action_idx]
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

    log_results(result, log_dir="logs")
    plot_single_result(result, save_path=f"results/{result.agent_name}.png")

    return result
