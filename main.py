import argparse
import numpy as np

from src.config import get_config
from src.env import EpidemicEnv, SimulationResult
from src.agents import (
    StaticAgent,
    RandomAgent,
    MyopicMaximizer,
    InterventionAction,
    Agent,
)
from src.utils import log_simulation_results, plot_single_simulation


def run_agent(agent: Agent, env: EpidemicEnv) -> SimulationResult:
    obs, _ = env.reset()
    done = False

    # Initialize storage for SimulationResult
    S_init, I_init, R_init = obs
    all_S = [S_init]
    all_I = [I_init]
    all_R = [R_init]

    actions_taken = []
    timesteps = []
    rewards = []
    observations = []

    current_timestep = 0

    while not done:
        # Save observation before action
        observations.append(obs)
        timesteps.append(current_timestep)

        action_idx, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action_idx)

        # Retrieve detailed step data from info dict
        S = info.get("S", [])
        I = info.get("I", [])
        R = info.get("R", [])

        # Extend arrays (skip first element as it duplicates the last state of previous step)
        if len(S) > 1:
            all_S.extend(S[1:])
            all_I.extend(I[1:])
            all_R.extend(R[1:])

        current_timestep += (len(S) - 1) if len(S) > 0 else 0

        # Store action and reward
        action_enum = env.action_map[action_idx]
        actions_taken.append(action_enum)
        rewards.append(reward)

    t = np.arange(len(all_S))

    result = SimulationResult(
        agent=agent,
        t=t,
        S=np.array(all_S),
        I=np.array(all_I),
        R=np.array(all_R),
        actions=actions_taken,
        timesteps=timesteps,
        rewards=rewards,
        observations=observations,
    )

    log_simulation_results(result, log_dir="logs")
    plot_single_simulation(result, save_path=f"results/{result.agent_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Which configuration to use",
    )
    args = parser.parse_args()

    config = get_config(args.config)

    env = EpidemicEnv(config)

    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.MILD),
        StaticAgent(InterventionAction.MODERATE),
        StaticAgent(InterventionAction.SEVERE),
        RandomAgent(),
        MyopicMaximizer(config),
    ]

    results = []
    for agent in agents:
        print(f"Running {agent.__class__.__name__}...")
        run_agent(agent, env)

    print("Done!")
