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
from src.utils import log_results, plot_single_result, plot_all_results


def run_agent(agent: Agent, env: EpidemicEnv) -> SimulationResult:
    obs, _ = env.reset()
    done = False

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
        observations.append(obs)
        timesteps.append(current_timestep)

        action_idx, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action_idx)

        S = info.get("S", [])
        I = info.get("I", [])
        R = info.get("R", [])

        if len(S) > 1:
            all_S.extend(S[1:])
            all_I.extend(I[1:])
            all_R.extend(R[1:])

        current_timestep += (len(S) - 1) if len(S) > 0 else 0

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

    log_results(result, log_dir="logs")
    plot_single_result(result, save_path=f"results/{result.agent_name}.png")

    return result


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
        result = run_agent(agent, env)
        results.append(result)

    plot_all_results(results, save_path="results/all_results.png")
    print("Done!")
