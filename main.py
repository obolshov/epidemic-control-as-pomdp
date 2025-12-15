import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from src.agents import (
    Agent,
    InterventionAction,
    MyopicMaximizer,
    RandomAgent,
    StaticAgent,
)
from src.config import get_config
from src.env import EpidemicEnv, SimulationResult
from src.train import train_ppo_agent
from src.utils import (
    log_results,
    plot_all_results,
    plot_learning_curve,
    plot_single_result,
)


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

        action_idx, _ = agent.predict(
            obs, deterministic=True, episode_start=(len(actions_taken) == 0)
        )

        obs, reward, done, truncated, info = env.step(action_idx)

        S = info.get("S", [])
        I = info.get("I", [])
        R = info.get("R", [])

        if len(S) > 0:
            all_S.extend(S)
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
    parser.add_argument(
        "--train_ppo",
        action="store_true",
        help="Train PPO agent",
    )
    args = parser.parse_args()

    config = get_config(args.config)

    if args.train_ppo:
        print("Training PPO agent...")
        # Train for enough steps to see some learning.
        # Simulation is 200 days, action interval 5 days -> 40 steps per episode.
        # 50k steps is ~1250 episodes.
        train_ppo_agent(EpidemicEnv, config, log_dir="logs/ppo", total_timesteps=50000)
        plot_learning_curve(
            log_folder="logs/ppo", save_path="results/ppo_learning_curve.png"
        )

    env = EpidemicEnv(config)

    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.MILD),
        StaticAgent(InterventionAction.MODERATE),
        StaticAgent(InterventionAction.SEVERE),
        RandomAgent(),
        MyopicMaximizer(config),
    ]

    # Load PPO agent if model exists
    ppo_model_path = "logs/ppo/ppo_model.zip"
    if os.path.exists(ppo_model_path):
        print("Loading PPO agent...")
        ppo_agent = PPO.load(ppo_model_path)
        agents.append(ppo_agent)
    else:
        print("PPO model not found. Run with --train_ppo to train it.")

    results = []
    for agent in agents:
        print(f"Running {agent.__class__.__name__}...")
        result = run_agent(agent, env)
        results.append(result)

    plot_all_results(results, save_path="results/all_results.png")
    print("Done!")
