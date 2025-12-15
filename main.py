import argparse
import os
from typing import List

from stable_baselines3 import PPO

from src.agents import (
    Agent,
    InterventionAction,
    MyopicMaximizer,
    RandomAgent,
    StaticAgent,
)
from src.config import DefaultConfig, get_config
from src.env import EpidemicEnv, SimulationResult
from src.evaluation import run_agent
from src.train import train_ppo_agent
from src.utils import (
    plot_all_results,
    plot_learning_curve,
)


def train_and_plot_ppo(config: DefaultConfig) -> None:
    """Trains the PPO agent and plots the learning curve."""
    print("Training PPO agent...")
    train_ppo_agent(EpidemicEnv, config, log_dir="logs/ppo", total_timesteps=50000)
    plot_learning_curve(
        log_folder="logs/ppo", save_path="results/ppo_learning_curve.png"
    )


def load_ppo_agent(model_path: str = "logs/ppo/ppo_model.zip") -> PPO:
    """Loads a trained PPO agent if available."""
    if os.path.exists(model_path):
        print("Loading PPO agent...")
        return PPO.load(model_path)
    else:
        print("PPO model not found. Run with --train_ppo to train it.")
        return None


def setup_agents(config: DefaultConfig) -> List[Agent]:
    """Initializes and returns the list of agents to evaluate."""
    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.MILD),
        StaticAgent(InterventionAction.MODERATE),
        StaticAgent(InterventionAction.SEVERE),
        RandomAgent(),
        MyopicMaximizer(config),
    ]

    ppo_agent = load_ppo_agent()
    if ppo_agent:
        agents.append(ppo_agent)

    return agents


def main():
    parser = argparse.ArgumentParser(description="Epidemic Control Simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Which configuration to use (default: 'default')",
    )
    parser.add_argument(
        "--train_ppo",
        action="store_true",
        help="Train PPO agent before running simulation",
    )
    args = parser.parse_args()

    config = get_config(args.config)

    if args.train_ppo:
        train_and_plot_ppo(config)

    env = EpidemicEnv(config)
    agents = setup_agents(config)

    results: List[SimulationResult] = []

    print("\nStarting simulations...")
    for agent in agents:
        print(f"Running simulation for agent: {agent.__class__.__name__}")

        result = run_agent(agent, env)
        results.append(result)

    print("\nPlotting results...")
    plot_all_results(results, save_path="results/all_results.png")
    print("Done! Results saved to 'results/' directory.")


if __name__ == "__main__":
    main()
