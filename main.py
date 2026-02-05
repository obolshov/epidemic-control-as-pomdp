import argparse
import os
from typing import List

from stable_baselines3 import PPO

from src.agents import (
    Agent,
    InterventionAction,
    RandomAgent,
    StaticAgent,
    ThresholdAgent,
)
from src.config import DefaultConfig, get_config
from src.env import EpidemicEnv, SimulationResult
from src.evaluation import run_agent
from src.train import train_ppo_agent
from src.wrappers import EpidemicObservationWrapper
from src.utils import (
    plot_all_results,
    plot_learning_curve,
    get_timestamped_results_dir,
)


def train_and_plot_ppo(config: DefaultConfig, results_dir: str) -> None:
    """Trains the PPO agent and plots the learning curve."""
    print("Training PPO agent...")
    train_ppo_agent(EpidemicEnv, config, log_dir="logs/ppo", total_timesteps=50000)
    plot_learning_curve(
        log_folder="logs/ppo", save_path=os.path.join(results_dir, "ppo_learning_curve.png")
    )


def load_ppo_agent(config: DefaultConfig) -> PPO:
    """
    Loads a trained PPO agent based on observability mode.
    
    Args:
        config: Configuration object containing include_exposed flag.
        
    Returns:
        Loaded PPO agent or None if model not found.
    """
    # Determine model path based on observability mode
    if config.include_exposed:
        model_path = "logs/ppo/ppo_model_full_obs.zip"
    else:
        model_path = "logs/ppo/ppo_model_partial_obs.zip"
    
    if os.path.exists(model_path):
        print(f"Loading PPO agent from {model_path}...")
        return PPO.load(model_path)
    else:
        obs_mode = "full observability" if config.include_exposed else "partial observability"
        print(f"PPO model for {obs_mode} not found at {model_path}. Run with --train_ppo to train it.")
        return None


def setup_agents(config: DefaultConfig) -> List[Agent]:
    """Initializes and returns the list of agents to evaluate."""
    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.MILD),
        StaticAgent(InterventionAction.MODERATE),
        StaticAgent(InterventionAction.SEVERE),
        RandomAgent(),
        ThresholdAgent(config),
    ]

    ppo_agent = load_ppo_agent(config)
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
        "--train-ppo",
        action="store_true",
        help="Train PPO agent before running simulation",
    )
    parser.add_argument(
        "--partial-obs",
        "--no-exposed",
        dest="include_exposed",
        action="store_false",
        help="Enable partial observability by masking E (Exposed) compartment",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    config.include_exposed = args.include_exposed

    # Create timestamped results directory
    results_dir = get_timestamped_results_dir()
    print(f"Results will be saved to: {results_dir}")

    if args.train_ppo:
        train_and_plot_ppo(config, results_dir)

    env = EpidemicEnv(config)
    
    # Apply POMDP wrapper if partial observability is enabled
    if not config.include_exposed:
        env = EpidemicObservationWrapper(env, include_exposed=False)
    
    agents = setup_agents(config)

    results: List[SimulationResult] = []

    print("\nStarting simulations...")
    for agent in agents:
        print(f"Running simulation for agent: {agent.__class__.__name__}")

        result = run_agent(agent, env, results_dir=results_dir)
        results.append(result)

    print("\nPlotting results...")
    plot_all_results(results, save_path=os.path.join(results_dir, "all_results.png"))
    print(f"Done! Results saved to '{results_dir}' directory.")


if __name__ == "__main__":
    main()
