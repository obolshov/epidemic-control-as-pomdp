import argparse

from src.simulation import Simulation
from src.agents import StaticAgent, RandomAgent, MyopicMaximizer, InterventionAction
from src.utils import plot_comparison, log_simulation_results
from src.config import get_config


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

    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.SEVERE),
        MyopicMaximizer(config),
    ]

    results = []
    for agent in agents:
        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()
        results.append(result)

    plot_comparison(results, save_path="results/comparison_plot.png")
    log_simulation_results(results, log_dir="logs")
