import argparse

from src.simulation import Simulation
from src.agents import StaticAgent, InterventionAction
from src.visualization import plot_comparison
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

    print(f"\nRunning simulation with '{args.config}' config...")
    print(f"Static agents will be evaluated every {config.action_interval} days\n")

    static_agents = [StaticAgent(action) for action in InterventionAction]

    results = []
    for static_agent in static_agents:
        simulation = Simulation(agent=static_agent, config=config)
        result = simulation.run()
        results.append(result)

    print("\n" + "=" * 92)
    print("SIMULATION RESULTS")
    print("=" * 92)
    print(
        f"\nParameters: N={config.N}, I0={config.I0}, R0={config.R0}, beta_0={config.beta_0}, gamma={config.gamma}"
    )
    print(f"Simulation period: {config.days} days")
    print(f"Action interval: {config.action_interval} days\n")

    print(
        f"{'Agent':<22} {'Peak I':<10} {'Total Inf':<12} {'Duration':<10} {'Actions':<10} {'Total Reward':<12}"
    )
    print("-" * 92)
    for result in results:
        num_decisions = len(result.actions)
        print(
            f"{result.agent_name:<22} "
            f"{result.peak_infected:<10.1f} "
            f"{result.total_infected:<12.1f} "
            f"{result.epidemic_duration:<10d} "
            f"{num_decisions:<10d} "
            f"{result.total_reward:<12.2f}"
        )

    print("=" * 92 + "\n")

    plot_comparison(results)
