import matplotlib.pyplot as plt
from src.actions import InterventionAction
from src.simulation import run_simulation
from src.agents import StaticAgent
from src.sir import EpidemicState
from src.visualization import plot_comparison


if __name__ == "__main__":
    N = 1000  # Total population
    I0, R0 = 1, 0  # Initial infected and recovered
    S0 = N - I0 - R0
    beta_0 = 0.4  # Base transmission rate
    gamma = 0.1  # Recovery rate
    days = 160  # Simulation days
    action_interval = 7  # Days between decisions

    print("\nRunning simulation...")
    print(f"Static agents will be evaluated every {action_interval} days\n")

    initial_state = EpidemicState(N=N, S=S0, I=I0, R=R0)

    static_agents = [StaticAgent(action) for action in InterventionAction]

    results = []
    for static_agent in static_agents:
        result = run_simulation(
            agent=static_agent,
            initial_state=initial_state,
            beta_0=beta_0,
            gamma=gamma,
            total_days=days,
            action_interval=action_interval,
        )
        results.append(result)

    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(f"\nParameters: N={N}, I0={I0}, R0={R0}, beta_0={beta_0}, gamma={gamma}")
    print(f"Simulation period: {days} days")
    print(f"Action interval: {action_interval} days\n")

    print(
        f"{'Agent':<20} {'Peak I':<10} {'Total Inf':<12} {'Duration':<10} {'Actions':<10}"
    )
    print("-" * 80)
    for result in results:
        num_decisions = len(result.actions)
        print(
            f"{result.agent_name:<20} "
            f"{result.peak_infected:<10.1f} "
            f"{result.total_infected:<12.1f} "
            f"{result.epidemic_duration:<10d} "
            f"{num_decisions:<10d}"
        )

    print("=" * 80 + "\n")

    plot_comparison(results)
