import matplotlib.pyplot as plt
from src.actions import InterventionAction
from src.simulation import SimulationResult
from src.sir import run_sir_model
from src.visualization import plot_comparison


if __name__ == "__main__":
    N = 1000  # Total population
    I0, R0 = 1, 0  # Initial infected and recovered
    beta_0 = 0.4  # Base transmission rate
    gamma = 0.1  # Recovery rate
    days = 160  # Simulation days

    print("\nRunning simulation with different intervention strategies...")

    results = []
    for action in list(InterventionAction):
        beta_t = action.apply_to_beta(beta_0)
        t, S, I, R = run_sir_model(N, I0, R0, beta_t, gamma, days)

        result = SimulationResult(action, beta_t, t, S, I, R)
        results.append(result)

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nParameters: N={N}, I0={I0}, R0={R0}, beta_0={beta_0}, gamma={gamma}")
    print(f"Simulation period: {days} days\n")

    print(
        f"{'Action':<12} {'Beta':<8} {'Peak I':<10} {'Total Inf':<12} {'Duration':<10}"
    )
    print("-" * 60)
    for result in results:
        print(
            f"{result.action.name:<12} "
            f"{result.beta:<8.3f} "
            f"{result.peak_infected:<10.1f} "
            f"{result.total_infected:<12.1f} "
            f"{result.epidemic_duration:<10d}"
        )

    print("=" * 60 + "\n")

    plot_comparison(results)
