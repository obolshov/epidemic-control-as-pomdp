"""
Script to find the optimal w_S and w_I values that maximize total reward for MyopicMaximizer.
"""

import numpy as np
from src.simulation import Simulation
from src.agents import MyopicMaximizer
from src.config import DefaultConfig


def find_optimal_weights():
    """
    Find the (w_S, w_I) combination that maximizes total reward for MyopicMaximizer.

    :return: Tuple of (optimal_w_S, optimal_w_I, optimal_reward, all_results)
    """
    w_S_values = np.arange(0.25, 4.25, 0.25)
    w_I_values = np.arange(0.25, 4.25, 0.25)
    # w_I_values = [0.0]

    results = []
    best_w_S = None
    best_w_I = None
    best_reward = float("-inf")
    total_simulations = len(w_S_values) * len(w_I_values)
    current_simulation = 0

    print("Searching for optimal w_S and w_I values...")
    print("=" * 80)
    print(
        f"{'w_S':<10} {'w_I':<10} {'Total Reward':<15} {'Status':<20} {'Progress':<10}"
    )
    print("-" * 80)

    for w_S in w_S_values:
        for w_I in w_I_values:
            current_simulation += 1
            progress = f"{current_simulation}/{total_simulations}"

            # Create config with current weight values
            config = DefaultConfig()
            config.w_S = w_S
            config.w_I = w_I

            # Create MyopicMaximizer agent with this config
            agent = MyopicMaximizer(config)

            # Run simulation
            simulation = Simulation(agent=agent, config=config)
            result = simulation.run()

            total_reward = result.total_reward
            results.append(
                {
                    "w_S": w_S,
                    "w_I": w_I,
                    "total_reward": total_reward,
                    "result": result,
                }
            )

            # Check if this is the best so far
            is_best = False
            if total_reward > best_reward:
                best_reward = total_reward
                best_w_S = w_S
                best_w_I = w_I
                is_best = True

            status = "✓ BEST" if is_best else ""
            print(
                f"{w_S:<10.2f} {w_I:<10.2f} {total_reward:<15.4f} {status:<20} {progress:<10}"
            )

    print("=" * 80)
    print(f"\nOptimal w_S: {best_w_S:.2f}")
    print(f"Optimal w_I: {best_w_I:.2f}")
    print(f"Optimal Total Reward: {best_reward:.4f}")

    return best_w_S, best_w_I, best_reward, results


if __name__ == "__main__":
    optimal_w_S, optimal_w_I, optimal_reward, all_results = find_optimal_weights()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Optimal w_S value: {optimal_w_S:.2f}")
    print(f"Optimal w_I value: {optimal_w_I:.2f}")
    print(f"Maximum total reward: {optimal_reward:.4f}")
    print(f"\nTotal simulations run: {len(all_results)}")

    # Show top 10 results
    print("\nTop 10 (w_S, w_I) combinations by total reward:")
    sorted_results = sorted(all_results, key=lambda x: x["total_reward"], reverse=True)
    print(f"{'Rank':<6} {'w_S':<10} {'w_I':<10} {'Total Reward':<15}")
    print("-" * 45)
    for i, res in enumerate(sorted_results[:10], 1):
        marker = "★" if res["w_S"] == optimal_w_S and res["w_I"] == optimal_w_I else " "
        print(
            f"{i:<6} {res['w_S']:<10.2f} {res['w_I']:<10.2f} {res['total_reward']:<15.4f} {marker}"
        )
