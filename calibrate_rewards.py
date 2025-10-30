"""
Calibration script for reward function parameters (w_I, w_S).

The goal is to find parameters such that for different R0 values, the optimal action
(one that gives the highest total reward) matches the expected action based on:
1. R0 <= 1.0 -> NO action (coefficient 1.0)
2. 1.0 < R0 <= 1.33 -> MILD (coefficient 0.75)
3. 1.33 < R0 <= 2.0 -> MODERATE (coefficient 0.5)
4. R0 > 2.0 -> SEVERE (coefficient 0.25)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from src.simulation import Simulation
from src.agents import StaticAgent, InterventionAction
from src.config import DefaultConfig
from itertools import product

RESULTS_DIR = "results/calibration"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_expected_action(r0: float) -> InterventionAction:
    """
    Returns the expected optimal action for a given R0 value.
    This is the weakest action that brings R0 below 1.
    """
    if r0 <= 1.0:
        return InterventionAction.NO
    elif r0 <= 1.33:
        return InterventionAction.MILD
    elif r0 <= 2.0:
        return InterventionAction.MODERATE
    else:
        return InterventionAction.SEVERE


def run_simulations_for_r0(
    r0: float, gamma: float, w_I: float, w_S: float
) -> Dict[InterventionAction, float]:
    """
    Run simulations for all actions with a given R0 and return total rewards.
    """
    beta_0 = r0 * gamma

    rewards_by_action = {}

    for action in InterventionAction:
        agent = StaticAgent(action)

        # Create config for this simulation
        config = DefaultConfig()
        config.beta_0 = beta_0
        config.gamma = gamma
        config.w_I = w_I
        config.w_S = w_S

        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()
        rewards_by_action[action] = result.total_reward

    return rewards_by_action


def evaluate_parameters(
    w_I: float,
    w_S: float,
    test_r0_values: List[float],
    gamma: float = 0.1,
    verbose: bool = False,
) -> Tuple[int, int, float]:
    """
    Evaluate how well the parameters match the expected optimal actions.

    Returns:
        - Number of correct matches
        - Total number of test cases
        - Accuracy score
    """
    correct = 0
    total = len(test_r0_values)

    if verbose:
        print(f"\nEvaluating: w_I={w_I:.3f}, w_S={w_S:.3f}")
        print("-" * 70)

    for r0 in test_r0_values:
        rewards = run_simulations_for_r0(r0, gamma, w_I, w_S)

        # Find action with highest reward
        optimal_action = max(rewards, key=rewards.get)
        expected_action = get_expected_action(r0)

        is_correct = optimal_action == expected_action
        if is_correct:
            correct += 1

        if verbose:
            status = "✓" if is_correct else "✗"
            print(
                f"R0={r0:.2f}: Expected={expected_action.name:8s}, "
                f"Optimal={optimal_action.name:8s} (reward={rewards[optimal_action]:.2f}) {status}"
            )

    accuracy = correct / total
    if verbose:
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")

    return correct, total, accuracy


def grid_search_calibration(
    w_I_range: np.ndarray,
    w_S_range: np.ndarray,
    test_r0_values: List[float],
    gamma: float = 0.1,
) -> Tuple[Dict, np.ndarray]:
    """
    Perform grid search over parameter space to find best parameters.

    Returns:
        - Best parameters dictionary
        - Accuracy grid for heat map
    """
    best_accuracy = 0
    best_params = None

    # Store accuracy for each (w_I, w_S) combination
    accuracy_grid = np.zeros((len(w_I_range), len(w_S_range)))

    total_combinations = len(w_I_range) * len(w_S_range)
    print(f"Testing {total_combinations} parameter combinations...")
    print(f"w_I: {len(w_I_range)} values, w_S: {len(w_S_range)} values\n")

    for i, (w_I_idx, w_S_idx) in enumerate(
        product(range(len(w_I_range)), range(len(w_S_range)))
    ):
        w_I = w_I_range[w_I_idx]
        w_S = w_S_range[w_S_idx]

        correct, total, accuracy = evaluate_parameters(
            w_I, w_S, test_r0_values, gamma, verbose=False
        )

        # Update accuracy grid
        accuracy_grid[w_I_idx, w_S_idx] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                "w_I": w_I,
                "w_S": w_S,
                "accuracy": accuracy,
            }

        if (i + 1) % 50 == 0 or (i + 1) == total_combinations:
            print(
                f"Progress: {i + 1}/{total_combinations} "
                f"({100 * (i + 1) / total_combinations:.1f}%) - "
                f"Best accuracy so far: {best_accuracy:.1%}"
            )

    return best_params, accuracy_grid


def plot_heatmap(
    accuracy_grid: np.ndarray,
    w_I_range: np.ndarray,
    w_S_range: np.ndarray,
    best_params: Dict,
):
    """
    Create a heat map showing accuracy for different (w_I, w_S) combinations.
    """
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        accuracy_grid,
        xticklabels=[f"{x:.2f}" for x in w_S_range],
        yticklabels=[f"{y:.2f}" for y in w_I_range],
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Accuracy"},
        vmin=0,
        vmax=1,
    )

    plt.xlabel("w_S (Stringency Weight)", fontsize=12)
    plt.ylabel("w_I (Infection Growth Weight)", fontsize=12)
    plt.title(
        f"Calibration Heat Map: Accuracy of Optimal Actions\n"
        f'Best: w_I={best_params["w_I"]:.3f}, w_S={best_params["w_S"]:.3f} '
        f'(Accuracy: {best_params["accuracy"]:.1%})',
        fontsize=14,
        pad=20,
    )

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "calibration_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHeat map saved to: {output_path}")
    plt.show()


def plot_reward_comparison(
    test_r0_values: List[float],
    best_params: Dict,
    gamma: float = 0.1,
):
    """
    Plot reward comparison for different actions across R0 values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Select representative R0 values from different regions
    representative_r0s = [0.8, 1.2, 1.7, 2.5]

    for idx, r0 in enumerate(representative_r0s):
        ax = axes[idx]

        rewards = run_simulations_for_r0(
            r0,
            gamma,
            best_params["w_I"],
            best_params["w_S"],
        )

        actions = list(rewards.keys())
        action_names = [a.name for a in actions]
        reward_values = [rewards[a] for a in actions]

        expected_action = get_expected_action(r0)
        optimal_action = max(rewards, key=rewards.get)

        colors = ["green" if a == expected_action else "lightcoral" for a in actions]
        colors = [
            "darkgreen" if a == optimal_action else c for a, c in zip(actions, colors)
        ]

        bars = ax.bar(
            action_names, reward_values, color=colors, alpha=0.7, edgecolor="black"
        )

        # Highlight optimal
        for i, (action, bar) in enumerate(zip(actions, bars)):
            if action == optimal_action:
                bar.set_linewidth(3)

        ax.set_xlabel("Action", fontsize=11)
        ax.set_ylabel("Total Reward", fontsize=11)
        ax.set_title(
            f"R0 = {r0:.1f} (Expected: {expected_action.name}, Optimal: {optimal_action.name})",
            fontsize=12,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.suptitle(
        f"Reward Comparison Across R0 Values\n"
        f'w_I={best_params["w_I"]:.3f}, w_S={best_params["w_S"]:.3f}',
        fontsize=14,
        y=1.00,
    )
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "reward_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Reward comparison plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # Test R0 values covering all regions
    test_r0_values = [
        0.7,
        0.9,  # Region 1: R0 <= 1.0 -> NO
        1.1,
        1.25,  # Region 2: 1.0 < R0 <= 1.33 -> MILD
        1.5,
        1.8,  # Region 3: 1.33 < R0 <= 2.0 -> MODERATE
        2.5,
        3.0,
        3.5,  # Region 4: R0 > 2.0 -> SEVERE
    ]

    gamma = 0.1

    print("=" * 70)
    print("REWARD FUNCTION CALIBRATION")
    print("=" * 70)
    print("\nExpected optimal actions:")
    for r0 in test_r0_values:
        expected = get_expected_action(r0)
        print(f"  R0 = {r0:.2f} -> {expected.name}")
    print()

    # Define parameter search space
    w_I_range = np.linspace(0.5, 3.0, 11)  # 11 values
    w_S_range = np.linspace(0.1, 2.0, 11)  # 11 values

    # Grid search
    best_params, accuracy_grid = grid_search_calibration(
        w_I_range, w_S_range, test_r0_values, gamma
    )

    print("\n" + "=" * 70)
    print("BEST PARAMETERS FOUND")
    print("=" * 70)
    print(f"w_I:      {best_params['w_I']:.4f}")
    print(f"w_S:      {best_params['w_S']:.4f}")
    print(f"Accuracy: {best_params['accuracy']:.1%}")
    print("=" * 70)

    # Detailed evaluation with best parameters
    print("\n" + "=" * 70)
    print("DETAILED EVALUATION WITH BEST PARAMETERS")
    print("=" * 70)
    evaluate_parameters(
        best_params["w_I"],
        best_params["w_S"],
        test_r0_values,
        gamma,
        verbose=True,
    )

    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_heatmap(accuracy_grid, w_I_range, w_S_range, best_params)
    plot_reward_comparison(test_r0_values, best_params, gamma)

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print("\nYou can now use these parameters in your simulation:")
    print(f"  w_I = {best_params['w_I']:.4f}")
    print(f"  w_S = {best_params['w_S']:.4f}")
    print()
