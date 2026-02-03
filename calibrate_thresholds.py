"""
Script for calibrating ThresholdAgent thresholds to match PPO agent performance.

This script performs grid search over threshold combinations and finds the optimal
thresholds that minimize the difference in total_reward and peak_infected metrics
compared to a trained PPO agent.
"""

import os
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.config import DefaultConfig, get_config
from src.agents import ThresholdAgent
from src.env import EpidemicEnv


def evaluate_agent(
    agent, env: EpidemicEnv
) -> Dict[str, float]:
    """
    Evaluate an agent and return metrics.
    
    Args:
        agent: The agent to evaluate.
        env: The environment to run simulations in.
        
    Returns:
        Dictionary with total_reward and peak_infected.
    """
    obs, _ = env.reset()
    done = False
    
    all_I = []
    rewards = []
    
    while not done:
        action_idx, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action_idx)
        
        I = info.get("I", [])
        if len(I) > 0:
            all_I.extend(I)
        rewards.append(reward)
    
    return {
        "total_reward_mean": sum(rewards),
        "total_reward_std": 0.0,
        "peak_infected_mean": max(all_I) if len(all_I) > 0 else 0.0,
        "peak_infected_std": 0.0,
    }


def get_ppo_metrics(
    ppo_agent: PPO, env: EpidemicEnv
) -> Dict[str, float]:
    """
    Get metrics for PPO agent.
    
    Args:
        ppo_agent: Trained PPO agent.
        env: The environment.
        
    Returns:
        Dictionary with metrics.
    """
    return evaluate_agent(ppo_agent, env)


def grid_search_thresholds(
    config: DefaultConfig,
    env: EpidemicEnv,
    ppo_metrics: Dict[str, float],
    threshold_ranges: List[Tuple[float, float, int]] = None,
) -> Tuple[List[float], Dict[str, float], List[Dict]]:
    """
    Perform grid search over threshold combinations.
    
    Args:
        config: Configuration object.
        env: The environment.
        ppo_metrics: Target metrics from PPO agent.
        threshold_ranges: List of (min, max, n_points) for each threshold.
                         Default: [(0.0001, 0.01, 20), (0.001, 0.05, 20), (0.01, 0.1, 20)]
        
    Returns:
        Tuple of (best_thresholds, best_metrics, all_results).
    """
    if threshold_ranges is None:
        # Default ranges: [low, medium, high] thresholds
        threshold_ranges = [
            (0.0001, 0.01, 15),   # First threshold: 0.0001 to 0.01
            (0.001, 0.05, 15),    # Second threshold: 0.001 to 0.05
            (0.01, 0.1, 15),      # Third threshold: 0.01 to 0.1
        ]
    
    # Generate threshold grids
    threshold_grids = [
        np.linspace(min_val, max_val, n_points)
        for min_val, max_val, n_points in threshold_ranges
    ]
    
    # Filter to ensure thresholds are in ascending order
    all_combinations = []
    for t1 in threshold_grids[0]:
        for t2 in threshold_grids[1]:
            for t3 in threshold_grids[2]:
                if t1 < t2 < t3:
                    all_combinations.append([t1, t2, t3])
    
    print(f"Testing {len(all_combinations)} threshold combinations...")
    
    best_thresholds = None
    best_score = float("inf")
    best_metrics = None
    all_results = []
    
    for idx, thresholds in enumerate(all_combinations):
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(all_combinations)} combinations tested")
        
        # Create agent with these thresholds
        config.thresholds = thresholds
        agent = ThresholdAgent(config)
        
        # Evaluate agent
        metrics = evaluate_agent(agent, env)
        
        # Calculate score: weighted combination of differences
        # Normalize differences by PPO metrics to make them comparable
        reward_diff = abs(metrics["total_reward_mean"] - ppo_metrics["total_reward_mean"])
        reward_diff_norm = reward_diff / (abs(ppo_metrics["total_reward_mean"]) + 1e-6)
        
        peak_diff = abs(metrics["peak_infected_mean"] - ppo_metrics["peak_infected_mean"])
        peak_diff_norm = peak_diff / (ppo_metrics["peak_infected_mean"] + 1e-6)
        
        # Combined score (equal weight for both metrics)
        score = reward_diff_norm + peak_diff_norm
        
        result = {
            "thresholds": thresholds.copy(),
            "metrics": metrics,
            "score": score,
            "reward_diff": reward_diff,
            "peak_diff": peak_diff,
        }
        all_results.append(result)
        
        if score < best_score:
            best_score = score
            best_thresholds = thresholds.copy()
            best_metrics = metrics
    
    return best_thresholds, best_metrics, all_results


def plot_calibration_results(
    all_results: List[Dict],
    best_thresholds: List[float],
    best_metrics: Dict[str, float],
    ppo_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """
    Plot calibration results showing threshold search space and best solution.
    
    Args:
        all_results: List of all evaluation results.
        best_thresholds: Best threshold combination found.
        best_metrics: Metrics achieved with best thresholds.
        ppo_metrics: Target PPO metrics.
        save_path: Path to save the plot.
    """
    # Extract data for plotting
    scores = [r["score"] for r in all_results]
    reward_diffs = [r["reward_diff"] for r in all_results]
    peak_diffs = [r["peak_diff"] for r in all_results]
    
    thresholds_1 = [r["thresholds"][0] for r in all_results]
    thresholds_2 = [r["thresholds"][1] for r in all_results]
    thresholds_3 = [r["thresholds"][2] for r in all_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score distribution
    ax = axes[0, 0]
    ax.hist(scores, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(min(scores), color="red", linestyle="--", linewidth=2, label="Best score")
    ax.set_xlabel("Combined Score (lower is better)")
    ax.set_ylabel("Frequency")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reward difference vs Peak difference
    ax = axes[0, 1]
    scatter = ax.scatter(
        reward_diffs,
        peak_diffs,
        c=scores,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    # Highlight best solution
    best_idx = np.argmin(scores)
    ax.scatter(
        reward_diffs[best_idx],
        peak_diffs[best_idx],
        color="red",
        marker="*",
        s=300,
        edgecolors="black",
        linewidth=1.5,
        label="Best thresholds",
        zorder=10,
    )
    ax.set_xlabel("Total Reward Difference")
    ax.set_ylabel("Peak Infected Difference")
    ax.set_title("Reward vs Peak Infected Differences")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Combined Score")
    
    # Plot 3: Threshold 1 vs Threshold 2 (colored by score)
    ax = axes[1, 0]
    scatter = ax.scatter(
        thresholds_1,
        thresholds_2,
        c=scores,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.scatter(
        best_thresholds[0],
        best_thresholds[1],
        color="red",
        marker="*",
        s=300,
        edgecolors="black",
        linewidth=1.5,
        label="Best thresholds",
        zorder=10,
    )
    ax.set_xlabel("Threshold 1")
    ax.set_ylabel("Threshold 2")
    ax.set_title("Threshold 1 vs Threshold 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Combined Score")
    
    # Plot 4: Comparison of metrics (with dual y-axis)
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    ppo_reward = ppo_metrics["total_reward_mean"]
    ppo_peak = ppo_metrics["peak_infected_mean"]
    best_reward = best_metrics["total_reward_mean"]
    best_peak = best_metrics["peak_infected_mean"]
    
    x_reward = 0
    x_peak = 1
    width = 0.35
    
    # Plot rewards on left y-axis
    bars1 = ax.bar(x_reward - width / 2, ppo_reward, width, label="PPO Agent", alpha=0.8, color="blue")
    bars2 = ax.bar(x_reward + width / 2, best_reward, width, label="ThresholdAgent", alpha=0.8, color="green")
    
    # Plot peak infected on right y-axis
    bars3 = ax2.bar(x_peak - width / 2, ppo_peak, width, label="PPO Agent", alpha=0.8, color="lightblue")
    bars4 = ax2.bar(x_peak + width / 2, best_peak, width, label="ThresholdAgent", alpha=0.8, color="lightgreen")
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Total Reward", color="blue", fontweight="bold")
    ax2.set_ylabel("Peak Infected", color="red", fontweight="bold")
    ax.set_title("PPO vs Calibrated ThresholdAgent")
    ax.set_xticks([x_reward, x_peak])
    ax.set_xticklabels(["Total Reward", "Peak Infected"])
    ax.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    
    # Combine legends (avoid duplicates)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Use only first set of labels since they're the same
    ax.legend([lines1[0], lines1[1]], ["PPO Agent", "ThresholdAgent"], loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Calibration plot saved to: {save_path}")


def main():
    # Fixed configuration
    ppo_model_path = "logs/ppo/ppo_model.zip"
    output_dir = "calibration_results"
    
    # Load configuration
    config = get_config("default")
    
    # Check if PPO model exists
    if not os.path.exists(ppo_model_path):
        print(f"Error: PPO model not found at {ppo_model_path}")
        print("Please train a PPO agent first using: python main.py --train-ppo")
        return
    
    # Create environment (MDP mode, no wrapper)
    env = EpidemicEnv(config)
    
    # Load PPO agent
    print(f"Loading PPO agent from {ppo_model_path}...")
    ppo_agent = PPO.load(ppo_model_path, env=env)
    
    # Get PPO metrics
    print("Evaluating PPO agent...")
    ppo_metrics = get_ppo_metrics(ppo_agent, env)
    print(f"\nPPO Agent Metrics:")
    print(f"  Total Reward: {ppo_metrics['total_reward_mean']:.2f}")
    print(f"  Peak Infected: {ppo_metrics['peak_infected_mean']:.2f}")
    
    # Perform grid search
    print("\nStarting grid search for optimal thresholds...")
    best_thresholds, best_metrics, all_results = grid_search_thresholds(
        config, env, ppo_metrics
    )
    
    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"\nBest Thresholds:")
    print(f"  Threshold 1: {best_thresholds[0]:.6f}")
    print(f"  Threshold 2: {best_thresholds[1]:.6f}")
    print(f"  Threshold 3: {best_thresholds[2]:.6f}")
    
    print(f"\nThresholdAgent Metrics (with best thresholds):")
    print(f"  Total Reward: {best_metrics['total_reward_mean']:.2f}")
    print(f"  Peak Infected: {best_metrics['peak_infected_mean']:.2f}")
    
    print(f"\nDifferences from PPO:")
    reward_diff = best_metrics["total_reward_mean"] - ppo_metrics["total_reward_mean"]
    peak_diff = best_metrics["peak_infected_mean"] - ppo_metrics["peak_infected_mean"]
    print(f"  Total Reward: {reward_diff:+.2f}")
    print(f"  Peak Infected: {peak_diff:+.2f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best thresholds to file
    thresholds_file = os.path.join(output_dir, "best_thresholds.txt")
    with open(thresholds_file, "w") as f:
        f.write("# Best thresholds for ThresholdAgent\n")
        f.write("# Format: [threshold_1, threshold_2, threshold_3]\n")
        f.write(f"thresholds = [{best_thresholds[0]:.6f}, {best_thresholds[1]:.6f}, {best_thresholds[2]:.6f}]\n")
        f.write("\n# PPO Metrics (target):\n")
        f.write(f"ppo_total_reward_mean = {ppo_metrics['total_reward_mean']:.2f}\n")
        f.write(f"ppo_peak_infected_mean = {ppo_metrics['peak_infected_mean']:.2f}\n")
        f.write("\n# ThresholdAgent Metrics (achieved):\n")
        f.write(f"threshold_total_reward_mean = {best_metrics['total_reward_mean']:.2f}\n")
        f.write(f"threshold_peak_infected_mean = {best_metrics['peak_infected_mean']:.2f}\n")
    print(f"\nBest thresholds saved to: {thresholds_file}")
    
    # Save all results to CSV
    import csv
    csv_file = os.path.join(output_dir, "all_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold_1", "threshold_2", "threshold_3",
            "total_reward_mean", "total_reward_std",
            "peak_infected_mean", "peak_infected_std",
            "score", "reward_diff", "peak_diff"
        ])
        for r in all_results:
            writer.writerow([
                r["thresholds"][0], r["thresholds"][1], r["thresholds"][2],
                r["metrics"]["total_reward_mean"], r["metrics"]["total_reward_std"],
                r["metrics"]["peak_infected_mean"], r["metrics"]["peak_infected_std"],
                r["score"], r["reward_diff"], r["peak_diff"]
            ])
    print(f"All results saved to: {csv_file}")
    
    # Plot results
    plot_path = os.path.join(output_dir, "calibration_results.png")
    plot_calibration_results(all_results, best_thresholds, best_metrics, ppo_metrics, plot_path)
    
    print(f"\n{'='*60}")
    print("Calibration complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
