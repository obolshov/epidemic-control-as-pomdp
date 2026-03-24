"""
Script for calibrating ThresholdAgent thresholds to match PPO baseline performance.

Performs grid search over threshold combinations to minimize the combined
normalized difference across four metrics: total_reward, peak_infected,
total_infected, and total_stringency.

PPO is evaluated with proper VecNormalize (loaded from _vecnormalize.pkl).
Results are averaged across all available seed models.
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.agents import ThresholdAgent
from src.config import Config
from src.evaluation import evaluate_agent
from src.results import AggregatedResult


def _get_vecnorm_path(model_path: str) -> str:
    """Derive VecNormalize path from model path (strips .zip, appends _vecnormalize.pkl)."""
    return model_path[: -len(".zip")] + "_vecnormalize.pkl"


def _metrics_from_agg(agg: AggregatedResult) -> Dict[str, float]:
    """Extract flat metrics dict from AggregatedResult."""
    return {
        "total_reward_mean": agg.mean_reward,
        "total_reward_std": float(np.std(agg.episode_rewards)),
        "peak_infected_mean": agg.mean_peak_infected,
        "peak_infected_std": float(np.std(agg.peak_infected_per_episode)),
        "total_infected_mean": agg.mean_total_infected,
        "total_infected_std": float(np.std(agg.total_infected_per_episode)),
        "total_stringency_mean": agg.mean_total_stringency,
        "total_stringency_std": float(np.std(agg.total_stringency_per_episode)),
    }


def evaluate_ppo_baselines(
    model_paths: List[str],
    config: Config,
    eval_seeds: List[int],
) -> Dict[str, float]:
    """Evaluate all PPO seed models with VecNormalize and average results.

    Args:
        model_paths: List of .zip model paths (one per training seed).
        config: Environment configuration.
        eval_seeds: Evaluation seeds (one episode per seed).

    Returns:
        Metrics dict averaged across seed models.
    """
    all_rewards, all_peaks, all_totals, all_stringencies = [], [], [], []

    for path in model_paths:
        vecnorm_path = _get_vecnorm_path(path)
        model = PPO.load(path)
        agg, _ = evaluate_agent(model, config, {}, "ppo_baseline", eval_seeds, vecnorm_path)
        all_rewards.append(agg.mean_reward)
        all_peaks.append(agg.mean_peak_infected)
        all_totals.append(agg.mean_total_infected)
        all_stringencies.append(agg.mean_total_stringency)

    return {
        "total_reward_mean": float(np.mean(all_rewards)),
        "total_reward_std": float(np.std(all_rewards)),
        "peak_infected_mean": float(np.mean(all_peaks)),
        "peak_infected_std": float(np.std(all_peaks)),
        "total_infected_mean": float(np.mean(all_totals)),
        "total_infected_std": float(np.std(all_totals)),
        "total_stringency_mean": float(np.mean(all_stringencies)),
        "total_stringency_std": float(np.std(all_stringencies)),
    }


def grid_search_thresholds(
    config: Config,
    ppo_metrics: Dict[str, float],
    eval_seeds: List[int],
    threshold_ranges: Optional[List[Tuple[float, float, int]]] = None,
) -> Tuple[List[float], Dict[str, float], List[Dict]]:
    """Grid search over threshold combinations to minimize 4-metric score vs PPO.

    Args:
        config: Environment configuration.
        ppo_metrics: Target metrics from evaluate_ppo_baselines().
        eval_seeds: Evaluation seeds (one episode per seed per combination).
        threshold_ranges: List of (min, max, n_points) for each of 3 thresholds.
            Default: [(0.0001, 0.01, 15), (0.001, 0.05, 15), (0.01, 0.1, 15)]

    Returns:
        Tuple of (best_thresholds, best_metrics, all_results).
    """
    if threshold_ranges is None:
        threshold_ranges = [
            (0.0001, 0.01, 15),
            (0.001, 0.05, 15),
            (0.01, 0.1, 15),
        ]

    grids = [np.linspace(lo, hi, n) for lo, hi, n in threshold_ranges]
    all_combinations = [
        [t1, t2, t3]
        for t1 in grids[0]
        for t2 in grids[1]
        for t3 in grids[2]
        if t1 < t2 < t3
    ]
    print(f"Testing {len(all_combinations)} threshold combinations...")

    ppo_reward = ppo_metrics["total_reward_mean"]
    ppo_peak = ppo_metrics["peak_infected_mean"]
    ppo_total = ppo_metrics["total_infected_mean"]
    ppo_stringency = ppo_metrics["total_stringency_mean"]

    best_thresholds = None
    best_score = float("inf")
    best_metrics = None
    all_results = []

    for idx, thresholds in enumerate(all_combinations):
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(all_combinations)} tested...")

        config.thresholds = thresholds
        agent = ThresholdAgent(config)
        agg, _ = evaluate_agent(agent, config, {}, "threshold", eval_seeds, vecnorm_path=None)
        metrics = _metrics_from_agg(agg)

        reward_diff = abs(metrics["total_reward_mean"] - ppo_reward)
        peak_diff = abs(metrics["peak_infected_mean"] - ppo_peak)
        total_infected_diff = abs(metrics["total_infected_mean"] - ppo_total)
        stringency_diff = abs(metrics["total_stringency_mean"] - ppo_stringency)

        score = (
            reward_diff / (abs(ppo_reward) + 1e-6)
            + peak_diff / (ppo_peak + 1e-6)
            + total_infected_diff / (ppo_total + 1e-6)
            + stringency_diff / (ppo_stringency + 1e-6)
        )

        all_results.append({
            "thresholds": list(thresholds),
            "metrics": metrics,
            "score": score,
            "reward_diff": reward_diff,
            "peak_diff": peak_diff,
            "total_infected_diff": total_infected_diff,
            "stringency_diff": stringency_diff,
        })

        if score < best_score:
            best_score = score
            best_thresholds = list(thresholds)
            best_metrics = metrics

    return best_thresholds, best_metrics, all_results


def plot_calibration_results(
    all_results: List[Dict],
    best_thresholds: List[float],
    best_metrics: Dict[str, float],
    ppo_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Plot calibration results: score distribution, diff scatters, metric comparisons.

    Args:
        all_results: All evaluated threshold combinations.
        best_thresholds: Best combination found.
        best_metrics: Metrics for best combination.
        ppo_metrics: Target PPO metrics.
        save_path: Path to save the PNG.
    """
    scores = [r["score"] for r in all_results]
    reward_diffs = [r["reward_diff"] for r in all_results]
    peak_diffs = [r["peak_diff"] for r in all_results]
    total_infected_diffs = [r["total_infected_diff"] for r in all_results]
    stringency_diffs = [r["stringency_diff"] for r in all_results]
    t1s = [r["thresholds"][0] for r in all_results]
    t2s = [r["thresholds"][1] for r in all_results]
    best_idx = int(np.argmin(scores))

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Panel 1: Score distribution
    ax = axes[0, 0]
    ax.hist(scores, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(min(scores), color="red", linestyle="--", linewidth=2, label="Best score")
    ax.set_xlabel("Combined Score (lower is better)")
    ax.set_ylabel("Frequency")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Threshold 1 vs Threshold 2 colored by score
    ax = axes[0, 1]
    sc = ax.scatter(t1s, t2s, c=scores, cmap="viridis", alpha=0.6, s=20, edgecolors="none")
    ax.scatter(best_thresholds[0], best_thresholds[1], color="red", marker="*",
               s=300, edgecolors="black", linewidth=1.5, label="Best", zorder=10)
    ax.set_xlabel("Threshold 1")
    ax.set_ylabel("Threshold 2")
    ax.set_title("Threshold 1 vs 2 (colored by score)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Score")

    # Panel 3: Reward diff vs Peak diff
    ax = axes[1, 0]
    sc = ax.scatter(reward_diffs, peak_diffs, c=scores, cmap="viridis",
                    alpha=0.6, s=20, edgecolors="none")
    ax.scatter(reward_diffs[best_idx], peak_diffs[best_idx], color="red", marker="*",
               s=300, edgecolors="black", linewidth=1.5, label="Best", zorder=10)
    ax.set_xlabel("Total Reward Difference")
    ax.set_ylabel("Peak Infected Difference")
    ax.set_title("Reward vs Peak Infected Differences")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Score")

    # Panel 4: Total infected diff vs Stringency diff
    ax = axes[1, 1]
    sc = ax.scatter(total_infected_diffs, stringency_diffs, c=scores, cmap="viridis",
                    alpha=0.6, s=20, edgecolors="none")
    ax.scatter(total_infected_diffs[best_idx], stringency_diffs[best_idx], color="red",
               marker="*", s=300, edgecolors="black", linewidth=1.5, label="Best", zorder=10)
    ax.set_xlabel("Total Infected Difference")
    ax.set_ylabel("Total Stringency Difference")
    ax.set_title("Total Infected vs Stringency Differences")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Score")

    # Panel 5: Reward + Peak comparison
    ax = axes[2, 0]
    ax2 = ax.twinx()
    width = 0.35
    ax.bar(-width / 2, ppo_metrics["total_reward_mean"], width, label="PPO", alpha=0.8, color="blue")
    ax.bar(width / 2, best_metrics["total_reward_mean"], width, label="Threshold", alpha=0.8, color="green")
    ax2.bar(1 - width / 2, ppo_metrics["peak_infected_mean"], width, alpha=0.8, color="lightblue")
    ax2.bar(1 + width / 2, best_metrics["peak_infected_mean"], width, alpha=0.8, color="lightgreen")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Total Reward", "Peak Infected"])
    ax.set_ylabel("Total Reward", color="blue")
    ax2.set_ylabel("Peak Infected", color="steelblue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax.set_title("PPO vs Calibrated Threshold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 6: Total infected + Stringency comparison
    ax = axes[2, 1]
    ax2 = ax.twinx()
    ax.bar(-width / 2, ppo_metrics["total_infected_mean"], width, alpha=0.8, color="blue")
    ax.bar(width / 2, best_metrics["total_infected_mean"], width, alpha=0.8, color="green")
    ax2.bar(1 - width / 2, ppo_metrics["total_stringency_mean"], width, alpha=0.8, color="lightblue")
    ax2.bar(1 + width / 2, best_metrics["total_stringency_mean"], width, alpha=0.8, color="lightgreen")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Total Infected", "Total Stringency"])
    ax.set_ylabel("Total Infected", color="blue")
    ax2.set_ylabel("Total Stringency", color="steelblue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax.set_title("PPO vs Calibrated Threshold (epidemic size & cost)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ThresholdAgent thresholds to match PPO baseline."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Directory containing ppo_baseline_seed*.zip models. "
            "Defaults to latest experiments/mdp*/weights/ by mtime."
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Episodes per threshold combination (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="calibration_results",
        help="Directory for output files (default: calibration_results)",
    )
    args = parser.parse_args()

    config = Config()

    # Resolve model directory
    model_dir = args.model_dir
    if model_dir is None:
        candidates = glob.glob("experiments/mdp*/weights")
        if not candidates:
            print("Error: no experiments/mdp*/weights/ directory found.")
            print("Pass --model-dir to specify explicitly.")
            return
        model_dir = max(candidates, key=os.path.getmtime)
    print(f"Using model dir: {model_dir}")

    model_paths = sorted(glob.glob(os.path.join(model_dir, "ppo_baseline_seed*.zip")))
    if not model_paths:
        print(f"Error: no ppo_baseline_seed*.zip files found in {model_dir}")
        return
    print(f"Found {len(model_paths)} seed models: {[os.path.basename(p) for p in model_paths]}")

    eval_seeds = list(range(42, 42 + args.n_episodes))

    # Evaluate PPO across all seeds
    print(f"\nEvaluating PPO baseline across {len(model_paths)} seeds × {args.n_episodes} episodes...")
    ppo_metrics = evaluate_ppo_baselines(model_paths, config, eval_seeds)
    print(f"\nPPO Metrics (target):")
    print(f"  Total Reward:      {ppo_metrics['total_reward_mean']:8.2f} ± {ppo_metrics['total_reward_std']:.2f}")
    print(f"  Peak Infected:     {ppo_metrics['peak_infected_mean']:8.1f} ± {ppo_metrics['peak_infected_std']:.1f}")
    print(f"  Total Infected:    {ppo_metrics['total_infected_mean']:8.1f} ± {ppo_metrics['total_infected_std']:.1f}")
    print(f"  Total Stringency:  {ppo_metrics['total_stringency_mean']:8.2f} ± {ppo_metrics['total_stringency_std']:.2f}")

    # Grid search
    print(f"\nStarting grid search (n_episodes={args.n_episodes} per combination)...")
    best_thresholds, best_metrics, all_results = grid_search_thresholds(
        config, ppo_metrics, eval_seeds
    )

    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"\nBest Thresholds: [{best_thresholds[0]:.6f}, {best_thresholds[1]:.6f}, {best_thresholds[2]:.6f}]")
    print(f"\n{'Metric':<22} {'PPO':>10} {'Threshold':>10} {'Diff':>10}")
    print("-" * 54)
    for label, key in [
        ("Total Reward", "total_reward"),
        ("Peak Infected", "peak_infected"),
        ("Total Infected", "total_infected"),
        ("Total Stringency", "total_stringency"),
    ]:
        ppo_val = ppo_metrics[f"{key}_mean"]
        thr_val = best_metrics[f"{key}_mean"]
        print(f"  {label:<20} {ppo_val:>10.2f} {thr_val:>10.2f} {thr_val - ppo_val:>+10.2f}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    thresholds_file = os.path.join(args.output_dir, "best_thresholds.txt")
    with open(thresholds_file, "w") as f:
        f.write("# Best thresholds for ThresholdAgent\n")
        f.write(f"thresholds = [{best_thresholds[0]:.6f}, {best_thresholds[1]:.6f}, {best_thresholds[2]:.6f}]\n\n")
        f.write("# PPO Metrics (target):\n")
        for key in ("total_reward", "peak_infected", "total_infected", "total_stringency"):
            f.write(f"ppo_{key}_mean = {ppo_metrics[f'{key}_mean']:.4f}\n")
        f.write("\n# ThresholdAgent Metrics (achieved):\n")
        for key in ("total_reward", "peak_infected", "total_infected", "total_stringency"):
            f.write(f"threshold_{key}_mean = {best_metrics[f'{key}_mean']:.4f}\n")
    print(f"\nBest thresholds saved to: {thresholds_file}")

    csv_file = os.path.join(args.output_dir, "all_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold_1", "threshold_2", "threshold_3",
            "total_reward_mean", "total_reward_std",
            "peak_infected_mean", "peak_infected_std",
            "total_infected_mean", "total_infected_std",
            "total_stringency_mean", "total_stringency_std",
            "score", "reward_diff", "peak_diff", "total_infected_diff", "stringency_diff",
        ])
        for r in all_results:
            m = r["metrics"]
            writer.writerow([
                r["thresholds"][0], r["thresholds"][1], r["thresholds"][2],
                m["total_reward_mean"], m["total_reward_std"],
                m["peak_infected_mean"], m["peak_infected_std"],
                m["total_infected_mean"], m["total_infected_std"],
                m["total_stringency_mean"], m["total_stringency_std"],
                r["score"], r["reward_diff"], r["peak_diff"],
                r["total_infected_diff"], r["stringency_diff"],
            ])
    print(f"All results saved to: {csv_file}")

    plot_path = os.path.join(args.output_dir, "calibration_results.png")
    plot_calibration_results(all_results, best_thresholds, best_metrics, ppo_metrics, plot_path)

    print(f"\nCalibration complete. Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
