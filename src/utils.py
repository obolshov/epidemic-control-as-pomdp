from math import ceil
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter

from .env import AggregatedResult, SimulationResult


def _plot_aggregated_seir(ax, agg: AggregatedResult, title: Optional[str] = None) -> None:
    """Plot mean +/- SD shaded SEIR curves for an AggregatedResult.

    Args:
        ax: Matplotlib axes.
        agg: AggregatedResult with mean/std arrays.
        title: Optional subplot title.
    """
    colors = {"S": "blue", "E": "orange", "I": "red", "R": "green"}
    labels = {"S": "Susceptible (S)", "E": "Exposed (E)", "I": "Infected (I)", "R": "Recovered (R)"}

    for comp, color in colors.items():
        mean = getattr(agg, f"{comp}_mean")
        std = getattr(agg, f"{comp}_std")
        ax.plot(agg.t, mean, color=color, label=labels[comp], linewidth=2)
        ax.fill_between(agg.t, mean - std, mean + std, color=color, alpha=0.2)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of people")
    ax.legend()
    ax.grid(True, alpha=0.3)

    info_text = (
        f"Reward: {agg.mean_reward:.2f} ± {agg.std_reward:.2f}\n"
        f"Peak I: {agg.mean_peak_infected:.1f} ± {agg.std_peak_infected:.1f}\n"
        f"Total inf: {agg.mean_total_infected:.1f} ± {agg.std_total_infected:.1f}\n"
        f"n = {agg.n_episodes} episodes"
    )

    ax.text(
        0.98, 0.98, info_text,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )


def plot_all_results(
    results: Dict[str, AggregatedResult], save_path: Optional[str] = None
) -> None:
    """Creates a comparison plot of mean +/- SD SEIR curves from aggregated results.

    Dynamically adjusts layout based on number of agents:
    - 1-5 agents: Single row
    - 6+ agents: Two rows

    Args:
        results: Dict mapping agent_name -> AggregatedResult.
        save_path: Optional path to save the plot. If None, displays the plot.
    """
    num_agents = len(results)

    if num_agents == 0:
        print("Warning: No results to plot")
        return

    if num_agents <= 5:
        nrows = 1
        ncols = num_agents
        figsize = (7 * num_agents, 6)
    else:
        nrows = 2
        ncols = ceil(num_agents / 2)
        figsize = (7 * ncols, 12)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if num_agents == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (agent_name, agg) in enumerate(results.items()):
        ax = axes[idx]
        _plot_aggregated_seir(ax, agg, title=agent_name)

    for idx in range(num_agents, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_single_aggregated(
    agg: AggregatedResult, title: Optional[str] = None, save_path: Optional[str] = None
) -> None:
    """Creates a standalone plot of mean +/- SD SEIR curves for one agent.

    Args:
        agg: AggregatedResult to visualize.
        title: Optional custom title (defaults to agent_name).
        save_path: Optional path to save the plot.
    """
    if title is None:
        title = agg.agent_name

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_aggregated_seir(ax, agg, title)

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def log_results(result: SimulationResult, log_path: str) -> None:
    """Logs simulation results to text files with table format.

    Args:
        result: SimulationResult to log.
        log_path: Full path to save log file.
    """
    dir_path = os.path.dirname(log_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Simulation Log: {result.agent_name}\n")
        f.write("=" * 141 + "\n\n")

        header = (
            f"{'Day':<8} {'S':<14} {'E':<14} {'I':<14} {'R':<14} "
            f"{'Reward':<12} {'R_inf':<12} {'R_str':<12} {'R_sw':<12} {'Action':<15}\n"
        )
        f.write(header)
        f.write("-" * 141 + "\n")

        for i, (day, action, reward, rc) in enumerate(
            zip(
                result.timesteps,
                result.actions,
                result.rewards,
                result.reward_components,
            )
        ):
            day_idx = min(int(day), len(result.S) - 1) if len(result.S) > 0 else 0
            S = result.S[day_idx]
            E = result.E[day_idx]
            I = result.I[day_idx]
            R = result.R[day_idx]
            row = (
                f"{day:<8} {S:<14.2f} {E:<14.2f} {I:<14.2f} {R:<14.2f} "
                f"{reward:<12.4f} {rc['reward_infection']:<12.4f} "
                f"{rc['reward_stringency']:<12.4f} {rc['reward_switching']:<12.4f} "
                f"{action.name:<15}\n"
            )
            f.write(row)

        f.write("\n" + "=" * 141 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Peak Infected: {result.peak_infected:.2f}\n")
        f.write(f"  Total Infected: {result.total_infected:.2f}\n")
        f.write(f"  Total Reward: {result.total_reward:.4f}\n")


def plot_learning_curve(
    log_folder: str, title: str = "Learning Curve", save_path: Optional[str] = None
) -> None:
    """Plot learning curves from SB3 training logs.

    Args:
        log_folder: Path to folder containing monitor logs.
        title: Title for the plot.
        save_path: Base path for saving plots.
    """
    x_axes = {
        "timesteps": results_plotter.X_TIMESTEPS,
    }

    for axis_name, axis_code in x_axes.items():
        try:
            results_plotter.plot_results(
                [log_folder],
                num_timesteps=None,
                x_axis=axis_code,
                task_name=f"{title} ({axis_name})",
                figsize=(10, 6),
            )

            if save_path:
                base, ext = os.path.splitext(save_path)
                current_save_path = f"{base}_{axis_name}{ext}"
                dir_path = os.path.dirname(current_save_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                plt.savefig(current_save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
                plt.close()

        except Exception as e:
            print(f"Error plotting {axis_name}: {e}")


def plot_evaluation_curves(
    eval_log_paths_by_agent: Dict[str, List[str]],
    title: str = "Evaluation During Training",
    save_path: Optional[str] = None,
) -> None:
    """Plot evaluation reward curves aggregated across seeds with SD band.

    For each agent, loads all seed evaluations.npz files, truncates to the
    shortest common timestep range, and plots mean ± SD across seeds.

    Args:
        eval_log_paths_by_agent: Dict mapping agent_name -> list of npz
            directory paths (one per seed).
        title: Plot title.
        save_path: Path to save plot. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for idx, (agent_name, log_dirs) in enumerate(eval_log_paths_by_agent.items()):
        seed_means: List[np.ndarray] = []
        min_len = None

        for log_dir in log_dirs:
            npz_path = os.path.join(log_dir, "evaluations.npz")
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            per_eval_mean = np.mean(data["results"], axis=1)
            seed_means.append((data["timesteps"], per_eval_mean))
            if min_len is None or len(per_eval_mean) < min_len:
                min_len = len(per_eval_mean)

        if not seed_means or min_len == 0:
            print(f"Warning: no eval data found for {agent_name}, skipping.")
            continue

        timesteps = seed_means[0][0][:min_len]
        stacked = np.stack([m[:min_len] for _, m in seed_means], axis=0)

        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)

        color = colors[idx % len(colors)]
        n_seeds = stacked.shape[0]
        label = f"{agent_name} (n={n_seeds} seeds)"
        ax.plot(timesteps, mean_curve, linewidth=2, label=label, color=color)
        ax.fill_between(
            timesteps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.25,
            color=color,
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()
