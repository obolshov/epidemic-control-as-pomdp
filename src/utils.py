from math import ceil
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter

from .env import SimulationResult


def _plot_seir_curves(ax, result: SimulationResult, title: str = None) -> None:
    """
    Helper function to plot SEIR curves on a given axes.
    """
    colors = {"S": "blue", "E": "orange", "I": "red", "R": "green"}

    ax.plot(result.t, result.S, color=colors["S"], label="Susceptible (S)", linewidth=2)
    ax.plot(result.t, result.E, color=colors["E"], label="Exposed (E)", linewidth=2)
    ax.plot(result.t, result.I, color=colors["I"], label="Infected (I)", linewidth=2)
    ax.plot(result.t, result.R, color=colors["R"], label="Recovered (R)", linewidth=2)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of people")
    ax.legend()
    ax.grid(True, alpha=0.3)

    info_text = f"Peak I: {result.peak_infected:.1f}\n"
    info_text += f"Total infected: {result.total_infected:.1f}"

    if hasattr(result, "total_reward"):
        info_text += f"\nTotal Reward: {result.total_reward:.2f}"

    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )

    for timestep in result.timesteps[1:]:
        ax.axvline(timestep, color="gray", linestyle="--", alpha=0.3, linewidth=1)


def plot_all_results(
    results: List[SimulationResult], save_path: Optional[str] = None
) -> None:
    """
    Creates a comparison plot of SEIR curves from simulation results.
    
    Dynamically adjusts layout based on number of agents:
    - 1-5 agents: Single row
    - 6+ agents: Two rows

    :param results: List of simulation results to plot
    :param save_path: Optional path to save the plot. If None, displays the plot.
    """
    num_agents = len(results)
    
    if num_agents == 0:
        print("Warning: No results to plot")
        return
    
    # Determine layout: prefer single row for 1-5 agents
    if num_agents <= 5:
        nrows = 1
        ncols = num_agents
        figsize = (7 * num_agents, 6)
    else:
        # Two rows for 6+ agents
        nrows = 2
        ncols = ceil(num_agents / 2)
        figsize = (7 * ncols, 12)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle single plot case
    if num_agents == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each result
    for idx, result in enumerate(results):
        ax = axes[idx]
        title = f"{result.agent_name}"
        _plot_seir_curves(ax, result, title)
    
    # Hide extra subplots if any (for 2-row layouts with odd numbers)
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


def plot_single_result(
    result: SimulationResult, title: str = None, save_path: str = None
) -> None:
    """
    Creates a simple plot of a single SEIR simulation result.
    
    Saves with consistent naming: {agent_name}_seir.png

    :param result: SimulationResult to visualize
    :param title: Optional custom title
    :param save_path: Optional path to save the plot
    """
    if title is None:
        title = f"{result.agent_name}"

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_seir_curves(ax, result, title)

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
    """
    Logs simulation results to text files with table format.
    
    Saves with consistent naming: {agent_name}.txt

    :param result: SimulationResult to log
    :param log_path: Full path to save log file
    """
    dir_path = os.path.dirname(log_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Simulation Log: {result.agent_name}\n")
        f.write("=" * 105 + "\n\n")

        header = f"{'Day':<8} {'S':<14} {'E':<14} {'I':<14} {'R':<14} {'Reward':<12} {'Action':<15}\n"
        f.write(header)
        f.write("-" * 105 + "\n")

        for i, (day, action, reward) in enumerate(
            zip(
                result.timesteps,
                result.actions,
                result.rewards,
            )
        ):
            # Use full state from result arrays (not agent observations which may be partial)
            # timesteps[i] corresponds to the day when the decision was made
            # Ensure index is within bounds
            day_idx = min(int(day), len(result.S) - 1) if len(result.S) > 0 else 0
            S = result.S[day_idx]
            E = result.E[day_idx]
            I = result.I[day_idx]
            R = result.R[day_idx]
            row = f"{day:<8} {S:<14.2f} {E:<14.2f} {I:<14.2f} {R:<14.2f} {reward:<12.4f} {action.name:<15}\n"
            f.write(row)

        f.write("\n" + "=" * 105 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Peak Infected: {result.peak_infected:.2f}\n")
        f.write(f"  Total Infected: {result.total_infected:.2f}\n")
        f.write(f"  Total Reward: {result.total_reward:.4f}\n")
        f.write(f"  Number of Actions: {len(result.actions)}\n")


def plot_learning_curve(
    log_folder: str, title: str = "Learning Curve", save_path: Optional[str] = None
) -> None:
    """
    Plot learning curves from SB3 training logs.
    
    Saves with consistent naming: {agent_name}_learning_episodes.png and {agent_name}_learning_timesteps.png
    
    :param log_folder: Path to folder containing monitor logs
    :param title: Title for the plot
    :param save_path: Base path for saving plots (will append _episodes.png and _timesteps.png)
    """
    x_axes = {
        # "episodes": results_plotter.X_EPISODES,
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
    """Plot evaluation reward curves aggregated across seeds with 95% CI.

    For each agent, loads all seed evaluations.npz files, truncates to the
    shortest common timestep range, and plots mean ± 95% CI across seeds.
    Produces one clean curve per agent instead of one per seed.

    Args:
        eval_log_paths_by_agent: Dict mapping agent_name -> list of npz
            directory paths (one per seed).
        title: Plot title.
        save_path: Path to save plot. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for idx, (agent_name, log_dirs) in enumerate(eval_log_paths_by_agent.items()):
        # Load all seeds for this agent
        seed_means: List[np.ndarray] = []
        min_len = None

        for log_dir in log_dirs:
            npz_path = os.path.join(log_dir, "evaluations.npz")
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path)
            # results shape: (n_evals, n_eval_episodes) — mean over episodes
            per_eval_mean = np.mean(data["results"], axis=1)
            seed_means.append((data["timesteps"], per_eval_mean))
            if min_len is None or len(per_eval_mean) < min_len:
                min_len = len(per_eval_mean)

        if not seed_means or min_len == 0:
            print(f"Warning: no eval data found for {agent_name}, skipping.")
            continue

        # Use timesteps from the first seed (deterministic eval_freq → same grid)
        timesteps = seed_means[0][0][:min_len]
        # Stack seed curves truncated to the shortest run (early stopping)
        stacked = np.stack([m[:min_len] for _, m in seed_means], axis=0)
        # stacked shape: (n_seeds, n_evals)

        n_seeds = stacked.shape[0]
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        ci = 1.96 * std_curve / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(std_curve)

        color = colors[idx % len(colors)]
        label = f"{agent_name} (n={n_seeds} seeds)"
        ax.plot(timesteps, mean_curve, linewidth=2, label=label, color=color)
        ax.fill_between(
            timesteps,
            mean_curve - ci,
            mean_curve + ci,
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
