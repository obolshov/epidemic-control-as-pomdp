import os
from typing import List, Optional

import matplotlib.pyplot as plt

from .env import SimulationResult


def _plot_sir_curves(ax, result: SimulationResult, title: str = None) -> None:
    """
    Helper function to plot SIR curves on a given axes.
    """
    colors = {"S": "blue", "I": "red", "R": "green"}

    ax.plot(result.t, result.S, color=colors["S"], label="Susceptible (S)", linewidth=2)
    ax.plot(result.t, result.I, color=colors["I"], label="Infected (I)", linewidth=2)
    ax.plot(result.t, result.R, color=colors["R"], label="Recovered (R)", linewidth=2)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of people")
    ax.legend()
    ax.grid(True, alpha=0.3)

    info_text = f"Peak I: {result.peak_infected:.1f}\n"
    info_text += f"Total infected: {result.total_infected:.1f}\n"
    info_text += f"Duration: {result.epidemic_duration} days"

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
    Creates a comparison plot of SIR curves from simulation results.

    :param results: List of simulation results to plot
    :param save_path: Optional path to save the plot. If None, displays the plot.
    """
    n_results = len(results)
    if n_results == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    elif n_results == 6:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_results, figsize=(7 * n_results, 6))
        if n_results == 1:
            axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        title = f"{result.agent_name}"
        _plot_sir_curves(ax, result, title)

    plt.tight_layout()
    plt.suptitle(
        "SIR Model with Different Agents",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_single_result(
    result: SimulationResult, title: str = None, save_path: str = None
) -> None:
    """
    Creates a simple plot of a single SIR simulation result.

    :param result: SimulationResult to visualize
    :param title: Optional custom title
    :param save_path: Optional path to save the plot
    """
    if title is None:
        title = f"SIR Model - {result.agent_name}"

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_sir_curves(ax, result, title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def log_results(result: SimulationResult, log_dir: str = "logs") -> None:
    """
    Logs simulation results to text files with table format.

    :param results: List of simulation results to log
    :param log_dir: Directory to save log files (default: "logs")
    """
    os.makedirs(log_dir, exist_ok=True)

    safe_name = result.agent_name.replace(" ", "_").replace("-", "")
    log_path = os.path.join(log_dir, f"{safe_name}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Simulation Log: {result.agent_name}\n")
        f.write("=" * 90 + "\n\n")

        header = (
            f"{'Day':<8} {'S':<14} {'I':<14} {'R':<14} {'Reward':<12} {'Action':<15}\n"
        )
        f.write(header)
        f.write("-" * 90 + "\n")

        for i, (day, obs, action, reward) in enumerate(
            zip(
                result.timesteps,
                result.observations,
                result.actions,
                result.rewards,
            )
        ):
            S, I, R = obs
            row = f"{day:<8} {S:<14.2f} {I:<14.2f} {R:<14.2f} {reward:<12.4f} {action.name:<15}\n"
            f.write(row)

        f.write("\n" + "=" * 90 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"  Peak Infected: {result.peak_infected:.2f}\n")
        f.write(f"  Total Infected: {result.total_infected:.2f}\n")
        f.write(f"  Epidemic Duration: {result.epidemic_duration} days\n")
        f.write(f"  Total Reward: {result.total_reward:.4f}\n")
        f.write(f"  Number of Actions: {len(result.actions)}\n")
