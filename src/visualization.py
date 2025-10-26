import matplotlib.pyplot as plt
from typing import List
from .simulation import SimulationResult


def plot_comparison(results: List[SimulationResult]) -> None:
    """
    Creates a comparison plot of SIR curves from simulation results.
    """
    colors = {"S": "blue", "I": "red", "R": "green"}

    n_results = len(results)
    if n_results == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_results, figsize=(7 * n_results, 6))
        if n_results == 1:
            axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]

        ax.plot(
            result.t, result.S, color=colors["S"], label="Susceptible (S)", linewidth=2
        )
        ax.plot(
            result.t, result.I, color=colors["I"], label="Infected (I)", linewidth=2
        )
        ax.plot(
            result.t, result.R, color=colors["R"], label="Recovered (R)", linewidth=2
        )

        title = f"{result.agent_name}"

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Number of people")
        ax.legend()
        ax.grid(True, alpha=0.3)

        peak_infected_text = f"Peak I: {result.peak_infected:.1f}"
        ax.text(
            0.98,
            0.98,
            peak_infected_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        total_reward_text = f"Total Reward: {result.total_reward:.2f}"
        ax.text(
            0.98,
            0.92,
            total_reward_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        for timestep in result.action_timesteps[1:]:
            ax.axvline(timestep, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.suptitle(
        "SIR Model with Different Agents",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )

    plt.show()


def plot_single_simulation(result: SimulationResult, title: str = None) -> None:
    """
    Creates a simple plot of a single SIR simulation result.

    :param result: SimulationResult to visualize
    :param title: Optional custom title
    """
    if title is None:
        title = f"SIR Model - {result.agent_name}"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.t, result.S, "b", label="Susceptible (S)", linewidth=2)
    ax.plot(result.t, result.I, "r", label="Infected (I)", linewidth=2)
    ax.plot(result.t, result.R, "g", label="Recovered (R)", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of people")
    ax.legend()
    ax.grid(True, alpha=0.3)

    info_text = f"Peak I: {result.peak_infected:.1f}\n"
    info_text += f"Total infected: {result.total_infected:.1f}\n"
    info_text += f"Duration: {result.epidemic_duration} days"

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

    for timestep in result.action_timesteps[1:]:
        ax.axvline(timestep, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    plt.show()
