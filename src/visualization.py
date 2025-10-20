import matplotlib.pyplot as plt
from typing import List
from .simulation import SimulationResult


def plot_comparison(results: List[SimulationResult]) -> None:
    """
    Creates a comparison plot of SIR curves from simulation results.
    """
    colors = {"S": "blue", "I": "red", "R": "green"}

    # Create subplots
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

        # Plot SIR curves
        ax.plot(
            result.t, result.S, color=colors["S"], label="Susceptible (S)", linewidth=2
        )
        ax.plot(
            result.t, result.I, color=colors["I"], label="Infected (I)", linewidth=2
        )
        ax.plot(
            result.t, result.R, color=colors["R"], label="Recovered (R)", linewidth=2
        )

        ax.set_title(
            f"{result.action.name.capitalize()} Intervention (beta = {result.beta:.3f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Number of people")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add info box
        ax.text(
            0.98,
            0.98,
            f"Peak I: {result.peak_infected:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.suptitle(
        "SIR Model with Different Intervention Strategies",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )

    plt.show()


def plot_single_sir(result: SimulationResult, title: str = None) -> None:
    """
    Creates a simple plot of a single SIR simulation result.

    :param result: SimulationResult to visualize
    :param title: Optional custom title
    :return: matplotlib figure
    """
    if title is None:
        title = f"SIR Model - {result.action.name.capitalize()} Intervention"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.t, result.S, "b", label="Susceptible (S)", linewidth=2)
    ax.plot(result.t, result.I, "r", label="Infected (I)", linewidth=2)
    ax.plot(result.t, result.R, "g", label="Recovered (R)", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of people")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add summary statistics
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

    plt.show()
