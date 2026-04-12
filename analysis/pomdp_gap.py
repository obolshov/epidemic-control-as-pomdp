"""POMDP Gap Plot: agent performance vs. partial observability.

Generates a 3-panel figure showing how performance degrades as partial
observability increases, and whether memory-based agents resist that
degradation.

Usage:
    python -m analysis.pomdp_gap
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from analysis.data import AnalysisRun, load_analysis
from src.utils import _save_or_show

SCENARIO_ORDER = ["mdp", "incompleteness", "incompleteness_and_noise", "pomdp"]
SCENARIO_LABELS = ["MDP", "+ Incompleteness", "+ Noise", "POMDP"]

AGENTS = ["ppo_baseline", "ppo_framestack", "ppo_recurrent"]
AGENT_LABELS = {
    "ppo_baseline": "PPO (no memory)",
    "ppo_framestack": "PPO + FrameStack",
    "ppo_recurrent": "RecurrentPPO",
}

METRICS = [
    ("cross_seed_mean_reward", "cross_seed_se_reward", "Total Reward"),
    ("cross_seed_mean_total_infected", "cross_seed_se_total_infected", "Total Infected"),
    ("cross_seed_mean_total_stringency", "cross_seed_se_total_stringency", "Total Stringency"),
]


def plot_pomdp_gap(
    runs: dict[str, AnalysisRun],
    save_path: Optional[str] = None,
) -> None:
    """Generate a 3-panel POMDP gap plot.

    Args:
        runs: Ordered dict from load_analysis("pomdp_gap").
        save_path: Path to save the figure. If None, displays interactively.
    """
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    agent_colors = {agent: colors[i] for i, agent in enumerate(AGENTS)}
    x = list(range(len(SCENARIO_ORDER)))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    for panel_idx, (mean_key, se_key, ylabel) in enumerate(METRICS):
        ax = axes[panel_idx]
        for agent_key in AGENTS:
            means = [
                runs[s].agent_metrics(agent_key)[mean_key]
                for s in SCENARIO_ORDER
            ]
            sds = [
                runs[s].agent_metrics(agent_key)[se_key]
                for s in SCENARIO_ORDER
            ]
            ax.errorbar(
                x,
                means,
                yerr=sds,
                marker="o",
                capsize=4,
                label=AGENT_LABELS[agent_key],
                color=agent_colors[agent_key],
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(SCENARIO_LABELS)
    axes[-1].set_xlabel("Scenario (increasing partial observability →)")

    fig.suptitle("POMDP Gap: Agent Performance vs. Observability", fontsize=14)
    plt.tight_layout()

    _save_or_show(save_path)


if __name__ == "__main__":
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    runs = load_analysis("pomdp_gap")

    plot_pomdp_gap(runs, save_path=str(output_dir / "pomdp_gap_plot.png"))
    print(f"Plot saved to {output_dir / 'pomdp_gap_plot.png'}")
