"""Distortion Ablation Study: isolated impact of each POMDP distortion group.

Generates a heatmap showing how much each distortion type degrades each agent's
performance relative to the MDP baseline. Identifies which distortion benefits
most from memory.

Usage:
    python -m analysis.distortion_ablation
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis.data import AnalysisRun, load_analysis
from src.utils import _save_or_show

ANALYSIS_NAME = "distortion_ablation"

BASELINE_LABEL = "mdp"
ABLATION_ORDER = ["only_incompleteness", "only_noise", "only_temporal"]
ABLATION_LABELS = ["Incompleteness", "Noise", "Temporal"]

AGENTS = ["ppo_baseline", "ppo_framestack", "ppo_recurrent"]
AGENT_LABELS = {
    "ppo_baseline": "PPO (no memory)",
    "ppo_framestack": "PPO + FrameStack",
    "ppo_recurrent": "RecurrentPPO",
}


def print_summary_table(runs: dict[str, AnalysisRun]) -> None:
    """Print a console table with MDP baseline reward and deltas per ablation.

    Args:
        runs: Ordered dict from load_analysis("distortion_ablation").
    """
    baseline = runs[BASELINE_LABEL]

    header = f"{'Agent':<20s} {'MDP':>12s}"
    for label in ABLATION_LABELS:
        header += f" {label:>14s}"
    print(header)
    print("-" * len(header))

    for agent in AGENTS:
        mdp_m = baseline.agent_metrics(agent)
        mdp_reward = mdp_m["cross_seed_mean_reward"]
        mdp_se = mdp_m["cross_seed_se_reward"]
        row = f"{AGENT_LABELS[agent]:<20s} {mdp_reward:>7.3f}±{mdp_se:<4.3f}"

        for ablation in ABLATION_ORDER:
            abl_m = runs[ablation].agent_metrics(agent)
            delta = abl_m["cross_seed_mean_reward"] - mdp_reward
            row += f" {delta:>+14.3f}"
        print(row)


def plot_distortion_ablation(
    runs: dict[str, AnalysisRun],
    save_path: Optional[str] = None,
) -> None:
    """Generate a heatmap of reward degradation per distortion per agent.

    Args:
        runs: Ordered dict from load_analysis("distortion_ablation").
        save_path: Path to save the figure. If None, displays interactively.
    """
    baseline = runs[BASELINE_LABEL]

    # Build delta matrix: agents x ablations
    delta = np.zeros((len(AGENTS), len(ABLATION_ORDER)))
    for i, agent in enumerate(AGENTS):
        mdp_reward = baseline.agent_metrics(agent)["cross_seed_mean_reward"]
        for j, ablation in enumerate(ABLATION_ORDER):
            abl_reward = runs[ablation].agent_metrics(agent)["cross_seed_mean_reward"]
            delta[i, j] = abl_reward - mdp_reward

    fig, ax = plt.subplots(figsize=(8, 4))

    # Symmetric color scale centered at 0
    abs_max = max(abs(delta.min()), abs(delta.max()), 0.05)
    im = ax.imshow(
        delta, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto",
    )

    # Annotate cells; pick text color based on position in colormap
    for i in range(len(AGENTS)):
        for j in range(len(ABLATION_ORDER)):
            norm_val = (delta[i, j] + abs_max) / (2 * abs_max)
            color = "white" if norm_val < 0.35 or norm_val > 0.85 else "black"
            ax.text(
                j, i, f"{delta[i, j]:+.2f}",
                ha="center", va="center", color=color, fontsize=11,
            )

    ax.set_xticks(range(len(ABLATION_ORDER)))
    ax.set_xticklabels(ABLATION_LABELS)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([AGENT_LABELS[a] for a in AGENTS])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta Reward (vs. MDP)")

    ax.set_title("Distortion Ablation: Performance Drop by Distortion Type")
    plt.tight_layout()
    _save_or_show(save_path)


if __name__ == "__main__":
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    runs = load_analysis(ANALYSIS_NAME)

    print_summary_table(runs)
    print()

    save_path = str(output_dir / "distortion_ablation.png")
    plot_distortion_ablation(runs, save_path=save_path)
    print(f"Heatmap saved to {save_path}")
