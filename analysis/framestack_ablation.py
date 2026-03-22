"""FrameStack Window Size Ablation: reward vs. n_stack.

Generates a line plot showing how FrameStack performance changes with window
size, with RecurrentPPO and PPO baseline as horizontal reference lines.

The manifest entry should include:
- "n_stack=X" labels for each window size (parsed as data points)
- "ppo_baseline" and "ppo_recurrent" labels for reference lines

Usage:
    python -m analysis.framestack_ablation
"""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from analysis.data import AnalysisRun, load_analysis
from src.utils import _save_or_show

ANALYSIS_NAME = "framestack_ablation"
FRAMESTACK_AGENT = "ppo_framestack"
REFERENCE_AGENT = "ppo_recurrent"
BASELINE_AGENT = "ppo_baseline"

_N_STACK_RE = re.compile(r"^n_stack=(\d+)$")


def _is_n_stack_label(label: str) -> bool:
    """Check if a manifest label is an n_stack data point."""
    return _N_STACK_RE.match(label) is not None


def _parse_n_stack(label: str) -> int:
    """Extract n_stack integer from a label like 'n_stack=10'."""
    match = _N_STACK_RE.match(label)
    if not match:
        raise ValueError(f"Cannot parse n_stack from label '{label}'")
    return int(match.group(1))


def plot_framestack_ablation(
    runs: dict[str, AnalysisRun],
    save_path: Optional[str] = None,
) -> None:
    """Generate FrameStack ablation plot.

    Args:
        runs: Dict from load_analysis("framestack_ablation").
            Must contain "ppo_baseline" and "ppo_recurrent" keys for
            reference lines, plus "n_stack=X" keys for data points.
        save_path: Path to save the figure. If None, displays interactively.
    """
    # Extract n_stack data points (skip reference entries)
    entries = []
    for label, run in runs.items():
        if not _is_n_stack_label(label):
            continue
        n = _parse_n_stack(label)
        metrics = run.agent_metrics(FRAMESTACK_AGENT)
        entries.append((n, metrics["mean_reward"], metrics["std_reward"]))
    entries.sort(key=lambda e: e[0])

    n_stacks = [e[0] for e in entries]
    means = [e[1] for e in entries]
    sds = [e[2] for e in entries]

    # Reference lines from dedicated manifest entries
    recurrent_mean = runs[REFERENCE_AGENT].agent_metrics(
        REFERENCE_AGENT
    )["mean_reward"]
    baseline_mean = runs[BASELINE_AGENT].agent_metrics(
        BASELINE_AGENT
    )["mean_reward"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        n_stacks, means, yerr=sds,
        marker="o", capsize=4, color="C0", label="PPO + FrameStack",
    )
    ax.axhline(
        recurrent_mean, linestyle="--", color="C2", alpha=0.8,
        label=f"RecurrentPPO ({recurrent_mean:.1f})",
    )
    ax.axhline(
        baseline_mean, linestyle="--", color="C1", alpha=0.8,
        label=f"PPO baseline ({baseline_mean:.1f})",
    )

    ax.set_xlabel("n_stack (window size in steps)")
    ax.set_ylabel("Total Reward (mean ± SD)")
    ax.set_title("FrameStack Window Size Ablation (POMDP scenario)")
    ax.set_xticks(n_stacks)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


if __name__ == "__main__":
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    runs = load_analysis(ANALYSIS_NAME)

    save_path = str(output_dir / "framestack_ablation.png")
    plot_framestack_ablation(runs, save_path=save_path)
    print(f"Plot saved to {save_path}")
