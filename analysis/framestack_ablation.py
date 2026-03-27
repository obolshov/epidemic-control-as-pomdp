"""FrameStack Window Size Ablation: reward vs. n_stack.

Generates a line plot showing how FrameStack performance changes with window
size, with RecurrentPPO and PPO baseline as horizontal reference lines.
Also prints a summary table of all metrics and plots training learning curves
for selected window sizes.

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
import numpy as np

from analysis.data import AnalysisRun, load_analysis
from src.utils import _save_or_show

ANALYSIS_NAME = "framestack_ablation"
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


def _get_framestack_agent(run: AnalysisRun) -> str:
    """Find the ppo_framestack variant agent name in this run's summary.

    Args:
        run: Analysis run loaded from the manifest.

    Returns:
        Agent name starting with "ppo_framestack" (e.g. "ppo_framestack_nstack5").

    Raises:
        ValueError: If no ppo_framestack agent is found.
    """
    for agent in run.available_agents:
        if agent.startswith("ppo_framestack"):
            return agent
    raise ValueError(
        f"No ppo_framestack agent found in run '{run.label}'. "
        f"Available agents: {run.available_agents}"
    )


def _collect_n_stack_entries(
    runs: dict[str, AnalysisRun],
) -> list[tuple[int, str, AnalysisRun]]:
    """Collect and sort (n_stack, agent_name, run) tuples from manifest."""
    entries = []
    for label, run in runs.items():
        if not _is_n_stack_label(label):
            continue
        n = _parse_n_stack(label)
        agent = _get_framestack_agent(run)
        entries.append((n, agent, run))
    entries.sort(key=lambda e: e[0])
    return entries


def print_summary_table(runs: dict[str, AnalysisRun]) -> None:
    """Print a formatted summary table of all n_stack configurations.

    Columns: n_stack, Reward, SE, Stringency, Peak Infected,
    Total Infected, Between-seed Std.

    Args:
        runs: Dict from load_analysis("framestack_ablation").
    """
    entries = _collect_n_stack_entries(runs)

    header = (
        f"{'n_stack':>7} | {'Reward':>7} | {'SE':>5} | {'Stringency':>10} | "
        f"{'Peak Inf':>9} | {'Total Inf':>10} | {'Seed Std':>8}"
    )
    print(header)
    print("-" * len(header))

    for n, agent, run in entries:
        m = run.agent_metrics(agent)
        n_seeds = run.evaluation[agent]["n_seeds"]
        between_seed_std = m["cross_seed_se_reward"] * np.sqrt(n_seeds)

        print(
            f"{n:>7} | {m['cross_seed_mean_reward']:>7.3f} | "
            f"{m['cross_seed_se_reward']:>5.3f} | "
            f"{m['cross_seed_mean_total_stringency']:>10.2f} | "
            f"{m['cross_seed_mean_peak_infected']:>9.0f} | "
            f"{m['cross_seed_mean_total_infected']:>10.0f} | "
            f"{between_seed_std:>8.3f}"
        )


DEFAULT_LEARNING_CURVE_NSTACKS = [1, 4, 10, 15, 30, 40, 60]


def plot_learning_curves(
    runs: dict[str, AnalysisRun],
    n_stacks: Optional[list[int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training evaluation curves for selected n_stack values.

    For each selected n_stack, loads evaluations.npz from all seeds and
    plots mean ± SD reward over training timesteps.

    Args:
        runs: Dict from load_analysis("framestack_ablation").
        n_stacks: Which window sizes to plot. Defaults to [1, 4, 10, 30, 60].
        save_path: Path to save the figure. If None, displays interactively.
    """
    if n_stacks is None:
        n_stacks = DEFAULT_LEARNING_CURVE_NSTACKS

    entries = _collect_n_stack_entries(runs)
    selected = [(n, agent, run) for n, agent, run in entries if n in n_stacks]

    if not selected:
        print("Warning: no matching n_stack entries found for learning curves.")
        return

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (n, agent, run) in enumerate(selected):
        color = cmap(i / max(len(selected) - 1, 1))

        # Find all evaluations.npz for this agent's seeds
        log_dir = run.run_dir / "logs"
        npz_paths = sorted(log_dir.glob(f"{agent}_seed*_eval/evaluations.npz"))

        seed_curves = []
        for npz_path in npz_paths:
            data = np.load(npz_path)
            per_eval_mean = np.mean(data["results"], axis=1)
            seed_curves.append((data["timesteps"], per_eval_mean))

        if not seed_curves:
            print(f"Warning: no eval data for n_stack={n}, skipping.")
            continue

        # Truncate to shortest seed (eval checkpoint counts may differ slightly)
        min_len = min(len(curve) for _, curve in seed_curves)
        timesteps = seed_curves[0][0][:min_len]
        stacked = np.stack([curve[:min_len] for _, curve in seed_curves], axis=0)

        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)

        ax.plot(timesteps, mean_curve, linewidth=2, color=color,
                label=f"n_stack={n}")
        ax.fill_between(timesteps, mean_curve - std_curve,
                        mean_curve + std_curve, alpha=0.2, color=color)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("Training Curves by FrameStack Window Size")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(save_path)


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
    collected = _collect_n_stack_entries(runs)

    n_stacks, means, ses = [], [], []
    for n, agent, run in collected:
        m = run.agent_metrics(agent)
        n_stacks.append(n)
        means.append(m["cross_seed_mean_reward"])
        ses.append(m["cross_seed_se_reward"])

    # Reference lines from dedicated manifest entries
    recurrent_mean = runs[REFERENCE_AGENT].agent_metrics(
        REFERENCE_AGENT
    )["cross_seed_mean_reward"]
    baseline_mean = runs[BASELINE_AGENT].agent_metrics(
        BASELINE_AGENT
    )["cross_seed_mean_reward"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        n_stacks, means, yerr=ses,
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
    ax.set_ylabel("Total Reward (mean ± SE)")
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

    print_summary_table(runs)
    print()

    save_path = str(output_dir / "framestack_ablation.png")
    plot_framestack_ablation(runs, save_path=save_path)
    print(f"Ablation plot saved to {save_path}")

    curves_path = str(output_dir / "framestack_learning_curves.png")
    plot_learning_curves(runs, save_path=curves_path)
    print(f"Learning curves saved to {curves_path}")
