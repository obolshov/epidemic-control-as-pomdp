"""Analyze trained experiments and produce a POMDP Gap Plot.

Reads summary.json files from predefined scenario experiments and generates
a 3-panel figure showing how performance degrades as partial observability
increases, and whether memory-based agents resist that degradation.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from src.utils import _save_or_show

SCENARIO_ORDER = ["mdp", "no_exposed", "underreporting", "noisy_pomdp", "pomdp"]
SCENARIO_LABELS = ["MDP", "+ No exposed", "+ Underreporting", "+ Noise", "POMDP"]

AGENTS = ["ppo_baseline", "ppo_framestack", "ppo_recurrent"]
AGENT_LABELS = {
    "ppo_baseline": "PPO (no memory)",
    "ppo_framestack": "PPO + FrameStack",
    "ppo_recurrent": "RecurrentPPO",
}

METRICS = [
    ("mean_reward", "std_reward", "Total Reward"),
    ("mean_total_infected", "std_total_infected", "Total Infected"),
    ("mean_total_stringency", "std_total_stringency", "Total Stringency"),
]


def discover_experiments(experiments_dir: Path) -> dict[str, Path]:
    """Find latest run dir for each predefined scenario.

    Scans experiments/ for directories matching predefined scenario base names
    with a _t* suffix, then picks the latest timestamped subfolder.

    Args:
        experiments_dir: Path to the experiments/ directory.

    Returns:
        Mapping from scenario base name to path of its summary.json.

    Raises:
        FileNotFoundError: If any scenario is missing.
    """
    result: dict[str, Path] = {}
    for scenario in SCENARIO_ORDER:
        matches = sorted(experiments_dir.glob(f"{scenario}_t*/"))
        if not matches:
            raise FileNotFoundError(
                f"No experiment directory found for scenario '{scenario}' "
                f"in {experiments_dir}"
            )
        scenario_dir = matches[-1]

        # Find latest timestamped subfolder
        timestamped = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir() and d.name != "weights"]
        )
        if not timestamped:
            raise FileNotFoundError(
                f"No timestamped run subfolder found in {scenario_dir}"
            )
        summary_path = timestamped[-1] / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found at {summary_path}")

        result[scenario] = summary_path
    return result


def load_gap_data(
    experiment_paths: dict[str, Path],
) -> dict[str, dict[str, dict[str, float]]]:
    """Load summary.json for each scenario, extract metrics for target agents.

    Args:
        experiment_paths: Mapping from scenario name to summary.json path.

    Returns:
        Nested dict: {scenario: {agent: {metric_key: value}}}.
    """
    data: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIO_ORDER:
        summary_path = experiment_paths[scenario]
        with open(summary_path) as f:
            summary = json.load(f)

        agents_by_name = {a["agent_name"]: a for a in summary["agents"]}
        data[scenario] = {}
        for agent_key in AGENTS:
            if agent_key not in agents_by_name:
                raise KeyError(
                    f"Agent '{agent_key}' not found in {summary_path}. "
                    f"Available: {list(agents_by_name.keys())}"
                )
            agent_data = agents_by_name[agent_key]
            data[scenario][agent_key] = {
                k: agent_data[k]
                for metric_pair in METRICS
                for k in metric_pair[:2]
            }
    return data


def plot_pomdp_gap(
    data: dict[str, dict[str, dict[str, float]]],
    save_path: Optional[str] = None,
) -> None:
    """Generate a 3-panel POMDP gap plot.

    Args:
        data: Nested dict from load_gap_data().
        save_path: Path to save the figure. If None, displays interactively.
    """
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    agent_colors = {agent: colors[i] for i, agent in enumerate(AGENTS)}
    x = list(range(len(SCENARIO_ORDER)))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    for panel_idx, (mean_key, std_key, ylabel) in enumerate(METRICS):
        ax = axes[panel_idx]
        for agent_key in AGENTS:
            means = [data[s][agent_key][mean_key] for s in SCENARIO_ORDER]
            sds = [data[s][agent_key][std_key] for s in SCENARIO_ORDER]
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

    # Legend on top panel
    axes[0].legend(loc="best")

    # X tick labels on bottom panel
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(SCENARIO_LABELS)
    axes[-1].set_xlabel("Scenario (increasing partial observability →)")

    fig.suptitle("POMDP Gap: Agent Performance vs. Observability", fontsize=14)
    plt.tight_layout()

    _save_or_show(save_path)


if __name__ == "__main__":
    experiments_dir = Path("experiments")
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    paths = discover_experiments(experiments_dir)
    data = load_gap_data(paths)
    plot_pomdp_gap(data, save_path=str(output_dir / "pomdp_gap_plot.png"))
    print(f"Plot saved to {output_dir / 'pomdp_gap_plot.png'}")
