"""Analyze trained experiments: POMDP Gap Plot and statistical significance tests.

Reads summary.json files from predefined scenario experiments and generates:
- A 3-panel figure showing how performance degrades as partial observability
  increases, and whether memory-based agents resist that degradation.
- Wilcoxon signed-rank tests with Holm-Bonferroni correction for pairwise
  agent comparisons across all scenarios.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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


PAIRS = [
    ("ppo_baseline", "ppo_framestack"),
    ("ppo_baseline", "ppo_recurrent"),
    ("ppo_framestack", "ppo_recurrent"),
]

PAIR_LABELS = {
    ("ppo_baseline", "ppo_framestack"): "PPO vs FrameStack",
    ("ppo_baseline", "ppo_recurrent"): "PPO vs Recurrent",
    ("ppo_framestack", "ppo_recurrent"): "FrameStack vs Recurrent",
}


def load_episode_data(
    experiment_paths: dict[str, Path],
) -> dict[str, dict[str, list[float]]]:
    """Load per-episode reward arrays from summary.json.

    Args:
        experiment_paths: Mapping from scenario name to summary.json path.

    Returns:
        Nested dict: {scenario: {agent: [episode rewards]}}.

    Raises:
        KeyError: If episode_rewards is missing (summary.json was not regenerated).
    """
    data: dict[str, dict[str, list[float]]] = {}
    for scenario in SCENARIO_ORDER:
        summary_path = experiment_paths[scenario]
        with open(summary_path) as f:
            summary = json.load(f)

        agents_by_name = {a["agent_name"]: a for a in summary["agents"]}
        data[scenario] = {}
        for agent_key in AGENTS:
            agent_data = agents_by_name[agent_key]
            if "episode_rewards" not in agent_data:
                raise KeyError(
                    f"'episode_rewards' not found for agent '{agent_key}' in "
                    f"{summary_path}. Re-run evaluation with --skip-training all "
                    f"to regenerate summary.json with per-episode data."
                )
            data[scenario][agent_key] = agent_data["episode_rewards"]
    return data


def _holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Holm-Bonferroni correction to an array of p-values.

    Args:
        p_values: Raw p-values, shape (m,).
        alpha: Family-wise error rate (unused in adjustment, kept for API clarity).

    Returns:
        Adjusted p-values, shape (m,), clipped to [0, 1].
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    cummax = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (m - rank)
        cummax = max(cummax, adj)
        adjusted[idx] = cummax
    return np.clip(adjusted, 0.0, 1.0)


def run_wilcoxon_tests(
    episode_data: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    """Wilcoxon signed-rank tests for all agent pairs across scenarios.

    Args:
        episode_data: Per-episode rewards from load_episode_data().

    Returns:
        DataFrame with columns: scenario, pair, p_value, p_adjusted, significant.
    """
    rows: list[dict] = []
    for scenario in SCENARIO_ORDER:
        for agent_a, agent_b in PAIRS:
            rewards_a = np.array(episode_data[scenario][agent_a])
            rewards_b = np.array(episode_data[scenario][agent_b])
            diff = rewards_a - rewards_b
            # If all differences are zero, p-value is 1.0 (no evidence of difference)
            if np.all(diff == 0):
                p_val = 1.0
            else:
                _, p_val = wilcoxon(rewards_a, rewards_b)
            rows.append({
                "scenario": scenario,
                "pair": PAIR_LABELS[(agent_a, agent_b)],
                "agent_a": agent_a,
                "agent_b": agent_b,
                "p_value": p_val,
            })

    df = pd.DataFrame(rows)
    df["p_adjusted"] = _holm_bonferroni(df["p_value"].values)
    df["significant"] = df["p_adjusted"].apply(
        lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else "")
    )
    return df


def print_significance_table(results: pd.DataFrame) -> None:
    """Print formatted significance table to console.

    Rows = scenarios, columns = agent pairs. Values are Holm-Bonferroni
    adjusted p-values with significance markers (* p<0.05, ** p<0.01).
    """
    pair_labels = [PAIR_LABELS[p] for p in PAIRS]
    header = f"{'Scenario':<20}" + "".join(f"{lbl:>25}" for lbl in pair_labels)
    separator = "-" * len(header)

    print("\n" + separator)
    print("Wilcoxon Signed-Rank Tests (Holm-Bonferroni adjusted)")
    print(separator)
    print(header)
    print(separator)

    for scenario, label in zip(SCENARIO_ORDER, SCENARIO_LABELS):
        row_data = results[results["scenario"] == scenario]
        cells = []
        for pair in PAIRS:
            pair_label = PAIR_LABELS[pair]
            match = row_data[row_data["pair"] == pair_label]
            if match.empty:
                cells.append(f"{'N/A':>25}")
            else:
                p_adj = match["p_adjusted"].values[0]
                sig = match["significant"].values[0]
                cells.append(f"{p_adj:>20.4f} {sig:<4}")
        print(f"{label:<20}" + "".join(cells))

    print(separator)
    print("* p < 0.05, ** p < 0.01 (Holm-Bonferroni corrected)\n")


if __name__ == "__main__":
    experiments_dir = Path("experiments")
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    paths = discover_experiments(experiments_dir)

    # POMDP gap plot
    data = load_gap_data(paths)
    plot_pomdp_gap(data, save_path=str(output_dir / "pomdp_gap_plot.png"))
    print(f"Plot saved to {output_dir / 'pomdp_gap_plot.png'}")

    # Statistical significance tests
    try:
        episode_data = load_episode_data(paths)
        results = run_wilcoxon_tests(episode_data)
        print_significance_table(results)
        csv_path = output_dir / "significance_tests.csv"
        results.to_csv(csv_path, index=False)
        print(f"Significance tests saved to {csv_path}")
    except KeyError as e:
        print(f"\nSkipping significance tests: {e}")
