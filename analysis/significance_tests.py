"""Wilcoxon signed-rank tests with Holm-Bonferroni correction.

Pairwise agent comparisons across all scenarios from the POMDP gap analysis.

Usage:
    python -m analysis.significance_tests
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.data import AnalysisRun, load_analysis

SCENARIO_ORDER = ["mdp", "no_exposed", "underreporting", "noisy_pomdp", "pomdp"]
SCENARIO_LABELS = ["MDP", "+ No exposed", "+ Underreporting", "+ Noise", "POMDP"]

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
    runs: dict[str, AnalysisRun],
) -> pd.DataFrame:
    """Wilcoxon signed-rank tests for all agent pairs across scenarios.

    Args:
        runs: Ordered dict from load_analysis("pomdp_gap").

    Returns:
        DataFrame with columns: scenario, pair, p_value, p_adjusted, significant.
    """
    rows: list[dict] = []
    for scenario in SCENARIO_ORDER:
        run = runs[scenario]
        for agent_a, agent_b in PAIRS:
            rewards_a = np.array(run.agent_episode_rewards(agent_a))
            rewards_b = np.array(run.agent_episode_rewards(agent_b))
            diff = rewards_a - rewards_b
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
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    runs = load_analysis("pomdp_gap")

    results = run_wilcoxon_tests(runs)
    print_significance_table(results)
    csv_path = output_dir / "significance_tests.csv"
    results.to_csv(csv_path, index=False)
    print(f"Significance tests saved to {csv_path}")
