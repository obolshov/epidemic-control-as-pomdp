"""Compare arbitrary experiment results in a summary table.

Reads from analyses.json["comparisons"], where each comparison maps
user-chosen labels to {"path": "...", "agent": "..."} entries.

Usage:
    python -m analysis.compare <comparison_name>
    python -m analysis.compare --list
"""

import argparse
import json

import numpy as np

from analysis.data import DEFAULT_MANIFEST_PATH, load_comparison


def print_comparison_table(
    entries: list[tuple[str, object, str]],
) -> None:
    """Print a formatted summary table for a comparison.

    Columns: Label, Reward, SE, Stringency, Peak Inf, Total Inf, Seed Std.

    Args:
        entries: List of (label, AnalysisRun, agent_name) from load_comparison.
    """
    max_label = max(len(label) for label, _, _ in entries)
    max_label = max(max_label, 5)  # minimum width

    header = (
        f"{'Label':>{max_label}} | {'Reward':>7} | {'SE':>5} | "
        f"{'Stringency':>10} | {'Peak Inf':>9} | {'Total Inf':>10} | "
        f"{'Seed Std':>8}"
    )
    print(header)
    print("-" * len(header))

    for label, run, agent in entries:
        m = run.agent_metrics(agent)
        n_seeds = run.evaluation[agent]["n_seeds"]
        between_seed_std = m["cross_seed_se_reward"] * np.sqrt(n_seeds)

        print(
            f"{label:>{max_label}} | {m['cross_seed_mean_reward']:>7.3f} | "
            f"{m['cross_seed_se_reward']:>5.3f} | "
            f"{m['cross_seed_mean_total_stringency']:>10.2f} | "
            f"{m['cross_seed_mean_peak_infected']:>9.0f} | "
            f"{m['cross_seed_mean_total_infected']:>10.0f} | "
            f"{between_seed_std:>8.3f}"
        )


def list_comparisons() -> None:
    """Print available comparison names from the manifest."""
    with open(DEFAULT_MANIFEST_PATH) as f:
        manifest = json.load(f)

    comparisons = manifest.get("comparisons", {})
    if not comparisons:
        print("No comparisons defined in analyses.json.")
        return

    print("Available comparisons:")
    for name, entries in comparisons.items():
        labels = list(entries.keys())
        print(f"  {name} ({len(labels)} entries): {', '.join(labels)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare arbitrary experiment results."
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Comparison name from analyses.json['comparisons'].",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available comparisons.",
    )
    args = parser.parse_args()

    if args.list or args.name is None:
        list_comparisons()
        return

    entries = load_comparison(args.name)
    print_comparison_table(entries)


if __name__ == "__main__":
    main()
