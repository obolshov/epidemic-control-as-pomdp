"""Compare cross-seed mean reward across scenarios × agents.

Reads from analyses.json["reward_grid"][<name>], a 2D mapping
scenario_label -> agent_label -> {"path": "...", "agent": "..."}.
Each cell can point to a different run so e.g. ppo_recurrent with a
non-default lstm_hidden_size can be slotted into one scenario without
affecting the others.

Usage:
    python -m analysis.reward_grid <grid_name>
    python -m analysis.reward_grid --list
"""

import argparse
import json

from analysis.data import DEFAULT_MANIFEST_PATH, AnalysisRun, load_reward_grid


MISSING = "—"


def print_reward_grid(
    grid: dict[str, dict[str, tuple[AnalysisRun, str]]],
) -> None:
    """Print reward mean ± SE as a scenario × agent table.

    Rows are scenarios in manifest order. Columns are the union of
    agent labels across all scenarios, ordered by first appearance.
    Missing cells are rendered as an em-dash.

    Args:
        grid: Output of `load_reward_grid` — scenario_label -> {agent_label -> (run, agent)}.
    """
    agent_labels: list[str] = []
    for row in grid.values():
        for agent_label in row:
            if agent_label not in agent_labels:
                agent_labels.append(agent_label)

    cells: dict[str, dict[str, str]] = {}
    for scenario_label, row in grid.items():
        cells[scenario_label] = {}
        for agent_label in agent_labels:
            if agent_label not in row:
                cells[scenario_label][agent_label] = MISSING
                continue
            run, agent = row[agent_label]
            m = run.agent_metrics(agent)
            cells[scenario_label][agent_label] = (
                f"{m['cross_seed_mean_reward']:.3f} ± "
                f"{m['cross_seed_se_reward']:.3f}"
            )

    scenario_width = max(len(s) for s in grid) if grid else len("Scenario")
    scenario_width = max(scenario_width, len("Scenario"))
    col_widths: dict[str, int] = {}
    for agent_label in agent_labels:
        width = len(agent_label)
        for scenario_label in grid:
            width = max(width, len(cells[scenario_label][agent_label]))
        col_widths[agent_label] = width

    header_parts = [f"{'Scenario':>{scenario_width}}"]
    for agent_label in agent_labels:
        header_parts.append(f"{agent_label:>{col_widths[agent_label]}}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for scenario_label, row_cells in cells.items():
        parts = [f"{scenario_label:>{scenario_width}}"]
        for agent_label in agent_labels:
            parts.append(
                f"{row_cells[agent_label]:>{col_widths[agent_label]}}"
            )
        print(" | ".join(parts))


def list_reward_grids() -> None:
    """Print available reward_grid names from the manifest."""
    with open(DEFAULT_MANIFEST_PATH) as f:
        manifest = json.load(f)

    grids = manifest.get("reward_grid", {})
    if not grids:
        print("No reward_grid entries defined in analyses.json.")
        return

    print("Available reward grids:")
    for name, scenarios in grids.items():
        scen_labels = list(scenarios.keys())
        print(f"  {name} ({len(scen_labels)} scenarios): {', '.join(scen_labels)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reward across scenarios × agents."
    )
    parser.add_argument(
        "name",
        nargs="?",
        help="Grid name from analyses.json['reward_grid'].",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available reward grids.",
    )
    args = parser.parse_args()

    if args.list or args.name is None:
        list_reward_grids()
        return

    grid = load_reward_grid(args.name)
    print_reward_grid(grid)


if __name__ == "__main__":
    main()
