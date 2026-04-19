"""Manifest-based experiment data loading for analysis scripts.

Reads analyses.json to resolve which experiment runs belong to which analysis,
then loads config.json, summary.json, and evaluation.json from each run directory.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENTS_DIR = Path("experiments")
DEFAULT_MANIFEST_PATH = Path("analyses.json")

METRIC_KEYS = [
    "cross_seed_mean_reward", "cross_seed_se_reward",
    "cross_seed_mean_total_infected", "cross_seed_se_total_infected",
    "cross_seed_mean_total_stringency", "cross_seed_se_total_stringency",
    "cross_seed_mean_peak_infected", "cross_seed_se_peak_infected",
]


@dataclass
class AnalysisRun:
    """One experiment run loaded from the manifest.

    Attributes:
        label: User-chosen label from the manifest key (e.g. "mdp", "n_stack=10").
        run_dir: Absolute path to the timestamped run directory.
        config: Parsed config.json contents.
        summary: Parsed summary.json contents.
        evaluation: Parsed evaluation.json contents.
    """

    label: str
    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]
    evaluation: dict[str, Any]

    @property
    def scenario_name(self) -> str:
        return self.config["scenario_name"]

    @property
    def n_stack(self) -> int:
        return self.config["base_config"]["n_stack"]

    @property
    def lstm_hidden_size(self) -> int:
        return self.config["base_config"]["lstm_hidden_size"]

    @property
    def total_timesteps(self) -> int:
        return self.config["base_config"].get("total_timesteps",
               self.config.get("total_timesteps", 0))

    @property
    def available_agents(self) -> list[str]:
        return list(self.summary["agents"].keys())

    def agent_metrics(self, agent_name: str) -> dict[str, float]:
        """Extract all metric key-value pairs for one agent from summary.json.

        Args:
            agent_name: Agent identifier (e.g. "ppo_baseline").

        Returns:
            Dict with keys from METRIC_KEYS.

        Raises:
            KeyError: If agent not found in this run's summary.
        """
        agent = self.summary["agents"][agent_name]
        return {k: agent[k] for k in METRIC_KEYS if k in agent}

    def agent_episode_rewards(self, agent_name: str) -> list[float]:
        """All per-episode rewards for one agent (concatenated across seeds).

        Args:
            agent_name: Agent identifier.

        Returns:
            Flat list of per-episode total rewards from all seeds.

        Raises:
            KeyError: If agent not found in evaluation data.
        """
        agent_eval = self.evaluation[agent_name]
        all_rewards: list[float] = []
        for seed_data in agent_eval["seeds"].values():
            all_rewards.extend(seed_data["total_reward"])
        return all_rewards

    def agent_seed_arrays(self, agent_name: str, metric: str) -> list[np.ndarray]:
        """Per-seed arrays of a metric from evaluation.json.

        Args:
            agent_name: Agent identifier.
            metric: One of "total_reward", "peak_infected", "total_infected",
                "total_stringency".

        Returns:
            List of numpy arrays, one per seed.
        """
        agent_eval = self.evaluation[agent_name]
        return [
            np.array(seed_data[metric])
            for seed_data in agent_eval["seeds"].values()
        ]

    def to_row(self, agent_name: str) -> dict[str, Any]:
        """Flat dict suitable for DataFrame construction.

        Includes label, agent name, all metrics, and key config values.

        Args:
            agent_name: Agent identifier.

        Returns:
            Flat dict with label, agent, metrics, n_stack, lstm_hidden_size.
        """
        row: dict[str, Any] = {
            "label": self.label,
            "agent": agent_name,
        }
        row.update(self.agent_metrics(agent_name))
        row["n_stack"] = self.n_stack
        row["lstm_hidden_size"] = self.lstm_hidden_size
        return row


def load_analysis(
    name: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> dict[str, AnalysisRun]:
    """Load all experiment runs for a named analysis from the manifest.

    Args:
        name: Analysis name (key in analyses.json, e.g. "pomdp_gap").
        manifest_path: Path to the manifest JSON file.
        experiments_dir: Base directory containing experiment folders.

    Returns:
        Ordered dict mapping label -> AnalysisRun.

    Raises:
        FileNotFoundError: If manifest file, run directory, or required
            JSON files are missing.
        KeyError: If analysis name not found in manifest.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            f"Create analyses.json with experiment run mappings."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    if name not in manifest:
        raise KeyError(
            f"Analysis '{name}' not found in {manifest_path}. "
            f"Available: {list(manifest.keys())}"
        )

    runs: dict[str, AnalysisRun] = {}
    for label, rel_path in manifest[name].items():
        run_dir = experiments_dir / rel_path
        context = f" (analysis='{name}', label='{label}')"
        runs[label] = _load_run(run_dir, label, context)

    return runs


def _load_run(
    run_dir: Path,
    label: str,
    context: str = "",
) -> AnalysisRun:
    """Load a single AnalysisRun from a directory.

    Args:
        run_dir: Path to the experiment run directory.
        label: User-facing label for this run.
        context: Extra context string for error messages.
    """
    for filename in ("config.json", "summary.json", "evaluation.json"):
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename} not found at {path}{context}")

    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "evaluation.json") as f:
        evaluation = json.load(f)

    return AnalysisRun(
        label=label,
        run_dir=run_dir.resolve(),
        config=config,
        summary=summary,
        evaluation=evaluation,
    )


def load_comparison(
    name: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> list[tuple[str, AnalysisRun, str]]:
    """Load a named comparison from analyses.json["comparisons"].

    Each entry maps a label to {"path": "...", "agent": "..."}.
    If agent is omitted, all agents in the run are returned as separate entries.

    Args:
        name: Comparison name (key under "comparisons" in analyses.json).
        manifest_path: Path to the manifest JSON file.
        experiments_dir: Base directory containing experiment folders.

    Returns:
        List of (label, AnalysisRun, agent_name) tuples in manifest order.

    Raises:
        FileNotFoundError: If manifest or run files are missing.
        KeyError: If comparison name or specified agent not found.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    comparisons = manifest.get("comparisons", {})
    if name not in comparisons:
        available = list(comparisons.keys())
        raise KeyError(
            f"Comparison '{name}' not found. Available: {available}"
        )

    entries: list[tuple[str, AnalysisRun, str]] = []
    for label, entry in comparisons[name].items():
        rel_path = entry["path"]
        agent = entry.get("agent")
        run_dir = experiments_dir / rel_path
        context = f" (comparison='{name}', label='{label}')"

        run = _load_run(run_dir, label, context)

        if agent is not None:
            if agent not in run.available_agents:
                raise KeyError(
                    f"Agent '{agent}' not found in {run_dir}. "
                    f"Available: {run.available_agents}"
                )
            entries.append((label, run, agent))
        else:
            for agent_name in run.available_agents:
                display = f"{label} ({agent_name})"
                entries.append((display, run, agent_name))

    return entries


def load_reward_grid(
    name: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> dict[str, dict[str, tuple[AnalysisRun, str]]]:
    """Load a 2D scenario × agent grid from analyses.json["reward_grid"][name].

    Each cell is `{"path": "...", "agent": "..."}` — same shape as
    `comparisons` entries, but nested one extra level by scenario.

    Args:
        name: Grid name (key under "reward_grid" in analyses.json).
        manifest_path: Path to the manifest JSON file.
        experiments_dir: Base directory containing experiment folders.

    Returns:
        Ordered dict `scenario_label -> {agent_label -> (AnalysisRun, agent_name)}`
        preserving manifest order for both scenarios and agent columns.

    Raises:
        FileNotFoundError: If manifest or run files are missing.
        KeyError: If grid name or a specified agent is not found.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    grids = manifest.get("reward_grid", {})
    if name not in grids:
        raise KeyError(
            f"Reward grid '{name}' not found. Available: {list(grids.keys())}"
        )

    grid: dict[str, dict[str, tuple[AnalysisRun, str]]] = {}
    for scenario_label, agents in grids[name].items():
        row: dict[str, tuple[AnalysisRun, str]] = {}
        for agent_label, entry in agents.items():
            rel_path = entry["path"]
            agent = entry["agent"]
            run_dir = experiments_dir / rel_path
            context = (
                f" (reward_grid='{name}', scenario='{scenario_label}', "
                f"agent_label='{agent_label}')"
            )
            run = _load_run(run_dir, scenario_label, context)
            if agent not in run.available_agents:
                raise KeyError(
                    f"Agent '{agent}' not found in {run_dir}{context}. "
                    f"Available: {run.available_agents}"
                )
            row[agent_label] = (run, agent)
        grid[scenario_label] = row

    return grid


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> list[str]:
    """Check all manifest entries and return warnings for issues.

    Args:
        manifest_path: Path to the manifest JSON file.
        experiments_dir: Base directory containing experiment folders.

    Returns:
        List of warning strings. Empty list means everything is OK.
    """
    warnings: list[str] = []

    if not manifest_path.exists():
        return [f"Manifest not found: {manifest_path}"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    for analysis_name, entries in manifest.items():
        if analysis_name == "comparisons":
            for comp_name, comp_entries in entries.items():
                for label, entry in comp_entries.items():
                    rel_path = entry["path"]
                    run_dir = experiments_dir / rel_path
                    prefix = f"[comparisons/{comp_name}/{label}]"
                    if not run_dir.exists():
                        warnings.append(
                            f"{prefix} directory not found: {run_dir}")
                        continue
                    for filename in ("config.json", "summary.json",
                                     "evaluation.json"):
                        if not (run_dir / filename).exists():
                            warnings.append(
                                f"{prefix} {filename} missing in {run_dir}")
            continue

        if analysis_name == "reward_grid":
            for grid_name, scenarios in entries.items():
                for scenario_label, agents in scenarios.items():
                    for agent_label, entry in agents.items():
                        rel_path = entry["path"]
                        run_dir = experiments_dir / rel_path
                        prefix = (
                            f"[reward_grid/{grid_name}/{scenario_label}/"
                            f"{agent_label}]"
                        )
                        if not run_dir.exists():
                            warnings.append(
                                f"{prefix} directory not found: {run_dir}")
                            continue
                        for filename in ("config.json", "summary.json",
                                         "evaluation.json"):
                            if not (run_dir / filename).exists():
                                warnings.append(
                                    f"{prefix} {filename} missing in {run_dir}")
            continue

        for label, rel_path in entries.items():
            run_dir = experiments_dir / rel_path
            prefix = f"[{analysis_name}/{label}]"

            if not run_dir.exists():
                warnings.append(f"{prefix} directory not found: {run_dir}")
                continue

            for filename in ("config.json", "summary.json", "evaluation.json"):
                if not (run_dir / filename).exists():
                    warnings.append(f"{prefix} {filename} missing in {run_dir}")

    return warnings
