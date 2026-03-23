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

        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found at {config_path} "
                f"(analysis='{name}', label='{label}')"
            )

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"summary.json not found at {summary_path} "
                f"(analysis='{name}', label='{label}'). "
                f"Run evaluation first."
            )

        eval_path = run_dir / "evaluation.json"
        if not eval_path.exists():
            raise FileNotFoundError(
                f"evaluation.json not found at {eval_path} "
                f"(analysis='{name}', label='{label}'). "
                f"Run evaluation first."
            )

        with open(config_path) as f:
            config = json.load(f)
        with open(summary_path) as f:
            summary = json.load(f)
        with open(eval_path) as f:
            evaluation = json.load(f)

        runs[label] = AnalysisRun(
            label=label,
            run_dir=run_dir.resolve(),
            config=config,
            summary=summary,
            evaluation=evaluation,
        )

    return runs


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
