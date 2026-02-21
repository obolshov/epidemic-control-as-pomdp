"""
Experiment management system for organizing experiment runs, configurations, and results.

This module provides infrastructure for:
- Managing experiment configurations (base config + POMDP parameters)
- Creating structured directory hierarchies for experiment outputs
- Saving and loading experiment metadata (config.json, summary.json)
- Providing consistent paths for weights, plots, and logs
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.config import Config


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration combining base config with POMDP parameters.

    Attributes:
        base_config: SEIR model and environment configuration.
        pomdp_params: Dictionary of POMDP modifications (e.g., {"include_exposed": False}).
        scenario_name: Name of the scenario ("mdp", "no_exposed", "custom").
        is_custom: Whether this is a custom (ad-hoc) or predefined scenario.
        target_agents: List of agent names to run (e.g., ["random", "threshold", "ppo_baseline"]).
        num_eval_episodes: Number of evaluation episodes (for future multi-seed evaluation).
        total_timesteps: Total timesteps for RL training.
        timestamp: ISO timestamp of when the experiment was created.
    """
    base_config: Config
    pomdp_params: Dict[str, Any]
    scenario_name: str
    is_custom: bool
    target_agents: List[str]
    num_eval_episodes: int = 1
    total_timesteps: int = 50_000
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ExperimentConfig to dictionary for JSON serialization.

        Returns:
            Fully serializable dict with all config fields and POMDP params.
        """
        return {
            "scenario_name": self.scenario_name,
            "is_custom": self.is_custom,
            "timestamp": self.timestamp,
            "base_config": asdict(self.base_config),
            "pomdp_params": self.pomdp_params,
            "target_agents": self.target_agents,
            "num_eval_episodes": self.num_eval_episodes,
            "total_timesteps": self.total_timesteps,
        }


class ExperimentDirectory:
    """
    Manages experiment directory structure and file paths.

    Creates nested structure: experiments/{scenario_name}/{timestamp}/
    Weights are stored at scenario level for reuse: experiments/{scenario_name}/weights/

    Attributes:
        config: ExperimentConfig for this experiment.
        root: Root directory for this experiment run (timestamped).
        scenario_dir: Scenario directory (e.g., experiments/mdp/).
        weights_dir: Directory for model weights (shared across runs).
        plots_dir: Directory for plots/figures.
        logs_dir: Directory for text logs.
        tensorboard_dir: Directory for TensorBoard logs.
    """

    def __init__(self, exp_config: ExperimentConfig, base_dir: str = "experiments"):
        """
        Initialize ExperimentDirectory and create directory structure.

        Args:
            exp_config: Complete experiment configuration.
            base_dir: Base directory for all experiments (default: "experiments").
        """
        self.config = exp_config
        self.base_dir = Path(base_dir)
        self.root = self._create_experiment_dir()

        # Weights are stored at scenario level (shared across runs)
        # experiments/mdp/weights/ instead of experiments/mdp/{timestamp}/weights/
        self.scenario_dir = self.base_dir / self.config.scenario_name
        self.weights_dir = self.scenario_dir / "weights"

        # Results are stored in timestamped directories
        self.plots_dir = self.root / "plots"
        self.logs_dir = self.root / "logs"
        self.tensorboard_dir = self.logs_dir / "tensorboard"

        # Create all directories
        for dir_path in [self.weights_dir, self.plots_dir, self.logs_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _create_experiment_dir(self) -> Path:
        """
        Create experiment directory with nested structure.

        Returns:
            Path to the created experiment directory.
        """
        # Create experiments/{scenario_name}/{timestamp}/
        scenario_dir = self.base_dir / self.config.scenario_name
        experiment_dir = scenario_dir / self.config.timestamp

        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def save_config(self) -> None:
        """Save experiment configuration to config.json."""
        config_path = self.root / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Experiment config saved to: {config_path}")

    def save_summary(self, results: List[Any]) -> None:
        """
        Save experiment summary with key metrics from all agents.

        Args:
            results: List of SimulationResult objects.
        """
        summary_path = self.root / "summary.json"

        summary_data = {
            "scenario_name": self.config.scenario_name,
            "timestamp": self.config.timestamp,
            "num_agents": len(results),
            "agents": []
        }

        for result in results:
            agent_summary = {
                "agent_name": result.agent_name,
                "peak_infected": float(result.peak_infected),
                "total_infected": float(result.total_infected),
                "epidemic_duration": int(result.epidemic_duration),
                "total_reward": float(result.total_reward),
                "num_actions": len(result.actions),
            }
            summary_data["agents"].append(agent_summary)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Experiment summary saved to: {summary_path}")

    def get_weight_path(self, agent_name: str) -> Path:
        """
        Get path for saving/loading agent weights.

        Args:
            agent_name: Name of the agent (e.g., "ppo_baseline").

        Returns:
            Path to weight file: weights/{agent_name}.zip
        """
        return self.weights_dir / f"{agent_name}.zip"

    def get_plot_path(self, plot_name: str) -> Path:
        """
        Get path for saving plots.

        Args:
            plot_name: Name of the plot (e.g., "comparison_all_agents.png").

        Returns:
            Path to plot file: plots/{plot_name}
        """
        return self.plots_dir / plot_name

    def get_log_path(self, agent_name: str) -> Path:
        """
        Get path for saving agent logs.

        Args:
            agent_name: Name of the agent.

        Returns:
            Path to log file: logs/{agent_name}.txt
        """
        return self.logs_dir / f"{agent_name}.txt"

    def __str__(self) -> str:
        """String representation showing experiment directory path."""
        return str(self.root)
