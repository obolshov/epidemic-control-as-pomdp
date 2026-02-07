"""
Experiment management system for organizing experiment runs, configurations, and results.

This module provides infrastructure for:
- Managing experiment configurations (base config + POMDP parameters)
- Creating structured directory hierarchies for experiment outputs
- Saving and loading experiment metadata (config.json, summary.json)
- Providing consistent paths for weights, plots, and logs
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import DefaultConfig


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
        train_rl: Whether to train RL agents or load existing weights.
        num_eval_episodes: Number of evaluation episodes (for future multi-seed evaluation).
        total_timesteps: Total timesteps for RL training.
        timestamp: ISO timestamp of when the experiment was created.
    """
    base_config: DefaultConfig
    pomdp_params: Dict[str, Any]
    scenario_name: str
    is_custom: bool
    target_agents: List[str]
    train_rl: bool
    num_eval_episodes: int = 1
    total_timesteps: int = 50000
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "is_custom": self.is_custom,
            "timestamp": self.timestamp,
            "base_config": {
                "N": self.base_config.N,
                "E0": self.base_config.E0,
                "I0": self.base_config.I0,
                "beta_0": self.base_config.beta_0,
                "sigma": self.base_config.sigma,
                "gamma": self.base_config.gamma,
                "days": self.base_config.days,
                "action_interval": self.base_config.action_interval,
                "w_I": self.base_config.w_I,
                "w_S": self.base_config.w_S,
                "thresholds": self.base_config.thresholds,
            },
            "pomdp_params": self.pomdp_params,
            "target_agents": self.target_agents,
            "train_rl": self.train_rl,
            "num_eval_episodes": self.num_eval_episodes,
            "total_timesteps": self.total_timesteps,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from dictionary (loaded from JSON)."""
        # Reconstruct DefaultConfig
        base_config = DefaultConfig()
        if "base_config" in data:
            bc = data["base_config"]
            base_config.N = bc.get("N", base_config.N)
            base_config.E0 = bc.get("E0", base_config.E0)
            base_config.I0 = bc.get("I0", base_config.I0)
            base_config.beta_0 = bc.get("beta_0", base_config.beta_0)
            base_config.sigma = bc.get("sigma", base_config.sigma)
            base_config.gamma = bc.get("gamma", base_config.gamma)
            base_config.days = bc.get("days", base_config.days)
            base_config.action_interval = bc.get("action_interval", base_config.action_interval)
            base_config.w_I = bc.get("w_I", base_config.w_I)
            base_config.w_S = bc.get("w_S", base_config.w_S)
            base_config.thresholds = bc.get("thresholds", base_config.thresholds)
        
        return ExperimentConfig(
            base_config=base_config,
            pomdp_params=data.get("pomdp_params", {}),
            scenario_name=data.get("scenario_name", "custom"),
            is_custom=data.get("is_custom", True),
            target_agents=data.get("target_agents", []),
            train_rl=data.get("train_rl", False),
            num_eval_episodes=data.get("num_eval_episodes", 1),
            total_timesteps=data.get("total_timesteps", 50000),
            timestamp=data.get("timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
        )


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


def load_experiment(experiment_path: str) -> ExperimentConfig:
    """
    Load experiment configuration from an existing experiment directory.
    
    Args:
        experiment_path: Path to experiment directory containing config.json.
        
    Returns:
        ExperimentConfig loaded from config.json.
        
    Raises:
        FileNotFoundError: If config.json does not exist.
        json.JSONDecodeError: If config.json is malformed.
    """
    config_path = Path(experiment_path) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {experiment_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return ExperimentConfig.from_dict(data)


def find_latest_experiment(scenario_name: str, base_dir: str = "experiments") -> Optional[Path]:
    """
    Find the most recent experiment directory for a given scenario.
    
    Args:
        scenario_name: Name of the scenario (e.g., "mdp").
        base_dir: Base directory for experiments.
        
    Returns:
        Path to the latest experiment directory, or None if no experiments found.
    """
    scenario_dir = Path(base_dir) / scenario_name
    
    if not scenario_dir.exists():
        return None
    
    # List all subdirectories (timestamps)
    experiment_dirs = [d for d in scenario_dir.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        return None
    
    # Sort by directory name (timestamp format ensures lexicographic = chronological)
    experiment_dirs.sort(reverse=True)
    
    return experiment_dirs[0]
