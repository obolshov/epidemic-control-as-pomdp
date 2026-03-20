"""
Data containers for simulation and evaluation results.

- SimulationResult: Single-episode trajectory data.
- AggregatedResult: Multi-episode aggregated statistics (mean ± SD).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.agents import Agent, StaticAgent
from src.env import InterventionAction


@dataclass
class SimulationResult:
    agent: Agent
    t: np.ndarray
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray
    actions: List[InterventionAction]
    timesteps: List[int]
    rewards: List[float]
    observations: List[np.ndarray]
    reward_components: List[Dict[str, float]]
    custom_name: Optional[str] = None

    @property
    def peak_infected(self) -> float:
        return np.max(self.I)

    @property
    def total_infected(self) -> float:
        return self.E[-1] + self.I[-1] + self.R[-1]

    @property
    def agent_name(self) -> str:
        # Use custom name if provided (for RL agents like ppo_baseline, ppo_framestack)
        if self.custom_name:
            return self.custom_name

        if isinstance(self.agent, StaticAgent):
            action_name = list(InterventionAction)[self.agent.action_idx].name
            return f"StaticAgent - {action_name}"

        return self.agent.__class__.__name__

    @property
    def total_stringency(self) -> float:
        """Total societal stringency cost: Σ(1 - action.value) over the episode."""
        return sum(1.0 - a.value for a in self.actions)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


@dataclass
class AggregatedResult:
    """Multi-episode aggregated evaluation result for any agent type.

    Attributes:
        agent_name: Name of the agent.
        t: Shared time axis, shape (n_days,).
        S_mean, S_std: Mean/std of Susceptible over episodes, shape (n_days,).
        E_mean, E_std: Mean/std of Exposed over episodes, shape (n_days,).
        I_mean, I_std: Mean/std of Infected over episodes, shape (n_days,).
        R_mean, R_std: Mean/std of Recovered over episodes, shape (n_days,).
        episode_rewards: Total reward per episode.
        peak_infected_per_episode: Peak I per episode.
        total_infected_per_episode: Total infected (E[-1]+I[-1]+R[-1]) per episode.
        n_episodes: Number of evaluation episodes.
    """
    agent_name: str
    t: np.ndarray
    S_mean: np.ndarray
    S_std: np.ndarray
    E_mean: np.ndarray
    E_std: np.ndarray
    I_mean: np.ndarray
    I_std: np.ndarray
    R_mean: np.ndarray
    R_std: np.ndarray
    episode_rewards: List[float]
    peak_infected_per_episode: List[float]
    total_infected_per_episode: List[float]
    total_stringency_per_episode: List[float]
    n_episodes: int

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards))

    @property
    def std_reward(self) -> float:
        return float(np.std(self.episode_rewards))

    @property
    def mean_peak_infected(self) -> float:
        return float(np.mean(self.peak_infected_per_episode))

    @property
    def std_peak_infected(self) -> float:
        return float(np.std(self.peak_infected_per_episode))

    @property
    def mean_total_infected(self) -> float:
        return float(np.mean(self.total_infected_per_episode))

    @property
    def std_total_infected(self) -> float:
        return float(np.std(self.total_infected_per_episode))

    @property
    def mean_total_stringency(self) -> float:
        return float(np.mean(self.total_stringency_per_episode))

    @property
    def std_total_stringency(self) -> float:
        return float(np.std(self.total_stringency_per_episode))
