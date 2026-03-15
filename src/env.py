from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import Config
from src.seir import run_seir, EpidemicState
from src.agents import InterventionAction, Agent, StaticAgent


def calculate_reward(
    I_t: float, action: InterventionAction, config: Config, prev_action_idx: int = 0
) -> tuple[float, Dict[str, float]]:
    """
    Args:
        I_t: Infected count.
        action: Action taken.
        config: Environment configuration with reward weights.
        prev_action_idx: Index of the previous action (for switching penalty).

    Returns:
        Tuple of (total_reward, components_dict) where components_dict
        contains the individual penalty terms (negative values).
    """
    infection_penalty = config.w_I * (I_t / config.N) ** 2
    stringency_penalty = config.w_S * (1 - action.value)
    action_idx = list(InterventionAction).index(action)
    delta = action_idx - prev_action_idx
    switching_penalty = config.w_switch * delta ** 2

    total = -(infection_penalty + stringency_penalty + switching_penalty)
    components = {
        "reward_infection": -infection_penalty,
        "reward_stringency": -stringency_penalty,
        "reward_switching": -switching_penalty,
    }
    return total, components


class EpidemicEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config, action_delay: int = 0):
        super().__init__()
        self.config = config
        self.action_delay = action_delay

        self.action_space = spaces.Discrete(len(InterventionAction))
        self.action_map = list(InterventionAction)

        n_actions = len(InterventionAction)
        obs_low = np.zeros(6, dtype=np.float32)
        obs_high = np.array(
            [self.config.N, self.config.N, self.config.N, self.config.N, n_actions - 1, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.current_state = None
        self.current_day = 0
        self.prev_action_idx: int = 0
        self._action_queue: deque = deque()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        S0 = self.config.N - self.config.I0 - self.config.E0
        self.current_state = EpidemicState(
            N=self.config.N,
            S=S0,
            E=self.config.E0,
            I=self.config.I0,
            R=0,
        )
        self.current_day = 0
        self.prev_action_idx = 0
        self._action_queue = deque([0] * self.action_delay)

        return self._get_obs(), {}

    def step(self, action_idx):
        if self.action_delay > 0:
            self._action_queue.append(action_idx)
            applied_action_idx = self._action_queue.popleft()
        else:
            applied_action_idx = action_idx

        action_enum = self.action_map[applied_action_idx]
        beta = self.config.beta_0 * action_enum.value

        days_to_simulate = min(
            self.config.action_interval, self.config.days - self.current_day
        )

        rng = self.np_random if self.config.stochastic else None

        S, E, I, R = run_seir(
            self.current_state,
            beta,
            self.config.sigma,
            self.config.gamma,
            days_to_simulate,
            rng=rng
        )

        if len(S) > 0:
            self.current_state = EpidemicState(
                N=self.current_state.N,
                S=S[-1],
                E=E[-1],
                I=I[-1],
                R=R[-1],
            )

        reward, reward_components = calculate_reward(self.current_state.I, action_enum, self.config, self.prev_action_idx)
        self.prev_action_idx = applied_action_idx

        self.current_day += days_to_simulate

        terminated = self.current_day >= self.config.days
        truncated = False

        info = {"S": S, "E": E, "I": I, "R": R, **reward_components}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(
            [
                self.current_state.S,
                self.current_state.E,
                self.current_state.I,
                self.current_state.R,
                float(self.prev_action_idx),
                self.current_day / self.config.days,
            ],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        print(
            f"Day {self.current_day}: S={self.current_state.S:.0f}, E={self.current_state.E:.0f}, I={self.current_state.I:.0f}, R={self.current_state.R:.0f}"
        )


class SimulationResult:
    def __init__(
        self,
        agent: Agent,
        t: np.ndarray,
        S: np.ndarray,
        E: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        actions: List[InterventionAction],
        timesteps: List[int],
        rewards: List[float],
        observations: List[np.ndarray],
        reward_components: List[Dict[str, float]],
        custom_name: str = None,
    ):
        self.agent = agent
        self.t = t
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.actions = actions
        self.timesteps = timesteps
        self.rewards = rewards
        self.observations = observations
        self.reward_components = reward_components
        self.custom_name = custom_name

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
