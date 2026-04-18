from collections import deque
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from enum import Enum

from src.config import Config
from src.seir import run_seir, EpidemicState


class InterventionAction(Enum):
    NO = 1.0
    MILD = 0.75
    MODERATE = 0.5
    SEVERE = 0.25


def calculate_reward(
    I_t: float,
    applied_action: InterventionAction,
    selected_action_idx: int,
    prev_selected_idx: int,
    config: Config,
) -> tuple[float, Dict[str, float]]:
    """
    Args:
        I_t: Infected count (already driven by the applied action via SEIR).
        applied_action: Action currently in effect (after action_delay queue).
            Drives stringency penalty — physical economic cost of the enforced
            regime.
        selected_action_idx: Action the agent just selected this step. Together
            with prev_selected_idx, determines switching penalty — cost of
            inconsistent announcements.
        prev_selected_idx: Index of the previously selected action.
        config: Environment configuration with reward weights.

    Returns:
        Tuple of (total_reward, components_dict) where components_dict
        contains the individual penalty terms (negative values).
    """
    infection_penalty = config.w_I * (I_t / config.N) ** 2
    stringency_penalty = config.w_S * (1 - applied_action.value)
    delta = selected_action_idx - prev_selected_idx
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
        self.prev_selected_idx: int = 0
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
        self.prev_selected_idx = 0
        self._action_queue = deque([0] * self.action_delay)

        return self._get_obs(), {}

    def step(self, action_idx):
        if self.action_delay > 0:
            self._action_queue.append(action_idx)
            applied_action_idx = self._action_queue.popleft()
        else:
            applied_action_idx = action_idx

        applied_action_enum = self.action_map[applied_action_idx]
        beta = self.config.beta_0 * applied_action_enum.value

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

        reward, reward_components = calculate_reward(
            self.current_state.I,
            applied_action_enum,
            action_idx,
            self.prev_selected_idx,
            self.config,
        )
        self.prev_selected_idx = action_idx

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
                float(self.prev_selected_idx),
                self.current_day / self.config.days,
            ],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        print(
            f"Day {self.current_day}: S={self.current_state.S:.0f}, E={self.current_state.E:.0f}, I={self.current_state.I:.0f}, R={self.current_state.R:.0f}"
        )
