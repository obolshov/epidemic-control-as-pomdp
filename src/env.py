from typing import List
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import Config
from src.seir import run_seir, EpidemicState
from src.agents import InterventionAction, Agent, StaticAgent


def calculate_reward(
    I_t: float, action: InterventionAction, config: Config
) -> float:
    """
    :param I_t: Infected count
    :param action: Action taken
    :return: Reward value
    """
    infection_penalty = config.w_I * (I_t / config.N) ** 2
    stringency_penalty = config.w_S * (1 - action.value)

    return -(infection_penalty + stringency_penalty)


class EpidemicEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.action_space = spaces.Discrete(len(InterventionAction))
        self.action_map = list(InterventionAction)

        self.observation_space = spaces.Box(
            low=0, high=self.config.N, shape=(4,), dtype=np.float32
        )

        self.current_state = None
        self.current_day = 0

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

        return self._get_obs(), {}

    def step(self, action_idx):
        action_enum = self.action_map[action_idx]
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

        reward = calculate_reward(self.current_state.I, action_enum, self.config)

        self.current_day += days_to_simulate

        terminated = self.current_day >= self.config.days
        truncated = False

        info = {"S": S, "E": E, "I": I, "R": R}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(
            [
                self.current_state.S,
                self.current_state.E,
                self.current_state.I,
                self.current_state.R,
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
