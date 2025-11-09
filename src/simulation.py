from typing import List
import numpy as np
from .agents import InterventionAction, Agent, StaticAgent
from .sir import EpidemicState, SIR
from .config import DefaultConfig


def calculate_reward(I_t: float, I_t1: float, action: InterventionAction, config: DefaultConfig) -> float:
    """
    Calculate reward based on infection change and action stringency.

    :param I_t: Infected count at time t
    :param I_t1: Infected count at time t+1
    :param action: Action taken
    :return: Reward value
    """
    epislon = 1e-6 * config.N
    infection_penalty = max(0.0, np.log((I_t1 + epislon) / (I_t + epislon)))

    action_stringency = (1 - action.value) ** 2
    stringency_penalty = config.w_S * action_stringency

    return -(infection_penalty + stringency_penalty)


class SimulationResult:
    def __init__(
        self,
        agent: Agent,
        t: np.ndarray,
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
        actions: List[InterventionAction],
        action_timesteps: List[int],
        rewards: List[float],
        action_states: List[EpidemicState],
    ):
        self.agent = agent
        self.t = t
        self.S = S
        self.I = I
        self.R = R
        self.actions = actions
        self.action_timesteps = action_timesteps
        self.rewards = rewards
        self.action_states = action_states

    @property
    def peak_infected(self) -> float:
        return np.max(self.I)

    @property
    def total_infected(self) -> float:
        return self.R[-1] + self.I[-1]

    @property
    def epidemic_duration(self) -> int:
        days_above_one = np.where(self.I >= 1.0)[0]
        return days_above_one[-1] if len(days_above_one) > 0 else 0

    @property
    def agent_name(self) -> str:
        if isinstance(self.agent, StaticAgent):
            return self.agent.__class__.__name__ + " - " + self.agent.action.name

        return self.agent.__class__.__name__

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


class Simulation:
    def __init__(
        self,
        agent: Agent,
        config: DefaultConfig,
    ):
        self.agent = agent
        self.config = config
        S0 = config.N - config.I0 - config.R0
        self.initial_state = EpidemicState(N=config.N, S=S0, I=config.I0, R=config.R0)

    def run(self) -> SimulationResult:
        """
        Runs epidemic simulation with an agent that selects actions at regular intervals.

        :param agent: Agent that selects actions based on state
        :param initial_state: Initial epidemic state
        :param beta_0: Base transmission rate (without interventions)
        :param gamma: Recovery rate
        :param total_days: Total simulation period
        :param action_interval: Number of days between agent decisions (default: 7)
        :return: SimulationResult with complete trajectory
        """
        current_state = self.initial_state

        all_S = [current_state.S]
        all_I = [current_state.I]
        all_R = [current_state.R]

        actions_taken = []
        action_timesteps = []
        rewards = []
        action_states = []

        current_day = 0

        sir = SIR()

        while current_day < self.config.days:
            action_states.append(current_state)

            action = self.agent.select_action(current_state)
            actions_taken.append(action)
            action_timesteps.append(current_day)

            beta = self.apply_action_to_beta(action)

            days_to_simulate = min(self.config.action_interval, self.config.days - current_day)

            I_before = current_state.I

            S, I, R = sir.run_interval(
                current_state, beta, self.config.gamma, days_to_simulate
            )

            all_S.extend(S[1:])
            all_I.extend(I[1:])
            all_R.extend(R[1:])

            current_state = EpidemicState(N=current_state.N, S=S[-1], I=I[-1], R=R[-1])

            I_after = current_state.I
            reward = calculate_reward(I_before, I_after, action, self.config)
            rewards.append(reward)

            current_day += days_to_simulate

        t = np.arange(len(all_S))

        return SimulationResult(
            agent=self.agent,
            t=t,
            S=np.array(all_S),
            I=np.array(all_I),
            R=np.array(all_R),
            actions=actions_taken,
            action_timesteps=action_timesteps,
            rewards=rewards,
            action_states=action_states,
        )

    def apply_action_to_beta(self, action: InterventionAction) -> float:
        return self.config.beta_0 * action.value
