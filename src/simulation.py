from typing import Dict, List
import numpy as np
from .actions import InterventionAction
from .sir import EpidemicState, run_sir_step
from .agents import Agent


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
    ):
        self.agent = agent
        self.t = t
        self.S = S
        self.I = I
        self.R = R
        self.actions = actions
        self.action_timesteps = action_timesteps

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
        return self.agent.__class__.__name__


def run_simulation(
    agent: Agent,
    initial_state: EpidemicState,
    beta_0: float,
    gamma: float,
    total_days: int,
    action_interval: int = 7,
) -> SimulationResult:
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
    current_state = initial_state

    all_S = [current_state.S]
    all_I = [current_state.I]
    all_R = [current_state.R]

    actions_taken = []
    action_timesteps = []

    current_day = 0

    while current_day < total_days:
        action = agent.select_action(current_state)
        actions_taken.append(action)
        action_timesteps.append(current_day)

        beta = action.apply_to_beta(beta_0)

        days_to_simulate = min(action_interval, total_days - current_day)

        S, I, R = run_sir_step(current_state, beta, gamma, days_to_simulate)

        all_S.extend(S[1:])
        all_I.extend(I[1:])
        all_R.extend(R[1:])

        current_state = EpidemicState(N=current_state.N, S=S[-1], I=I[-1], R=R[-1])
        current_day += days_to_simulate

    t = np.arange(len(all_S))

    return SimulationResult(
        agent=agent,
        t=t,
        S=np.array(all_S),
        I=np.array(all_I),
        R=np.array(all_R),
        actions=actions_taken,
        action_timesteps=action_timesteps,
    )
