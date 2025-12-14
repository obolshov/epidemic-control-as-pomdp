from dataclasses import dataclass

import numpy as np


@dataclass
class EpidemicState:
    N: int  # Total population
    S: float  # Susceptible
    I: float  # Infected
    R: float  # Recovered


def run_sir(state: EpidemicState, beta: float, gamma: float, days: int):
    """
    Simulates SIR model for a given number of days.

    :param state: Current epidemic state (DTO)
    :param beta: Transmission rate
    :param gamma: Recovery rate
    :param days: Number of days to simulate
    :return: Arrays of (S, I, R) for each day (excluding initial state)
    """
    S, I, R = [], [], []
    S_current, I_current, R_current = state.S, state.I, state.R

    for _ in range(days):
        new_infections = (beta * S_current * I_current) / state.N
        new_recoveries = gamma * I_current

        S_next = S_current - new_infections
        I_next = I_current + new_infections - new_recoveries
        R_next = R_current + new_recoveries

        S.append(S_next)
        I.append(I_next)
        R.append(R_next)

        S_current, I_current, R_current = S_next, I_next, R_next

    return np.array(S), np.array(I), np.array(R)
