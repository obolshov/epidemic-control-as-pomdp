from dataclasses import dataclass
import numpy as np


@dataclass
class EpidemicState:
    N: int  # Total population
    S: float  # Susceptible
    E: float  # Exposed
    I: float  # Infected
    R: float  # Recovered


def run_seir(state: EpidemicState, beta: float, sigma: float, gamma: float, days: int):
    """
    Simulates SEIR model for a given number of days.

    :param state: Current epidemic state (DTO)
    :param beta: Transmission rate
    :param sigma: Incubation rate
    :param gamma: Recovery rate
    :param days: Number of days to simulate
    :return: Arrays of (S, E, I, R) for each day (excluding initial state)
    """
    S, E, I, R = [], [], [], []
    S_current, E_current, I_current, R_current = state.S, state.E, state.I, state.R

    for _ in range(days):
        new_exposed = (beta * S_current * I_current) / state.N
        new_infections = sigma * E_current
        new_recoveries = gamma * I_current

        S_next = S_current - new_exposed
        E_next = E_current + new_exposed - new_infections
        I_next = I_current + new_infections - new_recoveries
        R_next = R_current + new_recoveries

        S.append(S_next)
        E.append(E_next)
        I.append(I_next)
        R.append(R_next)

        S_current, E_current, I_current, R_current = S_next, E_next, I_next, R_next

    return np.array(S), np.array(E), np.array(I), np.array(R)

