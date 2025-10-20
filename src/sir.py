import numpy as np


def run_sir_model(N: int, I0: int, R0: int, beta: float, gamma: float, days: int):
    """
    Runs a simple SIR model simulation.

    :param N: Total population
    :param I0: Initial number of infected individuals
    :param R0: Initial number of recovered individuals
    :param beta: Transmission rate
    :param gamma: Recovery rate
    :param days: Number of days to simulate
    :return: A tuple of (t, S, I, R) arrays
    """
    S0 = N - I0 - R0

    S, I, R = [S0], [I0], [R0]

    t = np.linspace(0, days, days + 1)

    for _ in range(days):
        S_current, I_current, R_current = S[-1], I[-1], R[-1]

        new_infections = (beta * S_current * I_current) / N
        new_recoveries = gamma * I_current

        S_next = S_current - new_infections
        I_next = I_current + new_infections - new_recoveries
        R_next = R_current + new_recoveries

        S.append(S_next)
        I.append(I_next)
        R.append(R_next)

    return t, np.array(S), np.array(I), np.array(R)
