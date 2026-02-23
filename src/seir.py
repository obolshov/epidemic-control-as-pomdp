from dataclasses import dataclass
import numpy as np


@dataclass
class EpidemicState:
    N: int  # Total population
    S: float  # Susceptible
    E: float  # Exposed
    I: float  # Infected
    R: float  # Recovered


def run_seir(
    state: EpidemicState,
    beta: float,
    sigma: float,
    gamma: float,
    days: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates SEIR model for a given number of days.

    Supports two modes:
    - Deterministic (rng=None): classic discrete-time ODE update.
    - Stochastic (rng provided): Binomial transitions. 

    Args:
        state: Current epidemic state.
        beta: Transmission rate.
        sigma: Incubation rate.
        gamma: Recovery rate.
        days: Number of days to simulate.
        rng: NumPy Generator for stochastic mode. If None, runs deterministically.

    Returns:
        Tuple of (S, E, I, R) arrays of shape (days,), excluding initial state.
    """
    S, E, I, R = [], [], [], []
    S_current, E_current, I_current, R_current = state.S, state.E, state.I, state.R

    for _ in range(days):
        if rng is None:
            # Deterministic path: classic discrete-time mass-action update
            new_exposed = (beta * S_current * I_current) / state.N
            new_infections = sigma * E_current
            new_recoveries = gamma * I_current

            S_next = S_current - new_exposed
            E_next = E_current + new_exposed - new_infections
            I_next = I_current + new_infections - new_recoveries
            R_next = R_current + new_recoveries
        else:
            # Stochastic path: Binomial transitions
            S_int = int(round(S_current))
            E_int = int(round(E_current))
            I_int = int(round(I_current))
            R_int = int(round(R_current))

            # Transition probabilities derived from exponential waiting times
            p_SE = 1.0 - np.exp(-beta * I_int / state.N)  # force of infection
            p_EI = 1.0 - np.exp(-sigma)
            p_IR = 1.0 - np.exp(-gamma)

            # Binomial draws
            dSE = rng.binomial(S_int, p_SE)
            dEI = rng.binomial(E_int, p_EI)
            dIR = rng.binomial(I_int, p_IR)

            S_next = S_int - dSE
            E_next = E_int + dSE - dEI
            I_next = I_int + dEI - dIR
            R_next = R_int + dIR

        S.append(S_next)
        E.append(E_next)
        I.append(I_next)
        R.append(R_next)

        S_current, E_current, I_current, R_current = S_next, E_next, I_next, R_next

    return np.array(S), np.array(E), np.array(I), np.array(R)

