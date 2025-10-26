from dataclasses import dataclass
import numpy as np


@dataclass
class EpidemicState:
    N: int  # Total population
    S: float  # Susceptible
    I: float  # Infected
    R: float  # Recovered


class SIR:
    def run_interval(self, state: EpidemicState, beta: float, gamma: float, days: int):
        """
        :param state: Current epidemic state (DTO)
        :param beta: Transmission rate
        :param gamma: Recovery rate
        :param days: Number of days to simulate
        :return: Arrays of (S, I, R) for each day (including initial state)
        """
        S, I, R = [state.S], [state.I], [state.R]

        for _ in range(days):
            S_current, I_current, R_current = S[-1], I[-1], R[-1]

            new_infections = (beta * S_current * I_current) / state.N
            new_recoveries = gamma * I_current

            S_next = S_current - new_infections
            I_next = I_current + new_infections - new_recoveries
            R_next = R_current + new_recoveries

            S.append(S_next)
            I.append(I_next)
            R.append(R_next)

        return np.array(S), np.array(I), np.array(R)

    def run_all_days(self, state: EpidemicState, beta: float, gamma: float, days: int):
        """
        :param state: Initial epidemic state
        :param beta: Transmission rate
        :param gamma: Recovery rate
        :param days: Number of days to simulate
        :return: A tuple of (t, S, I, R) arrays
        """
        S, I, R = self.run_interval(state, beta, gamma, days)
        t = np.linspace(0, days, days + 1)

        return t, S, I, R
