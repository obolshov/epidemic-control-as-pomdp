from typing import Dict
import numpy as np
from .actions import InterventionAction


class SimulationResult:
    """
    Container for a single simulation run results.
    """

    def __init__(
        self,
        action: InterventionAction,
        beta: float,
        t: np.ndarray,
        S: np.ndarray,
        I: np.ndarray,
        R: np.ndarray,
    ):
        self.action = action
        self.beta = beta
        self.t = t
        self.S = S
        self.I = I
        self.R = R

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


class EpidemicPolicy:
    """
    Base class for epidemic control policies.
    """

    def select_action(self, state: Dict) -> InterventionAction:
        """
        Selects an intervention action based on the current epidemic state.

        :param state: Dictionary with epidemic state (e.g., S, I, R values)
        :return: Selected intervention action
        """
        raise NotImplementedError("Subclasses must implement select_action")


class StaticPolicy(EpidemicPolicy):
    """
    A policy that always returns the same action.
    """

    def __init__(self, action: InterventionAction):
        self.action = action

    def select_action(self, state: Dict) -> InterventionAction:
        return self.action


class ThresholdPolicy(EpidemicPolicy):
    """
    A simple rule-based policy that selects actions based on infection thresholds.
    """

    def __init__(self, thresholds: Dict[str, float]):
        """
        :param thresholds: Dict with 'mild', 'moderate', 'severe' infection thresholds
        """
        self.thresholds = thresholds

    def select_action(self, state: Dict) -> InterventionAction:
        infected_pct = state["I"] / state["N"]

        if infected_pct >= self.thresholds.get("severe", 0.3):
            return InterventionAction.SEVERE
        elif infected_pct >= self.thresholds.get("moderate", 0.15):
            return InterventionAction.MODERATE
        elif infected_pct >= self.thresholds.get("mild", 0.05):
            return InterventionAction.MILD
        else:
            return InterventionAction.NO
