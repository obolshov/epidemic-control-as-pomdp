from enum import Enum


class InterventionAction(Enum):
    NO = 1.0
    MILD = 0.75
    MODERATE = 0.5
    SEVERE = 0.25

    def apply_to_beta(self, beta_0: float) -> float:
        return beta_0 * self.value
