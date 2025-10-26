import random
from src.sir import EpidemicState
from enum import Enum


class InterventionAction(Enum):
    NO = 1.0
    MILD = 0.75
    MODERATE = 0.5
    SEVERE = 0.25


class Agent:
    """
    Base class for epidemic control agents.
    """

    def select_action(self, state: EpidemicState) -> InterventionAction:
        raise NotImplementedError("Subclasses must implement select_action")


class StaticAgent(Agent):
    def __init__(self, action: InterventionAction):
        self.action = action

    def select_action(self, state: EpidemicState) -> InterventionAction:
        return self.action


class RandomAgent(Agent):
    def select_action(self, state: EpidemicState) -> InterventionAction:
        return random.choice(list(InterventionAction))
