import random
from src.actions import InterventionAction
from src.sir import EpidemicState


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
