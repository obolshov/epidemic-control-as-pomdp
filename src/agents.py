import random
from enum import Enum


class InterventionAction(Enum):
    NO = 1.0
    MILD = 0.75
    MODERATE = 0.5
    SEVERE = 0.25


class Agent:
    """
    Base class for epidemic control agents.
    Compatible with Stable Baselines3 interface.
    """

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        :param observation: np.ndarray [S, E, I, R] or [S, I, R] (if E is masked)
        :return: (action_index, state)
        """
        raise NotImplementedError("Subclasses must implement predict")


class StaticAgent(Agent):
    def __init__(self, action: InterventionAction):
        self.action_idx = list(InterventionAction).index(action)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return self.action_idx, state


class RandomAgent(Agent):
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        action_idx = random.choice(range(len(InterventionAction)))
        return action_idx, state

