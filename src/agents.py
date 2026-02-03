import random
from enum import Enum
from src.config import DefaultConfig
import numpy as np


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


class ThresholdAgent(Agent):
    """
    Deterministic rule-based agent that selects intervention level based on infected fraction thresholds.
    
    The agent automatically detects the index of I (Infected) in the observation vector,
    accounting for partial observability when E (Exposed) is masked.
    
    Action selection logic:
    - NO (0): I/N < thresholds[0]
    - MILD (1): thresholds[0] <= I/N < thresholds[1]
    - MODERATE (2): thresholds[1] <= I/N < thresholds[2]
    - SEVERE (3): I/N >= thresholds[2]
    
    Attributes:
        config: Configuration object containing population size N and thresholds.
        thresholds: List of threshold values for infected fraction (I/N).
        i_idx: Index of I compartment in observation vector (auto-detected on first call).
    """
    
    def __init__(self, config: DefaultConfig):
        """
        Initialize ThresholdAgent.
        
        Args:
            config: Configuration object with N (population size) and thresholds.
        """
        self.config = config
        self.thresholds = config.thresholds
        
        if len(self.thresholds) != 3:
            raise ValueError(
                f"ThresholdAgent requires exactly 3 thresholds, got {len(self.thresholds)}"
            )
        
        # Validate thresholds are in ascending order
        if not all(self.thresholds[i] < self.thresholds[i+1] for i in range(len(self.thresholds)-1)):
            raise ValueError("Thresholds must be in ascending order")
        
        # I index will be auto-detected on first predict call
        self.i_idx = None
    
    def _detect_i_index(self, observation: np.ndarray) -> int:
        """
        Automatically detect the index of I (Infected) compartment in observation vector.
        
        Args:
            observation: Observation vector [S, E, I, R] or [S, I, R].
            
        Returns:
            Index of I compartment (1 if E is masked, 2 if E is included).
        """
        obs_len = len(observation)
        if obs_len == 4:
            # Full observability: [S, E, I, R] -> I is at index 2
            return 2
        elif obs_len == 3:
            # Partial observability: [S, I, R] -> I is at index 1
            return 1
        else:
            raise ValueError(
                f"Unexpected observation shape: {obs_len}. Expected 3 or 4 elements."
            )
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        Predict action based on infected fraction thresholds.
        
        Args:
            observation: np.ndarray [S, E, I, R] or [S, I, R] (if E is masked).
            state: Unused (for SB3 compatibility).
            episode_start: Unused (for SB3 compatibility).
            deterministic: Unused (agent is always deterministic).
            
        Returns:
            Tuple of (action_index, state) where action_index is 0-3.
        """
        observation = np.asarray(observation)
        
        # Auto-detect I index on first call or if observation shape changed
        if self.i_idx is None or len(observation) != (4 if self.i_idx == 2 else 3):
            self.i_idx = self._detect_i_index(observation)
        
        # Extract infected count
        I = observation[self.i_idx]
        
        # Calculate infected fraction
        infected_fraction = I / self.config.N
        
        # Map to action based on thresholds
        if infected_fraction < self.thresholds[0]:
            action_idx = 0  # NO
        elif infected_fraction < self.thresholds[1]:
            action_idx = 1  # MILD
        elif infected_fraction < self.thresholds[2]:
            action_idx = 2  # MODERATE
        else:
            action_idx = 3  # SEVERE
        
        return action_idx, state

