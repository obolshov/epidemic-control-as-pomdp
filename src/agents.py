from typing import Any, Dict, List, Optional

from src.config import Config
from src.env import InterventionAction
import numpy as np


class Agent:
    """
    Base class for epidemic control agents.
    Compatible with Stable Baselines3 interface.
    """

    name: str  # set by subclasses; used as result dict key in evaluation

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        :param observation: np.ndarray [S, E, I, R] or [S, I, R] (if E is masked)
        :return: (action_index, state)
        """
        raise NotImplementedError("Subclasses must implement predict")


class StaticAgent(Agent):
    def __init__(self, action: InterventionAction, name: Optional[str] = None):
        self.action_idx = list(InterventionAction).index(action)
        self.name = name or action.name.lower()

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return self.action_idx, state


class RandomAgent(Agent):
    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.name = "random"

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        action_idx = int(self._rng.integers(len(InterventionAction)))
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

    def __init__(
        self,
        config: Config,
        name: str = "threshold",
        detection_rate: float = 1.0,
    ):
        """
        Initialize ThresholdAgent.

        Args:
            config: Configuration object with N (population size) and thresholds.
            name: Agent name used as result dict key in evaluation.
            detection_rate: Nominal detection rate for underreporting compensation.
                When < 1.0, observed I is divided by detection_rate before threshold
                comparison, compensating for the known underreporting factor.
        """
        self.config = config
        self.thresholds = config.thresholds
        self.name = name
        self.detection_rate = detection_rate

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
        if obs_len == 6:
            # Full observability: [S, E, I, R, prev_action, day_frac] -> I is at index 2
            return 2
        elif obs_len == 5:
            # Partial observability: [S, I, R, prev_action, day_frac] -> I is at index 1
            return 1
        else:
            raise ValueError(
                f"Unexpected observation shape: {obs_len}. Expected 5 or 6 elements."
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

        # Auto-detect I index on first call (obs shape is constant within an episode)
        if self.i_idx is None:
            self.i_idx = self._detect_i_index(observation)

        # Extract infected count and compensate for underreporting
        I = observation[self.i_idx]
        I_corrected = I / self.detection_rate if self.detection_rate < 1.0 else I

        # Calculate infected fraction
        infected_fraction = I_corrected / self.config.N

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


def create_baseline_agents(
    config: Config,
    agent_names: List[str],
    pomdp_params: Optional[Dict[str, Any]] = None,
) -> List[Agent]:
    """Initialize non-RL baseline agents for evaluation.

    Args:
        config: Configuration for agents.
        agent_names: List of agent names to initialize.
        pomdp_params: POMDP wrapper parameters. Used to extract detection_rate
            for ThresholdAgent underreporting compensation.

    Returns:
        List of initialized Agent objects.
    """
    agents: List[Agent] = []
    if "no_action" in agent_names:
        agents.append(StaticAgent(InterventionAction.NO, name="no_action"))
    if "severe" in agent_names:
        agents.append(StaticAgent(InterventionAction.SEVERE, name="severe"))
    if "random" in agent_names:
        agents.append(RandomAgent())
    if "threshold" in agent_names:
        # ThresholdAgent can't see the per-episode latent detection_rate — use range midpoint as prior.
        low, high = (pomdp_params or {}).get("detection_rate", (1.0, 1.0))
        agents.append(ThresholdAgent(config, detection_rate=0.5 * (float(low) + float(high))))
    return agents
