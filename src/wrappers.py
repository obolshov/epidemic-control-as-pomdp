from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import Config


class EpidemicObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that masks certain compartments of the epidemic state for POMDP scenarios.
    
    This wrapper allows partial observability by removing specific compartments from
    the observation vector. The observation_space is automatically adjusted to match
    the filtered observation shape, ensuring compatibility with Stable Baselines 3.
    
    Attributes:
        include_exposed: If False, the E (Exposed) compartment is masked from observations.
                        The observation will be [S, I, R] instead of [S, E, I, R].
    """

    def __init__(self, env: gym.Env, include_exposed: bool = True):
        """
        Initialize the observation wrapper.
        
        Args:
            env: The gymnasium environment to wrap.
            include_exposed: If True, include the E compartment in observations.
                           If False, mask the E compartment (index 1 in [S, E, I, R]).
        """
        super().__init__(env)
        self.include_exposed = include_exposed
        
        # Store original observation space properties
        original_space = env.observation_space
        if not isinstance(original_space, spaces.Box):
            raise ValueError(
                f"EpidemicObservationWrapper expects Box observation space, "
                f"got {type(original_space)}"
            )
        
        original_shape = original_space.shape
        if len(original_shape) != 1 or original_shape[0] != 4:
            raise ValueError(
                f"Expected observation shape (4,), got {original_shape}"
            )
        
        # Determine which indices to keep
        # Original: [S, E, I, R] = [0, 1, 2, 3]
        if include_exposed:
            self.keep_indices = np.array([0, 1, 2, 3])  # Keep all
        else:
            self.keep_indices = np.array([0, 2, 3])  # Remove E (index 1)
        
        # Update observation space
        new_shape = (len(self.keep_indices),)
        self.observation_space = spaces.Box(
            low=original_space.low[0],  # Same bounds for all compartments
            high=original_space.high[0],
            shape=new_shape,
            dtype=original_space.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Filter the observation by removing masked compartments.
        
        Args:
            obs: Original observation vector [S, E, I, R].
            
        Returns:
            Filtered observation vector. If include_exposed=False, returns [S, I, R].
        """
        if not self.include_exposed:
            # Remove E compartment (index 1) using np.delete
            filtered_obs = np.delete(obs, 1)
        else:
            filtered_obs = obs
        
        return filtered_obs.astype(self.observation_space.dtype)


class UnderReportingWrapper(gym.ObservationWrapper):
    """Simulates under-reporting by scaling the observed I and R compartments.

    In reality, only a fraction of infected (and recovered) individuals are
    officially detected via testing. The agent sees k*I and k*R instead of the
    true values. COVID-19 research suggests k ≈ 0.1–0.3 depending on the country.

    Must be applied AFTER EpidemicObservationWrapper (if E masking is used),
    since it infers I and R indices from the current observation shape.

    Args:
        env: Wrapped environment. Observation shape must be (3,) [S, I, R]
             or (4,) [S, E, I, R].
        detection_rate: Fraction of true I and R observed. Must be in (0.0, 1.0].
                        1.0 = full observation (no distortion); 0.3 = COVID-realistic.
    """

    def __init__(self, env: gym.Env, detection_rate: float = 1.0) -> None:
        super().__init__(env)
        if not (0.0 < detection_rate <= 1.0):
            raise ValueError(f"detection_rate must be in (0, 1], got {detection_rate}")
        self.detection_rate = detection_rate

        obs_size = env.observation_space.shape[0]
        if obs_size == 4:    # [S, E, I, R]
            self.i_index = 2
            self.r_index = 3
        elif obs_size == 3:  # [S, I, R] — E already masked
            self.i_index = 1
            self.r_index = 2
        else:
            raise ValueError(
                f"Unexpected observation size {obs_size}; expected 3 ([S, I, R]) or 4 ([S, E, I, R])."
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Scale I and R compartments by detection_rate.

        Args:
            obs: Observation vector of shape (3,) or (4,).

        Returns:
            Observation with I and R scaled by detection_rate. Shape unchanged.
        """
        scaled = obs.copy()
        scaled[self.i_index] *= self.detection_rate
        scaled[self.r_index] *= self.detection_rate
        return scaled.astype(self.observation_space.dtype)


class MultiplicativeNoiseWrapper(gym.ObservationWrapper):
    """Simulates per-compartment multiplicative measurement noise.

    Each compartment is multiplied by an independent noise factor:
        obs_noisy[i] = obs[i] * (1 + N(0, noise_stds[i]))

    Typical noise levels by compartment (epidemiological motivation):
    - S: 0.05 — population size is well known
    - E: 0.30 — pre-symptomatic, rarely directly detected (only when E is observed)
    - I: 0.30 — false positive/negative testing
    - R: 0.15 — incomplete recovery statistics

    Must be applied LAST (after EpidemicObservationWrapper and
    UnderReportingWrapper). The length of noise_stds must match the
    current observation shape (3 or 4) — a ValueError is raised otherwise.

    Args:
        env: Wrapped environment. Observation shape must be (3,) or (4,).
        noise_stds: Per-compartment noise stds matching the current observation
                    shape. E.g. [0.05, 0.30, 0.15] for [S, I, R] or
                    [0.05, 0.30, 0.30, 0.15] for [S, E, I, R].
    """

    def __init__(self, env: gym.Env, noise_stds: List[float]) -> None:
        super().__init__(env)
        obs_size = env.observation_space.shape[0]
        if len(noise_stds) != obs_size:
            raise ValueError(
                f"noise_stds length ({len(noise_stds)}) must match observation "
                f"size ({obs_size}). Current obs shape: {env.observation_space.shape}"
            )
        if any(s < 0 for s in noise_stds):
            raise ValueError(f"All noise_stds must be >= 0, got {noise_stds}")
        self._noise_stds = np.array(noise_stds, dtype=np.float32)
        self._pop_size: float = float(env.observation_space.high[0])

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply per-compartment multiplicative noise and clip.

        Args:
            obs: Observation vector of shape (3,) or (4,).

        Returns:
            Noisy observation clipped to [0, N]. Shape unchanged.
        """
        noise_factors = 1.0 + self.np_random.normal(0.0, self._noise_stds).astype(np.float32)
        noisy = obs * noise_factors
        return np.clip(noisy, 0.0, self._pop_size).astype(self.observation_space.dtype)


def create_environment(config: Config, pomdp_params: Dict[str, Any]) -> gym.Env:
    """
    Create environment with appropriate POMDP wrappers.

    Args:
        config: Base configuration.
        pomdp_params: POMDP parameters for wrappers.

    Returns:
        Environment with wrappers applied.
    """
    from src.env import EpidemicEnv

    env = EpidemicEnv(config)

    if not pomdp_params.get("include_exposed", True):
        env = EpidemicObservationWrapper(env, include_exposed=False)

    detection_rate = pomdp_params.get("detection_rate", 1.0)
    if detection_rate < 1.0:
        env = UnderReportingWrapper(env, detection_rate=detection_rate)

    noise_stds = pomdp_params.get("noise_stds")
    if noise_stds is not None:
        env = MultiplicativeNoiseWrapper(env, noise_stds=noise_stds)

    return env
