import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
