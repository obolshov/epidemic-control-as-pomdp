from collections import deque
from typing import Any, Dict, List, Optional

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
        if len(original_shape) != 1 or original_shape[0] != 6:
            raise ValueError(
                f"Expected observation shape (6,), got {original_shape}"
            )

        # Determine which indices to keep
        # Original: [S, E, I, R, prev_action, day_frac] = [0, 1, 2, 3, 4, 5]
        if include_exposed:
            self.keep_indices = np.array([0, 1, 2, 3, 4, 5])  # Keep all
        else:
            self.keep_indices = np.array([0, 2, 3, 4, 5])  # Remove E (index 1)

        # Update observation space with per-element bounds
        new_low = original_space.low[self.keep_indices]
        new_high = original_space.high[self.keep_indices]
        self.observation_space = spaces.Box(
            low=new_low,
            high=new_high,
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
    officially detected via testing. The agent sees k_eff*I and k_eff*R instead
    of the true values. Undetected individuals are redistributed into S,
    preserving population accounting and preventing the agent from inferring
    true infection levels via the S dynamics channel.

    Supports optional testing capacity saturation: when the infected fraction
    exceeds the testing system's throughput, k_effective drops below k_base.
    This models real-world testing bottlenecks observed during COVID-19
    surges (Lau et al. 2021).

    The effective detection rate follows Michaelis-Menten saturation kinetics:
        k_eff(I) = k_base * r / (k_base * (I/N) + r)
    where r is the testing capacity ratio (fraction of population testable per
    day). When I/N is small, k_eff ≈ k_base. When I/N >> r/k_base, I_obs
    plateaus at ~r*N (testing ceiling). The formula is scale-invariant —
    it depends only on the infected fraction, not absolute population size.

    Per-episode stochastic sampling: ``detection_rate`` and ``testing_capacity``
    are given as ``(low, high)`` tuples. On each ``reset()``, a fresh value is
    sampled from ``U[low, high]`` and held constant for the whole episode.
    This turns underreporting into a latent POMDP parameter that memory-based
    agents must infer from trajectory dynamics. To keep a fixed value across
    episodes, pass a degenerate range like ``(0.25, 0.25)``.

    Must be applied AFTER EpidemicObservationWrapper (if E masking is used),
    since it infers I and R indices from the current observation shape.

    Args:
        env: Wrapped environment. Observation shape must be (3,) [S, I, R]
             or (4,) [S, E, I, R].
        detection_rate: ``(low, high)`` range for k_base. Both bounds must be
                        in (0.0, 1.0] and low ≤ high. Default ``(1.0, 1.0)``
                        means no distortion.
        testing_capacity: ``(low, high)`` range for the fraction of the
                          population testable per day, or None for no
                          saturation (detection_rate constant across I). Both
                          bounds must be > 0 and low ≤ high.
        seed: RNG seed for per-episode sampling (default 42).
    """

    S_INDEX = 0  # S is always the first compartment

    def __init__(
        self,
        env: gym.Env,
        detection_rate: tuple = (1.0, 1.0),
        testing_capacity: Optional[tuple] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(env)
        self._detection_range = self._validate_range(
            detection_rate, name="detection_rate", high_bound=1.0,
        )
        self._testing_range: Optional[tuple] = (
            None if testing_capacity is None
            else self._validate_range(testing_capacity, name="testing_capacity")
        )
        self._rng = np.random.default_rng(seed)
        self.detection_rate: float = self._detection_range[0]
        self.testing_capacity: Optional[float] = (
            self._testing_range[0] if self._testing_range is not None else None
        )
        self._pop_size: float = float(env.observation_space.high[0])

        obs_size = env.observation_space.shape[0]
        if obs_size == 6:    # [S, E, I, R, prev_action, day_frac]
            self.i_index = 2
            self.r_index = 3
        elif obs_size == 5:  # [S, I, R, prev_action, day_frac] — E already masked
            self.i_index = 1
            self.r_index = 2
        else:
            raise ValueError(
                f"Unexpected observation size {obs_size}; expected 5 or 6."
            )

    @staticmethod
    def _validate_range(value, name: str, high_bound: Optional[float] = None) -> tuple:
        """Validate a ``(low, high)`` sampling range with ``0 < low <= high``."""
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError(
                f"{name} must be a (low, high) tuple, got {type(value).__name__}={value!r}"
            )
        low, high = float(value[0]), float(value[1])
        if low <= 0.0:
            raise ValueError(f"{name} low bound must be > 0, got {low}")
        if high_bound is not None and high > high_bound:
            raise ValueError(f"{name} high bound must be <= {high_bound}, got {high}")
        if low > high:
            raise ValueError(f"{name} low ({low}) must be <= high ({high})")
        return (low, high)

    def reset(self, **kwargs):
        # Sample before super().reset() so the chained observation() sees fresh values.
        self.detection_rate = float(self._rng.uniform(*self._detection_range))
        if self._testing_range is not None:
            self.testing_capacity = float(self._rng.uniform(*self._testing_range))
        return super().reset(**kwargs)

    def _effective_rate(self, true_I: float) -> float:
        """Compute effective detection rate accounting for testing saturation.

        Uses Michaelis-Menten kinetics in fraction form:
            k_eff = k_base * r / (k_base * (I/N) + r)
        When testing_capacity is None, returns k_base unchanged.

        Args:
            true_I: True infected count (before any scaling).

        Returns:
            Effective detection rate in (0, k_base].
        """
        if self.testing_capacity is None:
            return self.detection_rate
        k = self.detection_rate
        r = self.testing_capacity
        infected_fraction = true_I / self._pop_size
        return k * r / (k * infected_fraction + r)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Scale I and R by effective detection rate, redistribute undetected into S.

        Undetected individuals are absorbed into the susceptible pool, matching
        real-world surveillance where unconfirmed cases remain in the "healthy"
        population statistics.

        Args:
            obs: Observation vector of shape (3,) or (4,).

        Returns:
            Observation with I and R scaled down, S inflated by undetected. Shape unchanged.
        """
        scaled = obs.copy()
        true_I = scaled[self.i_index]
        k_eff = self._effective_rate(true_I)
        hidden_I = scaled[self.i_index] * (1.0 - k_eff)
        hidden_R = scaled[self.r_index] * (1.0 - k_eff)
        scaled[self.i_index] *= k_eff
        scaled[self.r_index] *= k_eff
        scaled[self.S_INDEX] += hidden_I + hidden_R
        return scaled.astype(self.observation_space.dtype)


class MultiplicativeNoiseWrapper(gym.ObservationWrapper):
    """Simulates per-compartment multiplicative measurement noise via AR(1).

    Each compartment accumulates an autocorrelated bias that evolves as:
        bias_t[i] = ρ * bias_{t-1}[i] + √(1 - ρ²) * ε_t[i],  ε_t ~ N(0, σ_i)
        obs_noisy[i] = obs[i] * (1 + bias_t[i])

    The √(1 - ρ²) scaling preserves the marginal variance at σ² regardless of ρ.
    When ρ=0 this reduces to iid multiplicative noise (backward compatible).
    When ρ>0 the bias persists across steps, creating measurement drift that
    rewards memory-based agents who can track and compensate for it.

    Typical noise levels by compartment (epidemiological motivation):
    - S: 0.05 — population size is well known
    - E: 0.30 — pre-symptomatic, rarely directly detected (only when E is observed)
    - I: 0.30 — false positive/negative testing
    - R: 0.15 — incomplete recovery statistics

    Must be applied AFTER EpidemicObservationWrapper and UnderReportingWrapper,
    but BEFORE TemporalLagWrapper — noise is physically generated at measurement
    time, so the lag buffer holds already-noisy observations. The length of
    noise_stds must match the current observation shape (3 or 4) — a ValueError
    is raised otherwise.

    Args:
        env: Wrapped environment. Observation shape must be (3,) or (4,).
        noise_stds: Per-compartment noise stds matching the current observation
                    shape. E.g. [0.05, 0.30, 0.15] for [S, I, R] or
                    [0.05, 0.30, 0.30, 0.15] for [S, E, I, R].
        noise_rho: AR(1) autocorrelation coefficient in [0, 1). Controls how
                   persistent the measurement bias is across steps.
                   0.0 = iid noise (default), 0.7 = decorrelation half-life ≈ 2 steps.
    """

    def __init__(self, env: gym.Env, noise_stds: List[float], noise_rho: float = 0.0) -> None:
        super().__init__(env)
        obs_size = env.observation_space.shape[0]
        n_compartments = obs_size - 2  # Exclude prev_action and day_frac
        if len(noise_stds) != n_compartments:
            raise ValueError(
                f"noise_stds length ({len(noise_stds)}) must match compartment "
                f"count ({n_compartments}) for obs size {obs_size}."
            )
        if any(s < 0 for s in noise_stds):
            raise ValueError(f"All noise_stds must be >= 0, got {noise_stds}")
        if not (0.0 <= noise_rho < 1.0):
            raise ValueError(f"noise_rho must be in [0, 1), got {noise_rho}")
        self._noise_stds = np.array(noise_stds, dtype=np.float32)
        self._noise_rho = noise_rho
        self._innovation_scale = np.sqrt(1.0 - noise_rho ** 2)
        self._bias = np.zeros(n_compartments, dtype=np.float32)
        self._pop_size: float = float(env.observation_space.high[0])

    def reset(self, **kwargs):
        """Reset AR(1) bias state and delegate to wrapped environment."""
        self._bias = np.zeros(len(self._noise_stds), dtype=np.float32)
        return super().reset(**kwargs)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply per-compartment AR(1) multiplicative noise and clip.

        Noise is applied only to the epidemic compartments (all elements except
        the trailing prev_action and day_frac). Those two pass through unchanged.

        Args:
            obs: Observation vector of shape (5,) or (6,).

        Returns:
            Noisy observation with compartments clipped to [0, N]. Shape unchanged.
        """
        n_compartments = len(self._noise_stds)
        result = obs.copy().astype(np.float32)
        innovation = self.np_random.normal(0.0, self._noise_stds).astype(np.float32)
        self._bias = self._noise_rho * self._bias + self._innovation_scale * innovation
        result[:n_compartments] = np.clip(
            obs[:n_compartments] * (1.0 + self._bias), 0.0, self._pop_size
        )
        return result.astype(self.observation_space.dtype)


class TemporalLagWrapper(gym.ObservationWrapper):
    """Simulates delayed reporting with FIFO monotonicity and a random-walk lag.

    The lag evolves as a random walk: L_t = clip(L_{t-1} + noise, min_lag, max_lag)
    where noise ~ DiscreteUniform{-1, 0, 1}. The observation pointer never goes
    backward (FIFO monotonicity), so the agent never sees older data after having
    seen newer data — matching real-world batch-reporting behaviour (e.g. weekends
    hold, Mondays catch up by jumping forward).

    During warmup (first min_lag steps), the pointer clamps to 0 so the agent
    sees the episode's first observation until enough history accumulates.

    NOTE: Lag is specified in **environment steps**, not days. Since one step
    corresponds to action_interval days (default 5), use create_environment()
    which accepts lag in days and converts automatically.

    Applied AFTER MultiplicativeNoiseWrapper so the FIFO buffer stores already-
    noisy observations. This matches reality: measurement noise is baked in at
    collection time, not re-sampled when the agent reads a stale report.

    Args:
        env: Wrapped environment.
        min_lag: Minimum lag in **steps** (default 1 → 5 days with action_interval=5).
        max_lag: Maximum lag in **steps** (default 3 → ~15 days with action_interval=5).
        seed: RNG seed for reproducible lag sampling (default 42).
    """

    def __init__(self, env: gym.Env, min_lag: int = 1, max_lag: int = 3, seed: int = 42) -> None:
        super().__init__(env)
        if min_lag < 1:
            raise ValueError(f"min_lag must be >= 1, got {min_lag}")
        if max_lag < min_lag:
            raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
        self.min_lag = min_lag
        self.max_lag = max_lag
        self._rng = np.random.default_rng(seed)
        # maxlen=max_lag+1 keeps obs[t-max_lag] in buffer when obs[t] is appended
        self._cache: deque = deque(maxlen=max_lag + 1)
        self._current_lag: int = min_lag
        self._last_pointer: int = -1
        self._step: int = 0

    def _advance_lag(self) -> None:
        """Advance lag by one random-walk step: noise ~ DiscreteUniform{-1, 0, 1}."""
        noise = int(self._rng.integers(-1, 2))
        self._current_lag = int(np.clip(self._current_lag + noise, self.min_lag, self.max_lag))

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Push obs into buffer and return the observation at the FIFO-monotone pointer.

        Args:
            obs: Current true observation vector.

        Returns:
            Observation from the FIFO-monotone delayed pointer.
        """
        self._cache.append(obs.copy())
        t = self._step
        self._step += 1

        available_index = t - self._current_lag
        # FIFO: pointer never goes backward
        pointer = max(self._last_pointer, available_index)
        # Warmup safety: clamp into [0, t]
        pointer = int(np.clip(pointer, 0, t))
        self._last_pointer = pointer

        oldest_in_buffer = max(0, t - self.max_lag)
        buffer_idx = max(0, pointer - oldest_in_buffer)
        return self._cache[buffer_idx].copy()

    def step(self, action):
        """Advance the lag random walk before the environment steps."""
        self._advance_lag()
        return super().step(action)

    def reset(self, **kwargs):
        """Clear buffer and state, sample a new initial lag, then reset env."""
        self._cache.clear()
        self._current_lag = int(self._rng.integers(self.min_lag, self.max_lag + 1))
        self._last_pointer = -1
        self._step = 0
        return super().reset(**kwargs)


def create_environment(config: Config, pomdp_params: Dict[str, Any], seed: int = 42) -> gym.Env:
    """
    Create environment with appropriate POMDP wrappers.

    Args:
        config: Base configuration.
        pomdp_params: POMDP parameters for wrappers. The ``lag`` key, if present,
            must be a ``[min_lag_days, max_lag_days]`` pair specified in **days**
            (e.g. ``[5, 14]``). It is converted to steps internally via
            ``config.action_interval`` before being passed to TemporalLagWrapper.
            The ``action_delay`` key, if present, specifies the delay in **days**
            between action selection and application (e.g. ``5`` → 1 step). It is
            converted to steps and passed directly to ``EpidemicEnv``.
        seed: RNG seed forwarded to stochastic wrappers (e.g. TemporalLagWrapper).

    Returns:
        Environment with wrappers applied.
    """
    from src.env import EpidemicEnv

    action_delay_days = pomdp_params.get("action_delay")
    action_delay_steps = max(0, round(action_delay_days / config.action_interval)) if action_delay_days else 0
    env = EpidemicEnv(config, action_delay=action_delay_steps)

    if not pomdp_params.get("include_exposed", True):
        env = EpidemicObservationWrapper(env, include_exposed=False)

    detection_rate = pomdp_params.get("detection_rate")
    if detection_rate is not None:
        env = UnderReportingWrapper(
            env,
            detection_rate=detection_rate,
            testing_capacity=pomdp_params.get("testing_capacity"),
            seed=seed,
        )

    noise_stds = pomdp_params.get("noise_stds")
    if noise_stds is not None:
        noise_rho = pomdp_params.get("noise_rho", 0.0)
        env = MultiplicativeNoiseWrapper(env, noise_stds=noise_stds, noise_rho=noise_rho)

    lag_range = pomdp_params.get("lag")
    if lag_range is not None:
        min_lag_days, max_lag_days = lag_range
        action_interval = config.action_interval
        min_lag_steps = max(1, round(min_lag_days / action_interval))
        max_lag_steps = max(min_lag_steps, round(max_lag_days / action_interval))
        env = TemporalLagWrapper(env, min_lag=min_lag_steps, max_lag=max_lag_steps, seed=seed)

    return env
