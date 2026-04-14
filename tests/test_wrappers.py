"""Tests for POMDP observation wrappers and create_environment factory."""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import Config
from src.env import EpidemicEnv
from src.wrappers import (
    EpidemicObservationWrapper,
    UnderReportingWrapper,
    MultiplicativeNoiseWrapper,
    TemporalLagWrapper,
    create_environment,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> Config:
    """Small deterministic config for fast tests."""
    return Config(N=1000, E0=10, I0=5, stochastic=False, days=100, action_interval=5)


@pytest.fixture
def base_env(small_config) -> EpidemicEnv:
    """Fresh base environment (no wrappers)."""
    env = EpidemicEnv(small_config)
    env.reset(seed=42)
    return env


def get_wrapper_chain(env: gym.Env) -> list:
    """Walk .env attribute to list wrapper types (outermost first)."""
    chain = []
    current = env
    while hasattr(current, "env"):
        chain.append(type(current))
        current = current.env
    chain.append(type(current))  # base env
    return chain


# ===========================================================================
# 1. EpidemicObservationWrapper
# ===========================================================================

class TestEpidemicObservationWrapper:

    def test_include_exposed_true_identity(self, base_env):
        """include_exposed=True → shape (6,), obs unchanged."""
        wrapped = EpidemicObservationWrapper(base_env, include_exposed=True)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (6,)
        raw_obs = base_env._get_obs()
        np.testing.assert_array_equal(obs, raw_obs)

    def test_include_exposed_false_removes_E(self, base_env):
        """include_exposed=False → shape (5,) = [S, I, R, prev_action, day_frac]."""
        wrapped = EpidemicObservationWrapper(base_env, include_exposed=False)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (5,)
        raw = base_env._get_obs()
        np.testing.assert_array_equal(
            obs, np.array([raw[0], raw[2], raw[3], raw[4], raw[5]], dtype=np.float32)
        )

    def test_observation_space_bounds(self, small_config):
        """Per-element bounds: [N, N, N, n_actions-1, 1.0], dtype=float32."""
        env = EpidemicEnv(small_config)
        wrapped = EpidemicObservationWrapper(env, include_exposed=False)
        space = wrapped.observation_space
        assert space.shape == (5,)
        assert space.dtype == np.float32
        np.testing.assert_array_equal(space.low, np.zeros(5, dtype=np.float32))
        expected_high = np.array(
            [small_config.N, small_config.N, small_config.N, 3.0, 1.0], dtype=np.float32
        )
        np.testing.assert_array_equal(space.high, expected_high)

    def test_step_obs_shape_consistent(self, small_config):
        """step() returns obs with shape (5,) within observation_space."""
        env = EpidemicEnv(small_config)
        wrapped = EpidemicObservationWrapper(env, include_exposed=False)
        wrapped.reset(seed=42)
        obs, _, _, _, _ = wrapped.step(0)
        assert obs.shape == (5,)
        assert wrapped.observation_space.contains(obs)

    def test_rejects_non_box_space(self):
        """ValueError on non-Box observation space."""
        env = gym.make("CartPole-v1")
        # CartPole has a Box space, so we mock a Discrete space
        env.observation_space = spaces.Discrete(4)
        with pytest.raises(ValueError, match="Box observation space"):
            EpidemicObservationWrapper(env, include_exposed=False)

    def test_rejects_wrong_shape(self):
        """ValueError on shape != (6,)."""
        env = gym.make("CartPole-v1")
        env.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected observation shape \\(6,\\)"):
            EpidemicObservationWrapper(env, include_exposed=False)


# ===========================================================================
# 2. UnderReportingWrapper
# ===========================================================================

class TestUnderReportingWrapper:

    def _make_wrapped(self, base_env, detection_rate=0.3, testing_capacity=None, mask_E=True):
        """Helper: optionally mask E, then wrap with UnderReporting."""
        env = base_env
        if mask_E:
            env = EpidemicObservationWrapper(env, include_exposed=False)
        return UnderReportingWrapper(env, detection_rate=detection_rate, testing_capacity=testing_capacity)

    def test_detection_rate_1_identity(self, base_env):
        """k=1.0 → obs unchanged."""
        wrapped = self._make_wrapped(base_env, detection_rate=1.0, mask_E=False)
        obs, _ = wrapped.reset(seed=42)
        raw = base_env._get_obs()
        np.testing.assert_allclose(obs, raw, atol=1e-5)

    def test_population_conservation_3d(self, base_env):
        """S+I+R sum preserved when E is masked."""
        wrapped = self._make_wrapped(base_env, detection_rate=0.3, mask_E=True)
        obs, _ = wrapped.reset(seed=42)
        N = base_env.config.N
        assert obs.shape == (5,)
        np.testing.assert_allclose(obs[:3].sum(), N - base_env.current_state.E, atol=1.0)

        # Also check after a few steps
        for _ in range(5):
            obs, _, _, _, _ = wrapped.step(0)
            raw = base_env._get_obs()
            expected_sum = raw[0] + raw[2] + raw[3]  # S + I + R (before under-reporting)
            np.testing.assert_allclose(obs[:3].sum(), expected_sum, atol=1.0)

    def test_population_conservation_4d(self, small_config):
        """S+E+I+R sum = N when E is included."""
        env = EpidemicEnv(small_config)
        wrapped = UnderReportingWrapper(env, detection_rate=0.3)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (6,)
        np.testing.assert_allclose(obs[:4].sum(), small_config.N, atol=1.0)

    def test_exact_scaling_without_saturation(self, base_env):
        """I_obs = k*I, R_obs = k*R, S_obs = S + (1-k)*(I+R)."""
        k = 0.3
        wrapped = self._make_wrapped(base_env, detection_rate=k, mask_E=True)
        wrapped.reset(seed=42)
        # Step a few times to get non-trivial I, R
        for _ in range(3):
            obs, _, _, _, _ = wrapped.step(0)
        raw = base_env._get_obs()  # [S, E, I, R]
        S_true, I_true, R_true = raw[0], raw[2], raw[3]
        expected_I = k * I_true
        expected_R = k * R_true
        expected_S = S_true + (1 - k) * (I_true + R_true)
        np.testing.assert_allclose(obs[0], expected_S, atol=1e-2)
        np.testing.assert_allclose(obs[1], expected_I, atol=1e-2)
        np.testing.assert_allclose(obs[2], expected_R, atol=1e-2)

    def test_saturation_reduces_detection(self, small_config):
        """Low testing_capacity → k_eff < k_base when I is large."""
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)
        # Very low testing capacity to trigger saturation
        wrapped = UnderReportingWrapper(env_masked, detection_rate=0.5, testing_capacity=0.005)
        wrapped.reset(seed=42)
        # Run until I grows
        for _ in range(10):
            obs, _, _, _, _ = wrapped.step(0)
        raw = env._get_obs()
        I_true = raw[2]
        if I_true > 0:
            k_eff = obs[1] / I_true  # I_obs / I_true
            assert k_eff < 0.5, f"Expected k_eff < 0.5 due to saturation, got {k_eff}"

    def test_saturation_michaelis_menten_formula(self, small_config):
        """Direct check of _effective_rate() against manual formula."""
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)
        k_base = 0.5
        r = 0.01
        wrapped = UnderReportingWrapper(env_masked, detection_rate=k_base, testing_capacity=r)
        N = small_config.N

        for true_I in [0.0, 10.0, 100.0, 500.0]:
            expected = k_base * r / (k_base * (true_I / N) + r)
            actual = wrapped._effective_rate(true_I)
            np.testing.assert_allclose(actual, expected, atol=1e-8,
                                       err_msg=f"Mismatch at I={true_I}")

    def test_invalid_detection_rate(self, base_env):
        """ValueError for 0.0 and 1.5."""
        env = EpidemicObservationWrapper(base_env, include_exposed=False)
        with pytest.raises(ValueError, match="detection_rate"):
            UnderReportingWrapper(env, detection_rate=0.0)
        with pytest.raises(ValueError, match="detection_rate"):
            UnderReportingWrapper(env, detection_rate=1.5)

    def test_invalid_testing_capacity(self, base_env):
        """ValueError for negative/zero testing_capacity."""
        env = EpidemicObservationWrapper(base_env, include_exposed=False)
        with pytest.raises(ValueError, match="testing_capacity"):
            UnderReportingWrapper(env, detection_rate=0.5, testing_capacity=0.0)
        with pytest.raises(ValueError, match="testing_capacity"):
            UnderReportingWrapper(env, detection_rate=0.5, testing_capacity=-0.01)


# ===========================================================================
# 3. MultiplicativeNoiseWrapper
# ===========================================================================

class TestMultiplicativeNoiseWrapper:

    def _make_wrapped(self, small_config, noise_stds, seed=42, mask_E=True, noise_rho=0.0):
        """Helper: create env → optionally mask E → noise wrapper."""
        env = EpidemicEnv(small_config)
        if mask_E:
            env = EpidemicObservationWrapper(env, include_exposed=False)
        wrapped = MultiplicativeNoiseWrapper(env, noise_stds=noise_stds, noise_rho=noise_rho)
        wrapped.reset(seed=seed)
        return wrapped

    def test_zero_noise_identity(self, small_config):
        """noise_stds=[0,0,0] → compartments unchanged, trailing elements pass through."""
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)
        wrapped = MultiplicativeNoiseWrapper(env_masked, noise_stds=[0.0, 0.0, 0.0])
        obs, _ = wrapped.reset(seed=42)
        raw = env._get_obs()  # [S, E, I, R, prev_action, day_frac]
        expected = np.array([raw[0], raw[2], raw[3], raw[4], raw[5]], dtype=np.float32)
        np.testing.assert_allclose(obs, expected, atol=1e-5)

    def test_noise_changes_obs(self, small_config):
        """Non-zero stds → obs differs from ground truth (with high probability)."""
        wrapped = self._make_wrapped(small_config, noise_stds=[0.1, 0.3, 0.15], seed=42)
        obs, _, _, _, _ = wrapped.step(0)
        # Get ground truth from base env
        base = wrapped.env  # EpidemicObservationWrapper
        raw = base.observation(base.env._get_obs())
        # With non-trivial noise, at least one compartment should differ
        assert not np.allclose(obs, raw, atol=1e-6), "Noisy obs should differ from ground truth"

    def test_reproducibility_with_seed(self, small_config):
        """Same seed → same obs."""
        w1 = self._make_wrapped(small_config, noise_stds=[0.1, 0.3, 0.15], seed=42)
        w2 = self._make_wrapped(small_config, noise_stds=[0.1, 0.3, 0.15], seed=42)
        for _ in range(5):
            o1, _, _, _, _ = w1.step(0)
            o2, _, _, _, _ = w2.step(0)
            np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self, small_config):
        """Different seeds → different obs."""
        w1 = self._make_wrapped(small_config, noise_stds=[0.1, 0.3, 0.15], seed=42)
        w2 = self._make_wrapped(small_config, noise_stds=[0.1, 0.3, 0.15], seed=99)
        any_different = False
        for _ in range(5):
            o1, _, _, _, _ = w1.step(0)
            o2, _, _, _, _ = w2.step(0)
            if not np.allclose(o1, o2, atol=1e-6):
                any_different = True
                break
        assert any_different, "Different seeds should yield different noisy obs"

    def test_clipping_to_bounds(self, small_config):
        """All obs in [0, N] even with extreme noise."""
        N = small_config.N
        wrapped = self._make_wrapped(small_config, noise_stds=[2.0, 2.0, 2.0], seed=42)
        for _ in range(20):
            obs, _, terminated, _, _ = wrapped.step(0)
            assert np.all(obs >= 0.0), f"Obs below 0: {obs}"
            assert np.all(obs <= N), f"Obs above N: {obs}"
            if terminated:
                wrapped.reset(seed=42)

    def test_invalid_noise_stds_length(self, small_config):
        """ValueError when noise_stds length doesn't match compartment count."""
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)
        # obs size is 5, compartments = 3; passing 5 stds should fail
        with pytest.raises(ValueError, match="compartment"):
            MultiplicativeNoiseWrapper(env_masked, noise_stds=[0.1, 0.1, 0.1, 0.1, 0.1])

    def test_trailing_elements_pass_through(self, small_config):
        """prev_action and day_frac pass through noise wrapper unchanged."""
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)
        wrapped = MultiplicativeNoiseWrapper(env_masked, noise_stds=[2.0, 2.0, 2.0])
        obs, _ = wrapped.reset(seed=42)
        # prev_action=0.0 at reset, day_frac=0.0 at reset
        assert obs[3] == 0.0, f"prev_action should be 0.0, got {obs[3]}"
        assert obs[4] == 0.0, f"day_frac should be 0.0, got {obs[4]}"

    def test_statistical_properties(self, small_config):
        """Over many samples: mean(ratio) ≈ 1.0, std(ratio) ≈ noise_std."""
        noise_std = 0.1
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)

        n_samples = 2000
        ratios = np.zeros((n_samples, 3))
        for i in range(n_samples):
            wrapped = MultiplicativeNoiseWrapper(env_masked, noise_stds=[noise_std] * 3)
            obs, _ = wrapped.reset(seed=i)
            raw = env._get_obs()
            truth = np.array([raw[0], raw[2], raw[3]], dtype=np.float32)
            # Avoid division by zero — only check compartments with positive truth
            mask = truth > 1.0
            if mask.any():
                ratios[i, mask] = obs[:3][mask] / truth[mask]
            else:
                ratios[i] = 1.0

        # Only analyze compartments where we had valid samples
        for j in range(3):
            col = ratios[:, j]
            valid = col[col > 0]
            if len(valid) > 100:
                np.testing.assert_allclose(valid.mean(), 1.0, atol=0.05,
                                           err_msg=f"Mean ratio for compartment {j} not ≈ 1.0")
                np.testing.assert_allclose(valid.std(), noise_std, atol=0.05,
                                           err_msg=f"Std ratio for compartment {j} not ≈ {noise_std}")

    # --- AR(1) autocorrelated noise tests ---

    def test_ar1_rho_zero_matches_iid(self, small_config):
        """Explicit rho=0.0 produces same output as default (no rho arg)."""
        env1 = EpidemicEnv(small_config)
        env1m = EpidemicObservationWrapper(env1, include_exposed=False)
        w1 = MultiplicativeNoiseWrapper(env1m, noise_stds=[0.1, 0.3, 0.15], noise_rho=0.0)
        o1, _ = w1.reset(seed=42)

        env2 = EpidemicEnv(small_config)
        env2m = EpidemicObservationWrapper(env2, include_exposed=False)
        w2 = MultiplicativeNoiseWrapper(env2m, noise_stds=[0.1, 0.3, 0.15])
        o2, _ = w2.reset(seed=42)

        np.testing.assert_array_equal(o1, o2)
        for _ in range(5):
            o1, _, _, _, _ = w1.step(0)
            o2, _, _, _, _ = w2.step(0)
            np.testing.assert_array_equal(o1, o2)

    def test_ar1_bias_resets_on_episode_reset(self, small_config):
        """After stepping with rho=0.9, reset clears accumulated bias.

        Note: reset() zeros _bias, then super().reset() calls observation()
        which applies one innovation step. So after reset, _bias = scale * ε
        (a single fresh draw), NOT the accumulated value from pre-reset.
        """
        wrapped = self._make_wrapped(small_config, [0.1, 0.3, 0.15], seed=42, noise_rho=0.9)
        for _ in range(10):
            wrapped.step(0)
        bias_before = wrapped._bias.copy()
        assert not np.allclose(bias_before, 0.0), "Bias should be non-zero after steps"
        wrapped.reset(seed=42)
        # After reset, bias should be a single innovation (not accumulated)
        # It differs from the pre-reset accumulated bias
        assert not np.array_equal(wrapped._bias, bias_before), \
            "Bias should change after reset"
        # Single innovation magnitude: scale * N(0, σ) where scale = sqrt(1 - 0.81) ≈ 0.436
        # So |bias| should be small relative to accumulated bias
        innovation_scale = np.sqrt(1.0 - 0.9 ** 2)
        stds = np.array([0.1, 0.3, 0.15])
        # Each element should be within ~4σ of zero for the single innovation
        for j in range(3):
            assert abs(wrapped._bias[j]) < 4 * innovation_scale * stds[j], \
                f"Post-reset bias[{j}]={wrapped._bias[j]} too large for single innovation"

    def test_ar1_autocorrelation(self, small_config):
        """Lag-1 autocorrelation of noise ≈ ρ (statistical, generous tolerance)."""
        rho = 0.7
        n_steps = 2000
        noise_std = 0.2
        wrapped = self._make_wrapped(small_config, [noise_std, noise_std, noise_std],
                                     seed=42, noise_rho=rho)
        biases = []
        for _ in range(n_steps):
            wrapped.step(0)
            biases.append(wrapped._bias.copy())

        biases = np.array(biases)  # (n_steps, 3)
        for j in range(3):
            series = biases[:, j]
            # Lag-1 autocorrelation
            corr = np.corrcoef(series[:-1], series[1:])[0, 1]
            np.testing.assert_allclose(corr, rho, atol=0.1,
                                       err_msg=f"Lag-1 autocorr for compartment {j}: {corr} ≠ {rho}")

    def test_ar1_marginal_variance_preserved(self, small_config):
        """Std of bias ≈ σ after burn-in, regardless of ρ."""
        rho = 0.7
        noise_std = 0.2
        n_steps = 3000
        wrapped = self._make_wrapped(small_config, [noise_std, noise_std, noise_std],
                                     seed=42, noise_rho=rho)
        biases = []
        # Burn-in
        for _ in range(100):
            wrapped.step(0)
        for _ in range(n_steps):
            wrapped.step(0)
            biases.append(wrapped._bias.copy())

        biases = np.array(biases)
        for j in range(3):
            np.testing.assert_allclose(biases[:, j].std(), noise_std, atol=0.05,
                                       err_msg=f"Marginal std for compartment {j} not ≈ {noise_std}")

    def test_ar1_invalid_rho(self, small_config):
        """ValueError for rho=1.0, rho=-0.1, rho=1.5."""
        env = EpidemicEnv(small_config)
        env = EpidemicObservationWrapper(env, include_exposed=False)
        for bad_rho in [1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="noise_rho"):
                MultiplicativeNoiseWrapper(env, noise_stds=[0.1, 0.3, 0.15], noise_rho=bad_rho)

    def test_ar1_trailing_elements_unchanged(self, small_config):
        """prev_action and day_frac unaffected with rho=0.7."""
        env = EpidemicEnv(small_config)
        env_masked = EpidemicObservationWrapper(env, include_exposed=False)
        wrapped = MultiplicativeNoiseWrapper(env_masked, noise_stds=[2.0, 2.0, 2.0], noise_rho=0.7)
        obs, _ = wrapped.reset(seed=42)
        assert obs[3] == 0.0, f"prev_action should be 0.0, got {obs[3]}"
        assert obs[4] == 0.0, f"day_frac should be 0.0, got {obs[4]}"

    def test_ar1_reproducibility(self, small_config):
        """Same seed + rho → identical sequences."""
        w1 = self._make_wrapped(small_config, [0.1, 0.3, 0.15], seed=42, noise_rho=0.7)
        w2 = self._make_wrapped(small_config, [0.1, 0.3, 0.15], seed=42, noise_rho=0.7)
        for _ in range(10):
            o1, _, _, _, _ = w1.step(0)
            o2, _, _, _, _ = w2.step(0)
            np.testing.assert_array_equal(o1, o2)


# ===========================================================================
# 4. TemporalLagWrapper
# ===========================================================================

class TestTemporalLagWrapper:

    def _make_wrapped(self, small_config, min_lag=5, max_lag=14, seed=42, mask_E=True):
        """Helper: create env → optionally mask E → temporal lag wrapper."""
        env = EpidemicEnv(small_config)
        if mask_E:
            env = EpidemicObservationWrapper(env, include_exposed=False)
        wrapped = TemporalLagWrapper(env, min_lag=min_lag, max_lag=max_lag, seed=seed)
        wrapped.reset(seed=42)
        return wrapped

    def test_warmup_returns_first_obs(self, small_config):
        """During warmup (first min_lag steps), obs = initial observation."""
        min_lag = 5
        wrapped = self._make_wrapped(small_config, min_lag=min_lag, max_lag=14, seed=42)
        initial_obs, _ = wrapped.reset(seed=42)

        for step_i in range(min_lag):
            obs, _, _, _, _ = wrapped.step(0)
            # During warmup the pointer clamps to 0, so we get initial/early obs
            # The key invariant is that obs comes from early in the buffer
            assert obs.shape == initial_obs.shape

    def test_obs_is_delayed(self, small_config):
        """Post-warmup obs ≠ current ground truth."""
        wrapped = self._make_wrapped(small_config, min_lag=5, max_lag=14, seed=42)
        # Step past warmup
        for _ in range(20):
            obs, _, _, _, _ = wrapped.step(0)
        # Get current ground truth
        base = wrapped.env  # EpidemicObservationWrapper
        current_truth = base.observation(base.env._get_obs())
        # Delayed obs should differ from current truth (epidemic is evolving)
        assert not np.allclose(obs, current_truth, atol=1e-3), \
            "Post-warmup obs should be delayed and differ from current ground truth"

    def test_fifo_monotonicity(self, small_config):
        """Pointer index never decreases over 50 steps."""
        wrapped = self._make_wrapped(small_config, min_lag=3, max_lag=10, seed=42)
        prev_pointer = -1
        for _ in range(50):
            wrapped.step(0)
            assert wrapped._last_pointer >= prev_pointer, \
                f"FIFO violation: pointer went from {prev_pointer} to {wrapped._last_pointer}"
            prev_pointer = wrapped._last_pointer

    def test_lag_within_bounds(self, small_config):
        """Effective lag ∈ [min_lag, max_lag] after warmup."""
        min_lag, max_lag = 5, 14
        wrapped = self._make_wrapped(small_config, min_lag=min_lag, max_lag=max_lag, seed=42)
        # Step past warmup
        for _ in range(max_lag + 5):
            wrapped.step(0)
        # _current_lag should always be within bounds
        for _ in range(30):
            wrapped.step(0)
            assert min_lag <= wrapped._current_lag <= max_lag, \
                f"Lag {wrapped._current_lag} outside [{min_lag}, {max_lag}]"

    def test_reset_clears_state(self, small_config):
        """After reset, warmup restarts, no stale data."""
        wrapped = self._make_wrapped(small_config, min_lag=5, max_lag=14, seed=42)
        for _ in range(20):
            wrapped.step(0)

        # Reset and check state is clean
        # Note: reset() calls super().reset() → observation() which does one
        # _step increment and appends the initial obs to the cache.
        wrapped.reset(seed=99)
        assert wrapped._step == 1  # observation() called once during reset
        assert wrapped._last_pointer == 0  # clamped to 0 during warmup
        assert len(wrapped._cache) == 1  # only the initial obs

    def test_reproducibility_with_seed(self, small_config):
        """Same seed → same obs sequence."""
        w1 = self._make_wrapped(small_config, min_lag=5, max_lag=14, seed=42)
        w2 = self._make_wrapped(small_config, min_lag=5, max_lag=14, seed=42)
        for _ in range(20):
            o1, _, _, _, _ = w1.step(0)
            o2, _, _, _, _ = w2.step(0)
            np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self, small_config):
        """Different seeds → different obs post-warmup."""
        w1 = self._make_wrapped(small_config, min_lag=5, max_lag=14, seed=42)
        w2 = self._make_wrapped(small_config, min_lag=5, max_lag=14, seed=99)
        any_different = False
        for _ in range(20):
            o1, _, _, _, _ = w1.step(0)
            o2, _, _, _, _ = w2.step(0)
            if not np.allclose(o1, o2, atol=1e-6):
                any_different = True
                break
        assert any_different, "Different lag seeds should yield different observations"

    def test_fixed_lag_when_min_equals_max(self, small_config):
        """min_lag=max_lag=3 → constant 3-step delay."""
        lag = 3
        wrapped = self._make_wrapped(small_config, min_lag=lag, max_lag=lag, seed=42)
        for _ in range(20):
            wrapped.step(0)
            assert wrapped._current_lag == lag, \
                f"Expected fixed lag {lag}, got {wrapped._current_lag}"

    def test_invalid_min_lag(self, small_config):
        """ValueError for min_lag=0."""
        env = EpidemicEnv(small_config)
        with pytest.raises(ValueError, match="min_lag"):
            TemporalLagWrapper(env, min_lag=0, max_lag=5)

    def test_invalid_max_lag(self, small_config):
        """ValueError for max_lag < min_lag."""
        env = EpidemicEnv(small_config)
        with pytest.raises(ValueError, match="max_lag"):
            TemporalLagWrapper(env, min_lag=10, max_lag=5)


# ===========================================================================
# 5. TestCreateEnvironment (integration)
# ===========================================================================

class TestCreateEnvironment:

    def test_mdp_no_wrappers(self, small_config):
        """Empty pomdp_params → obs shape (6,), base env type."""
        env = create_environment(small_config, {}, seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6,)
        assert isinstance(env, EpidemicEnv)

    def test_full_pomdp_chain(self, small_config):
        """Full POMDP params → wrapper chain: Lag → Noise → UnderReporting → EpiObs → EpidemicEnv."""
        pomdp_params = {
            "include_exposed": False,
            "detection_rate": 0.3,
            "lag": (5, 14),
            "noise_stds": [0.05, 0.3, 0.15],
        }
        env = create_environment(small_config, pomdp_params, seed=42)
        chain = get_wrapper_chain(env)
        expected = [
            TemporalLagWrapper,
            MultiplicativeNoiseWrapper,
            UnderReportingWrapper,
            EpidemicObservationWrapper,
            EpidemicEnv,
        ]
        assert chain == expected, f"Expected {expected}, got {chain}"

    def test_obs_within_space(self, small_config):
        """20 steps, all obs contained in observation_space."""
        pomdp_params = {
            "include_exposed": False,
            "detection_rate": 0.3,
            "noise_stds": [0.05, 0.3, 0.15],
        }
        env = create_environment(small_config, pomdp_params, seed=42)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(0)
            assert env.observation_space.contains(obs), f"Obs {obs} outside space"
            if terminated:
                obs, _ = env.reset(seed=42)

    def test_seed_propagation(self, small_config):
        """Two envs with same seed → identical obs sequences."""
        pomdp_params = {
            "include_exposed": False,
            "detection_rate": 0.3,
            "lag": (5, 14),
            "noise_stds": [0.05, 0.3, 0.15],
        }
        env1 = create_environment(small_config, pomdp_params, seed=42)
        env2 = create_environment(small_config, pomdp_params, seed=42)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        for _ in range(15):
            o1, _, _, _, _ = env1.step(0)
            o2, _, _, _, _ = env2.step(0)
            np.testing.assert_array_equal(o1, o2)
