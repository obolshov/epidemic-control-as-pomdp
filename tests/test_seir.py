"""Tests for SEIR model core functionality."""

import pytest
import numpy as np
from src.seir import run_seir, EpidemicState


def run_seir_with_initial(
    state: EpidemicState, beta: float, sigma: float, gamma: float, days: int
):
    """
    Helper function for tests that need initial state included.
    Concatenates initial state with run_seir results.
    """
    S_new, E_new, I_new, R_new = run_seir(state, beta, sigma, gamma, days)
    S = np.concatenate([[state.S], S_new])
    E = np.concatenate([[state.E], E_new])
    I = np.concatenate([[state.I], I_new])
    R = np.concatenate([[state.R], R_new])
    return S, E, I, R


@pytest.fixture
def default_params():
    """Default parameters for the SEIR model simulation."""
    return {
        "state": EpidemicState(N=1000, S=999, E=0, I=1, R=0),
        "beta": 0.4,
        "sigma": 0.2,
        "gamma": 0.1,
        "days": 160,
    }


class TestRunSeir:
    """Tests for run_seir function."""

    def test_output_shape(self, default_params):
        """Test that run_seir returns arrays of correct length."""
        days = 10
        default_params["days"] = days
        S, E, I, R = run_seir(**default_params)
        expected_length = days  # excludes initial state
        assert len(S) == expected_length
        assert len(E) == expected_length
        assert len(I) == expected_length
        assert len(R) == expected_length

    def test_excludes_initial_state(self, default_params):
        """Test that run_seir does not include initial state in output."""
        S, E, I, R = run_seir(**default_params)
        (
            S_with_initial,
            E_with_initial,
            I_with_initial,
            R_with_initial,
        ) = run_seir_with_initial(**default_params)

        # run_seir should return one less element than run_seir_with_initial
        assert len(S) == len(S_with_initial) - 1
        assert len(E) == len(E_with_initial) - 1
        assert len(I) == len(I_with_initial) - 1
        assert len(R) == len(R_with_initial) - 1

        # run_seir results should match run_seir_with_initial[1:]
        assert np.allclose(S, S_with_initial[1:])
        assert np.allclose(E, E_with_initial[1:])
        assert np.allclose(I, I_with_initial[1:])
        assert np.allclose(R, R_with_initial[1:])

    def test_zero_days(self, default_params):
        """Test that run_seir with days=0 returns empty arrays."""
        params = default_params.copy()
        params["days"] = 0
        S, E, I, R = run_seir(**params)
        assert len(S) == 0
        assert len(E) == 0
        assert len(I) == 0
        assert len(R) == 0


class TestSeirModelProperties:
    """Tests for SEIR model mathematical properties."""

    def test_population_conservation(self, default_params):
        """Tests if the total population (S + E + I + R) remains constant."""
        S, E, I, R = run_seir_with_initial(**default_params)
        total_population = S + E + I + R
        assert np.allclose(total_population, default_params["state"].N)

    def test_output_shapes(self, default_params):
        """Tests if the output arrays have the correct shape."""
        days = default_params["days"]
        S, E, I, R = run_seir_with_initial(**default_params)
        expected_length = days + 1
        assert len(S) == expected_length
        assert len(E) == expected_length
        assert len(I) == expected_length
        assert len(R) == expected_length

    def test_no_infection_when_beta_is_zero(self, default_params):
        """Tests that no new infections (E) occur when beta is 0."""
        params = default_params.copy()
        params["beta"] = 0
        # If I=1 initially, those will progress to R eventually, but no new E will form.
        S, E, I, R = run_seir_with_initial(**params)

        # E should only decrease (as E -> I)
        assert np.all(np.diff(E) <= 0), "E should not increase if beta=0"
        # S should be constant
        assert np.allclose(S, S[0]), "S should remain constant if beta=0"

    def test_initial_conditions(self, default_params):
        """Tests if the simulation starts with the correct initial conditions."""
        S, E, I, R = run_seir_with_initial(**default_params)
        default_state = default_params["state"]
        S0 = default_state.N - default_state.E - default_state.I - default_state.R
        assert S[0] == S0
        assert E[0] == default_state.E
        assert I[0] == default_state.I
        assert R[0] == default_state.R

    def test_susceptible_never_increases(self, default_params):
        """Tests that susceptible population never increases over time."""
        S, E, I, R = run_seir_with_initial(**default_params)
        # S should be monotonically decreasing
        assert np.all(np.diff(S) <= 0), "Susceptible population should never increase"

    def test_recovered_never_decreases(self, default_params):
        """Tests that recovered population never decreases over time."""
        S, E, I, R = run_seir_with_initial(**default_params)
        # R should be monotonically increasing
        assert np.all(np.diff(R) >= 0), "Recovered population should never decrease"

    def test_all_values_non_negative(self, default_params):
        """Tests that S, E, I, R are always non-negative."""
        S, E, I, R = run_seir_with_initial(**default_params)
        assert np.all(S >= 0), "Susceptible should be non-negative"
        assert np.all(E >= 0), "Exposed should be non-negative"
        assert np.all(I >= 0), "Infected should be non-negative"
        assert np.all(R >= 0), "Recovered should be non-negative"

    def test_higher_beta_increases_peak(self, default_params):
        """Tests that higher transmission rate leads to higher peak infections."""
        params_low = default_params.copy()
        params_low["beta"] = 0.2
        S_low, E_low, I_low, R_low = run_seir_with_initial(**params_low)
        peak_low = np.max(I_low)

        params_high = default_params.copy()
        params_high["beta"] = 0.6
        S_high, E_high, I_high, R_high = run_seir_with_initial(**params_high)
        peak_high = np.max(I_high)

        assert (
            peak_high > peak_low
        ), f"Higher beta should lead to higher peak (low: {peak_low:.2f}, high: {peak_high:.2f})"

    def test_higher_gamma_decreases_peak(self, default_params):
        """Tests that higher recovery rate leads to lower peak infections."""
        params_low = default_params.copy()
        params_low["gamma"] = 0.05
        S_low, E_low, I_low, R_low = run_seir_with_initial(**params_low)
        peak_low = np.max(I_low)

        params_high = default_params.copy()
        params_high["gamma"] = 0.2
        S_high, E_high, I_high, R_high = run_seir_with_initial(**params_high)
        peak_high = np.max(I_high)

        assert (
            peak_high < peak_low
        ), f"Higher gamma should lead to lower peak (low gamma: {peak_low:.2f}, high gamma: {peak_high:.2f})"


class TestStochasticRunSeir:
    """Tests for stochastic Binomial transition mode of run_seir."""

    @pytest.fixture
    def stochastic_params(self):
        """Parameters for stochastic SEIR tests. N=1000 gives noticeable variance."""
        return {
            "state": EpidemicState(N=1000, S=999, E=0, I=1, R=0),
            "beta": 0.4,
            "sigma": 0.2,
            "gamma": 0.1,
            "days": 160,
            "rng": np.random.default_rng(42),
        }

    def test_non_negative(self, stochastic_params):
        """Binomial draws are bounded by compartment size, so values must stay >= 0."""
        S, E, I, R = run_seir(**stochastic_params)
        assert np.all(S >= 0), "S must be non-negative"
        assert np.all(E >= 0), "E must be non-negative"
        assert np.all(I >= 0), "I must be non-negative"
        assert np.all(R >= 0), "R must be non-negative"

    def test_population_conservation(self, stochastic_params):
        """S+E+I+R must equal N exactly at every step (integer arithmetic)."""
        state = stochastic_params["state"]
        S, E, I, R = run_seir(**stochastic_params)
        totals = S + E + I + R
        assert np.all(totals == state.N), f"Population not conserved: {totals}"

    def test_reproducible_with_seed(self, stochastic_params):
        """Same rng seed must produce bitwise-identical trajectories."""
        params_a = stochastic_params.copy()
        params_a["rng"] = np.random.default_rng(99)
        S_a, E_a, I_a, R_a = run_seir(**params_a)

        params_b = stochastic_params.copy()
        params_b["rng"] = np.random.default_rng(99)
        S_b, E_b, I_b, R_b = run_seir(**params_b)

        assert np.array_equal(S_a, S_b)
        assert np.array_equal(I_a, I_b)

    def test_different_seeds_differ(self, stochastic_params):
        """Different seeds should produce distinct trajectories (N=1000 → noticeable variance)."""
        params_a = stochastic_params.copy()
        params_a["rng"] = np.random.default_rng(1)
        _, _, I_a, _ = run_seir(**params_a)

        params_b = stochastic_params.copy()
        params_b["rng"] = np.random.default_rng(2)
        _, _, I_b, _ = run_seir(**params_b)

        assert not np.array_equal(I_a, I_b), "Different seeds should yield different trajectories"

    def test_large_n_converges_to_deterministic(self):
        """Mean of many stochastic runs with large N should be close to deterministic solution."""
        state = EpidemicState(N=10_000, S=9_900, E=50, I=50, R=0)
        beta, sigma, gamma, days = 0.4, 0.2, 0.1, 100
        n_runs = 200
        rng = np.random.default_rng(0)

        I_stochastic = np.zeros((n_runs, days))
        for k in range(n_runs):
            _, _, I_run, _ = run_seir(state, beta, sigma, gamma, days, rng=rng)
            I_stochastic[k] = I_run

        _, _, I_det, _ = run_seir(state, beta, sigma, gamma, days, rng=None)

        mean_I = I_stochastic.mean(axis=0)
        peak_det = I_det.max()
        peak_mean = mean_I.max()

        relative_error = abs(peak_mean - peak_det) / peak_det
        assert relative_error < 0.05, (
            f"Stochastic mean peak ({peak_mean:.1f}) deviates >5% from "
            f"deterministic peak ({peak_det:.1f})"
        )

    def test_no_negative_with_extreme_rates(self):
        """Binomial transitions must stay non-negative even with extreme epidemic parameters."""
        state = EpidemicState(N=1000, S=990, E=5, I=5, R=0)
        rng = np.random.default_rng(7)
        S, E, I, R = run_seir(state, beta=1.5, sigma=0.9, gamma=0.9, days=50, rng=rng)
        assert np.all(S >= 0)
        assert np.all(E >= 0)
        assert np.all(I >= 0)
        assert np.all(R >= 0)

