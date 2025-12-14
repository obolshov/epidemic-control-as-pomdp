"""Tests for SIR model core functionality."""

import numpy as np
import pytest

from src.sir import EpidemicState, run_sir


def run_sir_with_initial(state: EpidemicState, beta: float, gamma: float, days: int):
    """
    Helper function for tests that need initial state included.
    Concatenates initial state with run_sir results.
    """
    S_new, I_new, R_new = run_sir(state, beta, gamma, days)
    S = np.concatenate([[state.S], S_new])
    I = np.concatenate([[state.I], I_new])
    R = np.concatenate([[state.R], R_new])
    return S, I, R


@pytest.fixture
def default_params():
    """Default parameters for the SIR model simulation."""
    return {
        "state": EpidemicState(N=1000, S=999, I=1, R=0),
        "beta": 0.4,
        "gamma": 0.1,
        "days": 160,
    }


class TestRunSir:
    """Tests for run_sir function."""

    def test_output_shape(self, default_params):
        """Test that run_sir returns arrays of correct length."""
        days = 10
        default_params["days"] = days
        S, I, R = run_sir(**default_params)
        expected_length = days  # excludes initial state
        assert len(S) == expected_length
        assert len(I) == expected_length
        assert len(R) == expected_length

    def test_excludes_initial_state(self, default_params):
        """Test that run_sir does not include initial state in output."""
        S, I, R = run_sir(**default_params)
        S_with_initial, I_with_initial, R_with_initial = run_sir_with_initial(
            **default_params
        )

        # run_sir should return one less element than run_sir_with_initial
        assert len(S) == len(S_with_initial) - 1
        assert len(I) == len(I_with_initial) - 1
        assert len(R) == len(R_with_initial) - 1

        # run_sir results should match run_sir_with_initial[1:]
        assert np.allclose(S, S_with_initial[1:])
        assert np.allclose(I, I_with_initial[1:])
        assert np.allclose(R, R_with_initial[1:])

    def test_zero_days(self, default_params):
        """Test that run_sir with days=0 returns empty arrays."""
        params = default_params.copy()
        params["days"] = 0
        S, I, R = run_sir(**params)
        assert len(S) == 0
        assert len(I) == 0
        assert len(R) == 0


class TestSirModelProperties:
    """Tests for SIR model mathematical properties."""

    def test_population_conservation(self, default_params):
        """Tests if the total population (S + I + R) remains constant."""
        S, I, R = run_sir_with_initial(**default_params)
        total_population = S + I + R
        assert np.allclose(total_population, default_params["state"].N)

    def test_output_shapes(self, default_params):
        """Tests if the output arrays have the correct shape."""
        days = default_params["days"]
        S, I, R = run_sir_with_initial(**default_params)
        expected_length = days + 1
        assert len(S) == expected_length
        assert len(I) == expected_length
        assert len(R) == expected_length

    def test_no_infection_when_beta_is_zero(self, default_params):
        """Tests that no new infections occur when beta is 0."""
        params = default_params.copy()
        params["beta"] = 0
        S, I, R = run_sir_with_initial(**params)
        # The number of infected should not grow, it can only decrease
        assert np.all(I[1:] <= default_params["state"].I)

    def test_initial_conditions(self, default_params):
        """Tests if the simulation starts with the correct initial conditions."""
        S, I, R = run_sir_with_initial(**default_params)
        default_state = default_params["state"]
        S0 = default_state.N - default_state.I - default_state.R
        assert S[0] == S0
        assert I[0] == default_state.I
        assert R[0] == default_state.R

    def test_susceptible_never_increases(self, default_params):
        """Tests that susceptible population never increases over time."""
        S, I, R = run_sir_with_initial(**default_params)
        # S should be monotonically decreasing
        assert np.all(np.diff(S) <= 0), "Susceptible population should never increase"

    def test_recovered_never_decreases(self, default_params):
        """Tests that recovered population never decreases over time."""
        S, I, R = run_sir_with_initial(**default_params)
        # R should be monotonically increasing
        assert np.all(np.diff(R) >= 0), "Recovered population should never decrease"

    def test_all_values_non_negative(self, default_params):
        """Tests that S, I, R are always non-negative."""
        S, I, R = run_sir_with_initial(**default_params)
        assert np.all(S >= 0), "Susceptible should be non-negative"
        assert np.all(I >= 0), "Infected should be non-negative"
        assert np.all(R >= 0), "Recovered should be non-negative"

    def test_higher_beta_increases_peak(self, default_params):
        """Tests that higher transmission rate leads to higher peak infections."""
        params_low = default_params.copy()
        params_low["beta"] = 0.2
        S_low, I_low, R_low = run_sir_with_initial(**params_low)
        peak_low = np.max(I_low)

        params_high = default_params.copy()
        params_high["beta"] = 0.6
        S_high, I_high, R_high = run_sir_with_initial(**params_high)
        peak_high = np.max(I_high)

        assert (
            peak_high > peak_low
        ), f"Higher beta should lead to higher peak (low: {peak_low:.2f}, high: {peak_high:.2f})"

    def test_higher_gamma_decreases_peak(self, default_params):
        """Tests that higher recovery rate leads to lower peak infections."""
        params_low = default_params.copy()
        params_low["gamma"] = 0.05
        S_low, I_low, R_low = run_sir_with_initial(**params_low)
        peak_low = np.max(I_low)

        params_high = default_params.copy()
        params_high["gamma"] = 0.2
        S_high, I_high, R_high = run_sir_with_initial(**params_high)
        peak_high = np.max(I_high)

        assert (
            peak_high < peak_low
        ), f"Higher gamma should lead to lower peak (low gamma: {peak_low:.2f}, high gamma: {peak_high:.2f})"
