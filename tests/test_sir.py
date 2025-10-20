"""Tests for SIR model core functionality."""

import pytest
import numpy as np
from src.sir import run_sir_model


@pytest.fixture
def default_params():
    """Default parameters for the SIR model simulation."""
    return {"N": 1000, "I0": 1, "R0": 0, "beta": 0.4, "gamma": 0.1, "days": 160}


def test_population_conservation(default_params):
    """Tests if the total population (S + I + R) remains constant."""
    t, S, I, R = run_sir_model(**default_params)
    total_population = S + I + R
    assert np.allclose(total_population, default_params["N"])


def test_output_shapes(default_params):
    """Tests if the output arrays have the correct shape."""
    days = default_params["days"]
    t, S, I, R = run_sir_model(**default_params)
    expected_length = days + 1
    assert len(t) == expected_length
    assert len(S) == expected_length
    assert len(I) == expected_length
    assert len(R) == expected_length


def test_no_infection_when_beta_is_zero(default_params):
    """Tests that no new infections occur when beta is 0."""
    params = default_params.copy()
    params["beta"] = 0
    t, S, I, R = run_sir_model(**params)
    # The number of infected should not grow, it can only decrease
    assert np.all(I[1:] <= params["I0"])


def test_initial_conditions(default_params):
    """Tests if the simulation starts with the correct initial conditions."""
    t, S, I, R = run_sir_model(**default_params)
    S0 = default_params["N"] - default_params["I0"] - default_params["R0"]
    assert S[0] == S0
    assert I[0] == default_params["I0"]
    assert R[0] == default_params["R0"]


def test_susceptible_never_increases(default_params):
    """Tests that susceptible population never increases over time."""
    t, S, I, R = run_sir_model(**default_params)
    # S should be monotonically decreasing
    assert np.all(np.diff(S) <= 0), "Susceptible population should never increase"


def test_recovered_never_decreases(default_params):
    """Tests that recovered population never decreases over time."""
    t, S, I, R = run_sir_model(**default_params)
    # R should be monotonically increasing
    assert np.all(np.diff(R) >= 0), "Recovered population should never decrease"


def test_all_values_non_negative(default_params):
    """Tests that S, I, R are always non-negative."""
    t, S, I, R = run_sir_model(**default_params)
    assert np.all(S >= 0), "Susceptible should be non-negative"
    assert np.all(I >= 0), "Infected should be non-negative"
    assert np.all(R >= 0), "Recovered should be non-negative"


def test_higher_beta_increases_peak(default_params):
    """Tests that higher transmission rate leads to higher peak infections."""
    params_low = default_params.copy()
    params_low["beta"] = 0.2
    t_low, S_low, I_low, R_low = run_sir_model(**params_low)
    peak_low = np.max(I_low)

    params_high = default_params.copy()
    params_high["beta"] = 0.6
    t_high, S_high, I_high, R_high = run_sir_model(**params_high)
    peak_high = np.max(I_high)

    assert (
        peak_high > peak_low
    ), f"Higher beta should lead to higher peak (low: {peak_low:.2f}, high: {peak_high:.2f})"


def test_higher_gamma_decreases_peak(default_params):
    """Tests that higher recovery rate leads to lower peak infections."""
    params_low = default_params.copy()
    params_low["gamma"] = 0.05
    t_low, S_low, I_low, R_low = run_sir_model(**params_low)
    peak_low = np.max(I_low)

    params_high = default_params.copy()
    params_high["gamma"] = 0.2
    t_high, S_high, I_high, R_high = run_sir_model(**params_high)
    peak_high = np.max(I_high)

    assert (
        peak_high < peak_low
    ), f"Higher gamma should lead to lower peak (low gamma: {peak_low:.2f}, high gamma: {peak_high:.2f})"
