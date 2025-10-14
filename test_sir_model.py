import pytest
import numpy as np
from sir_model import run_sir_model

@pytest.fixture
def default_params():
    """Default parameters for the SIR model simulation."""
    return {
        'N': 1000,
        'I0': 1,
        'R0': 0,
        'beta': 0.4,
        'gamma': 0.1,
        'days': 160
    }

def test_population_conservation(default_params):
    """Tests if the total population (S + I + R) remains constant."""
    t, S, I, R = run_sir_model(**default_params)
    total_population = S + I + R
    assert np.allclose(total_population, default_params['N'])

def test_output_shapes(default_params):
    """Tests if the output arrays have the correct shape."""
    days = default_params['days']
    t, S, I, R = run_sir_model(**default_params)
    expected_length = days + 1
    assert len(t) == expected_length
    assert len(S) == expected_length
    assert len(I) == expected_length
    assert len(R) == expected_length

def test_no_infection_when_beta_is_zero(default_params):
    """Tests that no new infections occur when beta is 0."""
    params = default_params.copy()
    params['beta'] = 0
    t, S, I, R = run_sir_model(**params)
    # The number of infected should not grow, it can only decrease
    assert np.all(I[1:] <= params['I0'])

def test_initial_conditions(default_params):
    """Tests if the simulation starts with the correct initial conditions."""
    t, S, I, R = run_sir_model(**default_params)
    S0 = default_params['N'] - default_params['I0'] - default_params['R0']
    assert S[0] == S0
    assert I[0] == default_params['I0']
    assert R[0] == default_params['R0']
