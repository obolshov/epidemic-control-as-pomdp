"""Tests for intervention actions."""

import pytest
import numpy as np
from src.actions import InterventionAction
from src.sir import run_sir_model
from src.sir import EpidemicState


def test_intervention_values_are_valid():
    """Tests that all intervention coefficients are between 0 and 1."""
    for action in InterventionAction:
        assert (
            0 < action.value <= 1.0
        ), f"Action {action.name} has invalid coefficient: {action.value}"


def test_apply_to_beta_returns_valid_beta():
    """Tests that apply_to_beta returns positive values."""
    beta_0 = 0.4

    for action in InterventionAction:
        beta_modified = action.apply_to_beta(beta_0)
        assert (
            beta_modified > 0
        ), f"Modified beta for {action.name} should be positive, got {beta_modified}"
        assert (
            beta_modified <= beta_0
        ), f"Modified beta for {action.name} should not exceed base beta"


def test_no_intervention_preserves_beta():
    """Tests that NO intervention doesn't change beta."""
    beta_0 = 0.4
    assert InterventionAction.NO.apply_to_beta(beta_0) == beta_0


def test_beta_strictly_decreases_with_interventions():
    """Tests that beta values strictly decrease with stronger interventions."""
    beta_0 = 0.4
    actions = [
        InterventionAction.NO,
        InterventionAction.MILD,
        InterventionAction.MODERATE,
        InterventionAction.SEVERE,
    ]

    beta_values = [action.apply_to_beta(beta_0) for action in actions]

    for i in range(len(beta_values) - 1):
        assert beta_values[i] > beta_values[i + 1], (
            f"Beta for {actions[i].name} ({beta_values[i]}) should be > "
            f"beta for {actions[i+1].name} ({beta_values[i+1]})"
        )


@pytest.fixture
def default_sir_params():
    """Default parameters for SIR model in intervention tests."""
    return {
        "state": EpidemicState(N=1000, S=999, I=1, R=0),
        "beta": 0.4,
        "gamma": 0.1,
        "days": 160,
    }


def test_peak_infected_decreases_with_interventions(default_sir_params):
    """Tests that peak infected decreases with stronger interventions."""
    actions = [
        InterventionAction.NO,
        InterventionAction.MILD,
        InterventionAction.MODERATE,
        InterventionAction.SEVERE,
    ]
    beta_0 = default_sir_params["beta"]

    peak_infected = []

    for action in actions:
        beta_t = action.apply_to_beta(beta_0)
        params = default_sir_params.copy()
        params["beta"] = beta_t

        t, S, I, R = run_sir_model(**params)
        peak_infected.append(np.max(I))

    for i in range(len(peak_infected) - 1):
        assert peak_infected[i] > peak_infected[i + 1], (
            f"Peak I for {actions[i].name} ({peak_infected[i]:.2f}) should be > "
            f"peak I for {actions[i+1].name} ({peak_infected[i+1]:.2f})"
        )


def test_severe_intervention_effectiveness(default_sir_params):
    """Tests that severe intervention significantly reduces infections compared to no intervention."""
    beta_no_intervention = InterventionAction.NO.apply_to_beta(
        default_sir_params["beta"]
    )
    beta_severe = InterventionAction.SEVERE.apply_to_beta(default_sir_params["beta"])

    # Run with no intervention
    params_no = default_sir_params.copy()
    params_no["beta"] = beta_no_intervention
    t, S_no, I_no, R_no = run_sir_model(**params_no)
    peak_no = np.max(I_no)

    # Run with severe intervention
    params_severe = default_sir_params.copy()
    params_severe["beta"] = beta_severe
    t, S_severe, I_severe, R_severe = run_sir_model(**params_severe)
    peak_severe = np.max(I_severe)

    # Severe intervention should significantly reduce peak
    assert peak_severe < peak_no * 0.9, (
        f"Severe intervention should significantly reduce peak "
        f"(peak_no={peak_no:.2f}, peak_severe={peak_severe:.2f})"
    )


def test_action_preserves_population_conservation(default_sir_params):
    """Tests that population is conserved regardless of intervention."""
    actions = list(InterventionAction)

    for action in actions:
        beta_t = action.apply_to_beta(default_sir_params["beta"])
        params = default_sir_params.copy()
        params["beta"] = beta_t

        t, S, I, R = run_sir_model(**params)
        total_population = S + I + R

        assert np.allclose(
            total_population, default_sir_params["state"].N
        ), f"Population not conserved for action '{action.name}'"
