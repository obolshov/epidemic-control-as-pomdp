import pytest
import numpy as np
from src.agents import InterventionAction
from src.sir import SIR
from src.simulation import Simulation
from src.agents import StaticAgent
from src.config import DefaultConfig


@pytest.fixture
def default_simulation() -> Simulation:
    return Simulation(
        agent=StaticAgent(InterventionAction.NO),
        config=DefaultConfig(),
    )


def test_intervention_values_are_valid():
    """Tests that all intervention coefficients are between 0 and 1."""
    for action in InterventionAction:
        assert (
            0 < action.value <= 1.0
        ), f"Action {action.name} has invalid coefficient: {action.value}"


def test_apply_to_beta_returns_valid_beta(default_simulation: Simulation):
    """Tests that apply_to_beta returns positive values."""
    for action in InterventionAction:
        beta_modified = default_simulation.apply_action_to_beta(action)
        assert (
            beta_modified > 0
        ), f"Modified beta for {action.name} should be positive, got {beta_modified}"
        assert (
            beta_modified <= default_simulation.beta_0
        ), f"Modified beta for {action.name} should not exceed base beta"


def test_no_intervention_preserves_beta(default_simulation: Simulation):
    """Tests that NO intervention doesn't change beta."""
    assert (
        default_simulation.apply_action_to_beta(InterventionAction.NO)
        == default_simulation.beta_0
    )


def test_beta_strictly_decreases_with_interventions(default_simulation: Simulation):
    """Tests that beta values strictly decrease with stronger interventions."""
    actions = [
        InterventionAction.NO,
        InterventionAction.MILD,
        InterventionAction.MODERATE,
        InterventionAction.SEVERE,
    ]

    beta_values = [
        default_simulation.apply_action_to_beta(action) for action in actions
    ]

    for i in range(len(beta_values) - 1):
        assert beta_values[i] > beta_values[i + 1], (
            f"Beta for {actions[i].name} ({beta_values[i]}) should be > "
            f"beta for {actions[i+1].name} ({beta_values[i+1]})"
        )


def test_peak_infected_decreases_with_interventions(default_simulation: Simulation):
    """Tests that peak infected decreases with stronger interventions."""
    actions = [
        InterventionAction.NO,
        InterventionAction.MILD,
        InterventionAction.MODERATE,
        InterventionAction.SEVERE,
    ]

    peak_infected = []

    sir = SIR()

    for action in actions:
        beta_t = default_simulation.apply_action_to_beta(action)
        t, S, I, R = sir.run_all_days(
            default_simulation.initial_state,
            beta_t,
            default_simulation.gamma,
            default_simulation.total_days,
        )
        peak_infected.append(np.max(I))

    for i in range(len(peak_infected) - 1):
        assert peak_infected[i] > peak_infected[i + 1], (
            f"Peak I for {actions[i].name} ({peak_infected[i]:.2f}) should be > "
            f"peak I for {actions[i+1].name} ({peak_infected[i+1]:.2f})"
        )


def test_severe_intervention_effectiveness(default_simulation: Simulation):
    """Tests that severe intervention significantly reduces infections compared to no intervention."""
    beta_no_intervention = default_simulation.apply_action_to_beta(
        InterventionAction.NO
    )
    beta_severe = default_simulation.apply_action_to_beta(InterventionAction.SEVERE)

    sir = SIR()

    t, S_no, I_no, R_no = sir.run_all_days(
        default_simulation.initial_state,
        beta_no_intervention,
        default_simulation.gamma,
        default_simulation.total_days,
    )
    peak_no = np.max(I_no)

    t, S_severe, I_severe, R_severe = sir.run_all_days(
        default_simulation.initial_state,
        beta_severe,
        default_simulation.gamma,
        default_simulation.total_days,
    )
    peak_severe = np.max(I_severe)

    assert peak_severe < peak_no * 0.9, (
        f"Severe intervention should significantly reduce peak "
        f"(peak_no={peak_no:.2f}, peak_severe={peak_severe:.2f})"
    )


def test_action_preserves_population_conservation(default_simulation: Simulation):
    """Tests that population is conserved regardless of intervention."""
    actions = list(InterventionAction)

    sir = SIR()

    for action in actions:
        beta_t = default_simulation.apply_action_to_beta(action)
        t, S, I, R = sir.run_all_days(
            default_simulation.initial_state,
            beta_t,
            default_simulation.gamma,
            default_simulation.total_days,
        )
        total_population = S + I + R

        assert np.allclose(
            total_population, default_simulation.initial_state.N
        ), f"Population not conserved for action '{action.name}'"
