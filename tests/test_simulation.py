import pytest
import numpy as np
from src.sir import EpidemicState
from src.simulation import run_simulation
from src.agents import StaticAgent
from src.actions import InterventionAction


@pytest.fixture
def default_state():
    """Default initial epidemic state."""
    return EpidemicState(N=1000, S=999, I=1, R=0)


@pytest.fixture
def simulation_params():
    """Default simulation parameters."""
    return {"beta_0": 0.4, "gamma": 0.1, "total_days": 100, "action_interval": 7}


class TestStaticPolicy:
    """Tests for StaticPolicy."""

    def test_static_agent_returns_same_action(self):
        """Test that static policy always returns the same action."""
        agent = StaticAgent(InterventionAction.MODERATE)
        state1 = EpidemicState(N=1000, S=900, I=50, R=50)
        state2 = EpidemicState(N=1000, S=500, I=300, R=200)

        assert agent.select_action(state1) == InterventionAction.MODERATE
        assert agent.select_action(state2) == InterventionAction.MODERATE


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_policy_simulation_basic(self, default_state, simulation_params):
        """Test basic policy simulation execution."""
        agent = StaticAgent(InterventionAction.NO)
        result = run_simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )

        expected_length = simulation_params["total_days"] + 1
        assert len(result.t) == expected_length
        assert len(result.S) == expected_length
        assert len(result.I) == expected_length
        assert len(result.R) == expected_length

    def test_policy_simulation_action_count(self, default_state, simulation_params):
        """Test that correct number of actions are recorded."""
        agent = StaticAgent(InterventionAction.MILD)
        result = run_simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )

        # Should have ceiling(total_days / action_interval) actions
        expected_actions = (
            simulation_params["total_days"] // simulation_params["action_interval"]
        ) + 1
        assert len(result.actions) == expected_actions
        assert len(result.action_timesteps) == expected_actions

    def test_policy_simulation_action_timesteps(self, default_state, simulation_params):
        """Test that action timesteps are at correct intervals."""
        agent = StaticAgent(InterventionAction.NO)
        result = run_simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )

        interval = simulation_params["action_interval"]
        for i, timestep in enumerate(result.action_timesteps[:-1]):
            assert timestep == i * interval

    def test_policy_simulation_static_all_same_action(
        self, default_state, simulation_params
    ):
        """Test that static policy uses same action throughout."""
        action = InterventionAction.MODERATE
        agent = StaticAgent(action)
        result = run_simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )

        assert all(a == action for a in result.actions)

    def test_policy_simulation_severe_reduces_infections(self, default_state):
        """Test that severe intervention reduces infections more than no intervention."""
        params = {"beta_0": 0.5, "gamma": 0.1, "total_days": 100, "action_interval": 7}

        agent_no = StaticAgent(InterventionAction.NO)
        result_no = run_simulation(
            agent=agent_no, initial_state=default_state, **params
        )

        agent_severe = StaticAgent(InterventionAction.SEVERE)
        result_severe = run_simulation(
            agent=agent_severe, initial_state=default_state, **params
        )

        assert (
            result_severe.peak_infected < result_no.peak_infected
        ), "Severe intervention should reduce peak infections"
        assert (
            result_severe.total_infected < result_no.total_infected
        ), "Severe intervention should reduce total infections"

    def test_policy_simulation_result_properties(
        self, default_state, simulation_params
    ):
        """Test that result properties are calculated correctly."""
        agent = StaticAgent(InterventionAction.MODERATE)
        result = run_simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )

        # Test peak_infected
        assert result.peak_infected == np.max(result.I)

        # Test total_infected
        assert result.total_infected == result.R[-1] + result.I[-1]
