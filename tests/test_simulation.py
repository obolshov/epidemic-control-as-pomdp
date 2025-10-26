import pytest
import numpy as np
from src.sir import EpidemicState
from src.simulation import Simulation
from src.agents import StaticAgent, InterventionAction


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
        simulation = Simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )
        result = simulation.run()

        expected_length = simulation_params["total_days"] + 1
        assert len(result.t) == expected_length
        assert len(result.S) == expected_length
        assert len(result.I) == expected_length
        assert len(result.R) == expected_length

    def test_policy_simulation_action_count(self, default_state, simulation_params):
        """Test that correct number of actions are recorded."""
        agent = StaticAgent(InterventionAction.MILD)
        simulation = Simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )
        result = simulation.run()

        # Should have ceiling(total_days / action_interval) actions
        expected_actions = (
            simulation_params["total_days"] // simulation_params["action_interval"]
        ) + 1
        assert len(result.actions) == expected_actions
        assert len(result.action_timesteps) == expected_actions

    def test_policy_simulation_action_timesteps(self, default_state, simulation_params):
        """Test that action timesteps are at correct intervals."""
        agent = StaticAgent(InterventionAction.NO)
        simulation = Simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )
        result = simulation.run()

        interval = simulation_params["action_interval"]
        for i, timestep in enumerate(result.action_timesteps[:-1]):
            assert timestep == i * interval

    def test_policy_simulation_static_all_same_action(
        self, default_state, simulation_params
    ):
        """Test that static policy uses same action throughout."""
        action = InterventionAction.MODERATE
        agent = StaticAgent(action)
        simulation = Simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )
        result = simulation.run()

        assert all(a == action for a in result.actions)

    def test_policy_simulation_severe_reduces_infections(self, default_state):
        """Test that severe intervention reduces infections more than no intervention."""
        params = {"beta_0": 0.5, "gamma": 0.1, "total_days": 100, "action_interval": 7}

        agent_no = StaticAgent(InterventionAction.NO)
        simulation = Simulation(agent=agent_no, initial_state=default_state, **params)
        result_no = simulation.run()

        agent_severe = StaticAgent(InterventionAction.SEVERE)
        simulation_severe = Simulation(
            agent=agent_severe, initial_state=default_state, **params
        )
        result_severe = simulation_severe.run()

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
        simulation = Simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )
        result = simulation.run()

        # Test peak_infected
        assert result.peak_infected == np.max(result.I)

        # Test total_infected
        assert result.total_infected == result.R[-1] + result.I[-1]

    def test_policy_simulation_has_rewards(self, default_state, simulation_params):
        """Test that simulation result includes rewards."""
        agent = StaticAgent(InterventionAction.NO)
        simulation = Simulation(
            agent=agent, initial_state=default_state, **simulation_params
        )
        result = simulation.run()

        # Should have same number of rewards as actions
        assert len(result.rewards) == len(result.actions)


class TestRewardCalculation:
    """Tests for reward calculation functions."""

    @pytest.fixture
    def default_simulation(self, default_state, simulation_params) -> Simulation:
        return Simulation(
            agent=StaticAgent(InterventionAction.NO),
            initial_state=default_state,
            **simulation_params
        )

    def test_reward_no_growth_no_action(self, default_simulation: Simulation):
        """Test reward when no infection growth and no action."""
        reward = default_simulation.calculate_reward(
            100.0, 100.0, InterventionAction.NO
        )

        # No growth penalty (log(1)=0)
        # No action penalty (stringency=0)
        expected = 0.0
        assert np.isclose(reward, expected)

    def test_reward_doubling_no_action(self, default_simulation: Simulation):
        """Test reward when infections double."""
        reward = default_simulation.calculate_reward(
            100.0, 200.0, InterventionAction.NO
        )

        # Growth penalty: -1.0 * log(2) = -0.693
        # Action penalty: 0
        expected = -0.693
        assert np.isclose(reward, expected, atol=0.01)

    def test_reward_halving_severe_action(self, default_simulation: Simulation):
        """Test reward when infections halve with severe action."""
        reward = default_simulation.calculate_reward(
            200.0, 100.0, InterventionAction.SEVERE
        )

        # Growth penalty: -1.0 * log(0.5) = 0.693
        # Action penalty: -0.1 * 1.0 = -0.1
        expected = 0.593
        assert np.isclose(reward, expected, atol=0.01)

    def test_reward_custom_weights(self, default_simulation: Simulation):
        """Test reward calculation with custom weights."""
        default_simulation.w_I = 2.0
        default_simulation.w_S = 0.5

        reward = default_simulation.calculate_reward(
            100.0, 200.0, InterventionAction.MODERATE
        )

        # Growth penalty: -2.0 * log(2) = -1.386
        # Action penalty: -0.5 * 0.5 = -0.25
        expected = -1.636
        assert np.isclose(reward, expected, atol=0.01)

    def test_reward_zero_initial_infections(self, default_simulation: Simulation):
        """Test reward calculation when starting from zero infections."""
        reward = default_simulation.calculate_reward(0.0, 100.0, InterventionAction.NO)

        # Should handle zero gracefully
        # Growth ratio should be 0 (no penalty for increasing from zero)
        # Action penalty: 0
        expected = 0.0
        assert np.isclose(reward, expected)
