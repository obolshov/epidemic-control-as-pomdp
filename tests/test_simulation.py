import pytest
import numpy as np
from src.sir import EpidemicState
from src.simulation import Simulation
from src.agents import StaticAgent, InterventionAction
from src.config import DefaultConfig


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

    def test_policy_simulation_basic(self):
        """Test basic policy simulation execution."""
        agent = StaticAgent(InterventionAction.NO)
        config = DefaultConfig()
        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()

        expected_length = config.days + 1
        assert len(result.t) == expected_length
        assert len(result.S) == expected_length
        assert len(result.I) == expected_length
        assert len(result.R) == expected_length

    def test_policy_simulation_action_count(self):
        """Test that correct number of actions are recorded."""
        agent = StaticAgent(InterventionAction.MILD)
        config = DefaultConfig()

        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()

        # Should have ceiling(total_days / action_interval) actions
        expected_actions = (config.days // config.action_interval) + 1
        assert len(result.actions) == expected_actions
        assert len(result.action_timesteps) == expected_actions

    def test_policy_simulation_action_timesteps(self):
        """Test that action timesteps are at correct intervals."""
        agent = StaticAgent(InterventionAction.NO)
        config = DefaultConfig()

        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()

        interval = config.action_interval
        for i, timestep in enumerate(result.action_timesteps[:-1]):
            assert timestep == i * interval

    def test_policy_simulation_static_all_same_action(self):
        """Test that static policy uses same action throughout."""
        action = InterventionAction.MODERATE
        agent = StaticAgent(action)
        config = DefaultConfig()

        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()

        assert all(a == action for a in result.actions)

    def test_policy_simulation_severe_reduces_infections(self):
        """Test that severe intervention reduces infections more than no intervention."""
        agent_no = StaticAgent(InterventionAction.NO)
        config = DefaultConfig()

        simulation = Simulation(agent=agent_no, config=config)
        result_no = simulation.run()

        agent_severe = StaticAgent(InterventionAction.SEVERE)
        simulation_severe = Simulation(agent=agent_severe, config=config)
        result_severe = simulation_severe.run()

        assert (
            result_severe.peak_infected < result_no.peak_infected
        ), "Severe intervention should reduce peak infections"
        assert (
            result_severe.total_infected < result_no.total_infected
        ), "Severe intervention should reduce total infections"

    def test_policy_simulation_result_properties(self):
        """Test that result properties are calculated correctly."""
        agent = StaticAgent(InterventionAction.MODERATE)
        config = DefaultConfig()

        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()

        # Test peak_infected
        assert result.peak_infected == np.max(result.I)

        # Test total_infected
        assert result.total_infected == result.R[-1] + result.I[-1]

    def test_policy_simulation_has_rewards(self):
        """Test that simulation result includes rewards."""
        agent = StaticAgent(InterventionAction.NO)
        config = DefaultConfig()

        simulation = Simulation(agent=agent, config=config)
        result = simulation.run()

        # Should have same number of rewards as actions
        assert len(result.rewards) == len(result.actions)
