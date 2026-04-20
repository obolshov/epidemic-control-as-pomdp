"""Tests for EpidemicEnv: step/reward semantics and calculate_reward."""

import numpy as np
import pytest

from src.config import Config
from src.env import EpidemicEnv, InterventionAction, calculate_reward


@pytest.fixture
def small_config() -> Config:
    """Deterministic config; reward weights pinned to known values."""
    return Config(
        N=1000, E0=10, I0=5, stochastic=False,
        days=100, action_interval=5,
        w_I=10.0, w_S=0.1, w_switch=0.05,
    )


class TestStepRewardSemantics:
    """Verify that obs, stringency, and switching all track the current action."""

    def test_obs_prev_action_tracks_action(self, small_config):
        env = EpidemicEnv(small_config)
        env.reset(seed=0)
        for a in [3, 3, 3, 0, 0]:
            obs, _, _, _, _ = env.step(a)
            assert obs[4] == float(a)

    def test_obs_and_rewards(self, small_config):
        env = EpidemicEnv(small_config)
        env.reset(seed=0)
        w_switch = small_config.w_switch
        w_S = small_config.w_S

        sequence = [0, 3, 3, 2, 0]
        prev = 0
        for a in sequence:
            obs, _, _, _, info = env.step(a)
            assert obs[4] == float(a)
            expected_stringency = w_S * (1 - list(InterventionAction)[a].value)
            assert info["reward_stringency"] == pytest.approx(-expected_stringency)
            expected_switching = w_switch * (a - prev) ** 2
            assert info["reward_switching"] == pytest.approx(-expected_switching)
            prev = a


class TestCalculateReward:
    def test_zero_stringency_for_no_intervention(self, small_config):
        _, comps = calculate_reward(
            I_t=0.0,
            action=InterventionAction.NO,
            action_idx=0,
            prev_action_idx=0,
            config=small_config,
        )
        assert comps["reward_stringency"] == pytest.approx(0.0)

    def test_switching_penalty_on_action_change(self, small_config):
        _, comps = calculate_reward(
            I_t=0.0,
            action=InterventionAction.SEVERE,
            action_idx=3,
            prev_action_idx=0,
            config=small_config,
        )
        assert comps["reward_switching"] == pytest.approx(-small_config.w_switch * 9)
