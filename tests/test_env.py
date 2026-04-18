"""Tests for EpidemicEnv: applied/selected semantics of prev_selected_idx and reward."""

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


# ---------------------------------------------------------------------------
# action_delay > 0: applied ≠ selected
# ---------------------------------------------------------------------------

class TestAppliedSelectedSemantics:
    """With action_delay=2, verify that obs/switching track selected while
    stringency/β track applied."""

    DELAY = 2

    def test_obs_prev_action_tracks_selected(self, small_config):
        env = EpidemicEnv(small_config, action_delay=self.DELAY)
        env.reset(seed=0)
        for a in [3, 3, 3, 0, 0]:
            obs, _, _, _, _ = env.step(a)
            assert obs[4] == float(a)

    def test_switching_penalty_on_selected_delta(self, small_config):
        env = EpidemicEnv(small_config, action_delay=self.DELAY)
        env.reset(seed=0)
        w_switch = small_config.w_switch

        # Step 1: select SEVERE (idx=3); prev_selected=0 → delta=3.
        _, _, _, _, info = env.step(3)
        assert info["reward_switching"] == pytest.approx(-w_switch * 9), (
            "switching penalty should fire on selected delta even while "
            "applied is still 0 (queued SEVERE not yet out)"
        )

        # Step 2: select SEVERE again; prev_selected=3 → delta=0.
        _, _, _, _, info = env.step(3)
        assert info["reward_switching"] == pytest.approx(0.0)

        # Step 3: switch to NO (idx=0); prev_selected=3 → delta=-3 → delta²=9.
        _, _, _, _, info = env.step(0)
        assert info["reward_switching"] == pytest.approx(-w_switch * 9)

    def test_stringency_penalty_on_applied(self, small_config):
        env = EpidemicEnv(small_config, action_delay=self.DELAY)
        env.reset(seed=0)
        w_S = small_config.w_S

        # For first `DELAY` steps, queue yields applied=NO (initial zeros).
        # NO.value = 1.0 → stringency = w_S * (1 - 1.0) = 0.
        for _ in range(self.DELAY):
            _, _, _, _, info = env.step(3)  # selected=SEVERE, applied=NO
            assert info["reward_stringency"] == pytest.approx(0.0), (
                "while queue still emits NO, stringency must stay 0 regardless "
                "of what is selected"
            )

        # Step DELAY+1: the first SEVERE comes out of the queue.
        # SEVERE.value = 0.25 → stringency = w_S * (1 - 0.25) = 0.075.
        _, _, _, _, info = env.step(3)
        assert info["reward_stringency"] == pytest.approx(-w_S * 0.75)

    def test_beta_follows_applied(self, small_config):
        """β is applied-driven: selecting SEVERE under delay must not reduce β
        during the first DELAY steps. Compare I trajectories between an env
        where the agent selects SEVERE immediately and one where it always
        selects NO — they must coincide for the first DELAY steps."""
        env_severe = EpidemicEnv(small_config, action_delay=self.DELAY)
        env_no = EpidemicEnv(small_config, action_delay=self.DELAY)
        env_severe.reset(seed=0)
        env_no.reset(seed=0)

        for step_idx in range(self.DELAY):
            obs_s, _, _, _, _ = env_severe.step(3)
            obs_n, _, _, _, _ = env_no.step(0)
            # Compartments S, E, I, R (first 4 elements) must match: β=β_0*1.0
            # for both, since queue emits NO in both.
            np.testing.assert_allclose(
                obs_s[:4], obs_n[:4],
                err_msg=f"step {step_idx}: applied=NO in both cases, I trajectory "
                        "should coincide",
            )

        # After DELAY steps, applied diverges → I trajectories must differ.
        obs_s, _, _, _, _ = env_severe.step(3)
        obs_n, _, _, _, _ = env_no.step(0)
        assert obs_s[2] < obs_n[2], (
            "after DELAY steps applied=SEVERE (env_severe) vs applied=NO "
            "(env_no) → I should be lower in env_severe"
        )


# ---------------------------------------------------------------------------
# action_delay = 0: selected ≡ applied → MDP regression
# ---------------------------------------------------------------------------

class TestMDPRegression:
    """Without action_delay the semantic refactor must be a no-op: behavior
    must match the legacy applied-everywhere convention byte-for-byte."""

    def test_obs_and_rewards_match_legacy(self, small_config):
        env = EpidemicEnv(small_config, action_delay=0)
        env.reset(seed=0)
        w_switch = small_config.w_switch
        w_S = small_config.w_S

        sequence = [0, 3, 3, 2, 0]
        prev = 0
        for a in sequence:
            obs, _, _, _, info = env.step(a)
            # obs[4] = selected == applied (no delay) == `a`
            assert obs[4] == float(a)
            # Stringency: applied == selected == a
            expected_stringency = w_S * (1 - list(InterventionAction)[a].value)
            assert info["reward_stringency"] == pytest.approx(-expected_stringency)
            # Switching: delta = a - prev (selected-to-selected == applied-to-applied)
            expected_switching = w_switch * (a - prev) ** 2
            assert info["reward_switching"] == pytest.approx(-expected_switching)
            prev = a


# ---------------------------------------------------------------------------
# calculate_reward unit tests (pure function)
# ---------------------------------------------------------------------------

class TestCalculateReward:
    def test_stringency_from_applied_only(self, small_config):
        """With applied=NO but selected=SEVERE, stringency must reflect NO (0)."""
        _, comps = calculate_reward(
            I_t=0.0,
            applied_action=InterventionAction.NO,
            selected_action_idx=3,
            prev_selected_idx=0,
            config=small_config,
        )
        assert comps["reward_stringency"] == pytest.approx(0.0)

    def test_switching_from_selected_only(self, small_config):
        """With applied=NO (no delta in applied), switching must still fire on
        selected transition 0 → SEVERE."""
        _, comps = calculate_reward(
            I_t=0.0,
            applied_action=InterventionAction.NO,
            selected_action_idx=3,
            prev_selected_idx=0,
            config=small_config,
        )
        assert comps["reward_switching"] == pytest.approx(-small_config.w_switch * 9)
