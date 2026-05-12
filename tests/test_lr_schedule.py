"""Tests for linear LR schedule."""

import pytest
from src.train import linear_schedule


INITIAL_LR = 3e-4
MIN_LR = INITIAL_LR * 0.1
DECAY_STEPS = 2_000_000


def _progress_remaining(step: int, total: int) -> float:
    """Convert absolute step to SB3's progress_remaining."""
    return 1.0 - step / total


class TestLinearSchedule:
    def test_starts_at_initial_value(self):
        schedule = linear_schedule(INITIAL_LR, 3_000_000, DECAY_STEPS)
        assert schedule(1.0) == pytest.approx(INITIAL_LR)

    def test_reaches_floor_at_decay_steps(self):
        total = 3_000_000
        schedule = linear_schedule(INITIAL_LR, total, DECAY_STEPS)
        pr = _progress_remaining(DECAY_STEPS, total)
        assert schedule(pr) == pytest.approx(MIN_LR)

    def test_holds_floor_after_decay_steps(self):
        total = 3_000_000
        schedule = linear_schedule(INITIAL_LR, total, DECAY_STEPS)
        for step in [DECAY_STEPS + 1, DECAY_STEPS + 500_000, total]:
            pr = _progress_remaining(step, total)
            assert schedule(pr) == pytest.approx(MIN_LR)

    def test_monotonically_decreasing(self):
        total = 3_000_000
        schedule = linear_schedule(INITIAL_LR, total, DECAY_STEPS)
        steps = range(0, total, 10_000)
        lrs = [schedule(_progress_remaining(s, total)) for s in steps]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1]

    def test_no_discontinuity_at_boundary(self):
        total = 3_000_000
        schedule = linear_schedule(INITIAL_LR, total, DECAY_STEPS)
        just_before = _progress_remaining(DECAY_STEPS - 1, total)
        at_boundary = _progress_remaining(DECAY_STEPS, total)
        assert abs(schedule(just_before) - schedule(at_boundary)) < 1e-8

    def test_independent_of_total_timesteps(self):
        """LR at a given step must not change when total_timesteps changes."""
        step = 1_000_000
        s1 = linear_schedule(INITIAL_LR, 2_000_000, DECAY_STEPS)
        s2 = linear_schedule(INITIAL_LR, 5_000_000, DECAY_STEPS)
        lr1 = s1(_progress_remaining(step, 2_000_000))
        lr2 = s2(_progress_remaining(step, 5_000_000))
        assert lr1 == pytest.approx(lr2)

    def test_midpoint_value(self):
        total = 3_000_000
        schedule = linear_schedule(INITIAL_LR, total, DECAY_STEPS)
        mid = DECAY_STEPS // 2
        pr = _progress_remaining(mid, total)
        expected = MIN_LR + (INITIAL_LR - MIN_LR) * 0.5
        assert schedule(pr) == pytest.approx(expected)

    def test_never_below_floor(self):
        total = 3_000_000
        schedule = linear_schedule(INITIAL_LR, total, DECAY_STEPS)
        steps = range(0, total, 10_000)
        for s in steps:
            assert schedule(_progress_remaining(s, total)) >= MIN_LR - 1e-12
