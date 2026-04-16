"""Tests for custom training callbacks."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.callbacks import StopTrainingOnNoModelImprovementWithDelta


def _make_callback(
    patience: int = 3,
    min_evals: int = 0,
    min_delta: float = 0.05,
) -> StopTrainingOnNoModelImprovementWithDelta:
    """Build a callback with a fake EvalCallback parent (only `best_mean_reward` is used)."""
    cb = StopTrainingOnNoModelImprovementWithDelta(
        max_no_improvement_evals=patience,
        min_evals=min_evals,
        min_delta=min_delta,
        verbose=0,
    )
    cb.parent = SimpleNamespace(best_mean_reward=-np.inf)
    return cb


def _step(cb: StopTrainingOnNoModelImprovementWithDelta, best_reward: float) -> bool:
    """Emulate one eval: set parent.best_mean_reward, bump n_calls, invoke `_on_step`."""
    cb.parent.best_mean_reward = best_reward
    cb.n_calls += 1
    return cb._on_step()


def test_significant_improvements_keep_counter_at_zero():
    """Each eval bumps best reward by > min_delta → counter never grows."""
    cb = _make_callback(patience=3, min_evals=0, min_delta=0.05)

    rewards = [-3.0, -2.9, -2.8, -2.7, -2.6, -2.5]
    for r in rewards:
        assert _step(cb, r) is True
        assert cb.no_improvement_evals == 0

    assert cb.last_significant_best == pytest.approx(-2.5)


def test_micro_improvements_accumulate_then_reset_on_threshold_crossing():
    """Tiny per-eval bumps (0.01) accumulate counter until cumulative delta > min_delta."""
    cb = _make_callback(patience=100, min_evals=0, min_delta=0.05)

    # First eval: delta vs -inf is inf > 0.05 → counted as significant, anchor=-3.0.
    assert _step(cb, -3.00) is True
    assert cb.no_improvement_evals == 0
    assert cb.last_significant_best == pytest.approx(-3.00)

    # Evals 2-6: each step is +0.01, cumulative delta 0.01..0.05 — none strictly > 0.05.
    for i, r in enumerate([-2.99, -2.98, -2.97, -2.96, -2.95], start=1):
        assert _step(cb, r) is True
        assert cb.no_improvement_evals == i
        assert cb.last_significant_best == pytest.approx(-3.00)

    # Eval 7: cumulative delta = 0.06 > 0.05 → anchor shifts, counter resets.
    assert _step(cb, -2.94) is True
    assert cb.no_improvement_evals == 0
    assert cb.last_significant_best == pytest.approx(-2.94)


def test_stagnation_triggers_stop_after_patience_plus_one():
    """Constant reward → counter grows each eval, stops after max_no_improvement_evals + 1."""
    cb = _make_callback(patience=3, min_evals=0, min_delta=0.05)

    # First eval: -inf → -3.0 counts as significant.
    assert _step(cb, -3.0) is True
    assert cb.no_improvement_evals == 0

    # Three "bad" evals allowed (patience=3).
    for expected in [1, 2, 3]:
        assert _step(cb, -3.0) is True
        assert cb.no_improvement_evals == expected

    # Fourth bad eval: counter=4 > patience=3 → stop.
    assert _step(cb, -3.0) is False
    assert cb.no_improvement_evals == 4


def test_warmup_blocks_counter_and_stop():
    """During the first `min_evals` calls, counter stays at 0 even with stagnant reward."""
    cb = _make_callback(patience=1, min_evals=3, min_delta=0.05)

    # Calls 1..3 satisfy n_calls <= min_evals=3 → no-op branch, counter frozen at 0.
    for _ in range(3):
        assert _step(cb, -3.0) is True
        assert cb.no_improvement_evals == 0
        assert cb.last_significant_best == -np.inf

    # Call 4: n_calls > min_evals, first significant improvement (-inf → -3.0) anchors.
    assert _step(cb, -3.0) is True
    assert cb.no_improvement_evals == 0
    assert cb.last_significant_best == pytest.approx(-3.0)

    # Two more stagnant evals: counter=1 (allowed, == patience), counter=2 > 1 → stop.
    assert _step(cb, -3.0) is True
    assert cb.no_improvement_evals == 1
    assert _step(cb, -3.0) is False
    assert cb.no_improvement_evals == 2


def test_zero_min_delta_matches_original_sb3_semantics():
    """With min_delta=0, any strictly positive improvement resets counter (original SB3 behavior)."""
    cb = _make_callback(patience=5, min_evals=0, min_delta=0.0)

    assert _step(cb, -3.0) is True
    assert cb.no_improvement_evals == 0

    # Tiny positive bump (0.001) still counts as significant because delta > 0.
    assert _step(cb, -2.999) is True
    assert cb.no_improvement_evals == 0
    assert cb.last_significant_best == pytest.approx(-2.999)

    # Exact equality → not strictly greater → counter increments.
    assert _step(cb, -2.999) is True
    assert cb.no_improvement_evals == 1
