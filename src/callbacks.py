"""Custom training callbacks for Stable Baselines 3."""

from typing import ClassVar

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class StopTrainingOnNoModelImprovementWithDelta(BaseCallback):
    """Stop training when `best_mean_reward` fails to improve by at least `min_delta`.

    Drop-in replacement for SB3's :class:`StopTrainingOnNoModelImprovement` that
    handles noisy evaluations correctly. An evaluation counts as a *significant*
    improvement only if `best_mean_reward` exceeds the last significant best by
    at least `min_delta`. Micro-fluctuations on a plateau do not reset the
    patience counter, while genuine slow drift still does — because the anchor
    (`last_significant_best`) is updated only on significant steps, cumulative
    small improvements eventually cross the threshold and reset the counter.

    Must be attached as `callback_after_eval` to an :class:`EvalCallback`.

    Args:
        max_no_improvement_evals: Stop after this many consecutive evaluations
            without a significant improvement.
        min_evals: Minimum number of evaluations before stopping can trigger
            (warmup period).
        min_delta: Minimum increase in `best_mean_reward` (vs the last
            significant best) required to count as an improvement. Set to 0 for
            behavior identical to SB3's original callback (any new best counts).
        verbose: 0 for silent, 1 to print when stopping.
    """

    parent: ClassVar[EvalCallback]

    def __init__(
        self,
        max_no_improvement_evals: int,
        min_evals: int = 0,
        min_delta: float = 0.0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.min_delta = min_delta
        self.last_significant_best = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "StopTrainingOnNoModelImprovementWithDelta must be used with an EvalCallback"
        )

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward - self.last_significant_best > self.min_delta:
                self.no_improvement_evals = 0
                self.last_significant_best = self.parent.best_mean_reward
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training: no improvement > {self.min_delta:.3f} in "
                f"the last {self.no_improvement_evals} evaluations "
                f"(last significant best: {self.last_significant_best:.3f})"
            )

        return continue_training
