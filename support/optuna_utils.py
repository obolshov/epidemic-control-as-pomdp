"""Shared utilities for Optuna hyperparameter sweeps."""

from typing import ClassVar

import optuna
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

NET_ARCH_MAP = {
    "64_64": [64, 64],
    "128_128": [128, 128],
    "256_256": [256, 256],
    "64_64_64": [64, 64, 64],
}


class TrialReportCallback(BaseCallback):
    """Report eval reward to Optuna trial and check pruning."""

    parent: ClassVar[EvalCallback]

    def __init__(self, trial: optuna.Trial) -> None:
        super().__init__()
        self.trial = trial

    def _on_step(self) -> bool:
        self.trial.report(self.parent.best_mean_reward, self.n_calls)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def create_optuna_study(study_name: str) -> optuna.Study:
    """Create a standardized Optuna study with TPE sampler and median pruner."""
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
        ),
        storage=f"sqlite:///optuna_{study_name}.db",
        load_if_exists=True,
    )
