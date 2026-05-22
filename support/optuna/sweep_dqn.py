"""Optuna hyperparameter sweep for DQN agent.

Usage:
    python -m support.optuna.sweep_dqn --scenario pomdp --n-trials 30
    python -m support.optuna.sweep_dqn --scenario pomdp --n-trials 3 -t 100000  # quick test
"""

import argparse
import tempfile
from pathlib import Path

import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from src.callbacks import StopTrainingOnNoModelImprovementWithDelta
from src.config import Config
from src.scenarios import get_scenario
from src.train import create_eval_env, create_vec_env
from support.optuna.utils import NET_ARCH_MAP, TrialReportCallback, create_optuna_study


def objective(
    trial: optuna.Trial,
    scenario: str,
    timesteps: int,
    seed: int,
) -> float:
    """Train DQN with trial-suggested hyperparameters and return best eval reward."""
    scenario_config = get_scenario(scenario)
    pomdp_params = scenario_config["pomdp_params"]

    config = Config()

    config.dqn_gamma = trial.suggest_float("gamma", 0.90, 0.999)
    config.dqn_learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config.dqn_batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    config.dqn_exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    config.dqn_exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.3)
    config.dqn_tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    config.dqn_gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
    net_arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys()))
    config.dqn_net_arch = NET_ARCH_MAP[net_arch_key][:]

    with tempfile.TemporaryDirectory() as tmpdir:
        monitor_dir = Path(tmpdir) / "monitor"
        monitor_dir.mkdir()

        env = create_vec_env(config, pomdp_params, seed, n_envs=1, monitor_dir=monitor_dir, agent_name="dqn")
        eval_env = create_eval_env(config, pomdp_params, seed, agent_name="dqn")

        stop_callback = StopTrainingOnNoModelImprovementWithDelta(
            max_no_improvement_evals=config.dqn_early_stop_patience,
            min_evals=config.dqn_early_stop_min_evals,
            min_delta=config.early_stop_min_delta,
            verbose=0,
        )
        trial_callback = TrialReportCallback(trial)

        n_envs = 1
        adjusted_eval_freq = max(1, config.eval_freq // n_envs)
        eval_callback = EvalCallback(
            eval_env,
            callback_after_eval=CallbackList([stop_callback, trial_callback]),
            n_eval_episodes=config.n_eval_episodes,
            eval_freq=adjusted_eval_freq,
            best_model_save_path=None,
            log_path=None,
            deterministic=True,
            verbose=0,
        )

        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            policy_kwargs={"net_arch": config.dqn_net_arch},
            learning_rate=config.dqn_learning_rate,
            buffer_size=config.dqn_buffer_size,
            learning_starts=config.dqn_learning_starts,
            batch_size=config.dqn_batch_size,
            tau=config.dqn_tau,
            gradient_steps=config.dqn_gradient_steps,
            gamma=config.dqn_gamma,
            train_freq=4,
            target_update_interval=config.dqn_target_update_interval,
            exploration_fraction=config.dqn_exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=config.dqn_exploration_final_eps,
        )

        try:
            model.learn(
                total_timesteps=timesteps,
                callback=CallbackList([eval_callback]),
                progress_bar=False,
            )
        finally:
            env.close()
            eval_env.close()

        best_reward = eval_callback.best_mean_reward

    return best_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for DQN")
    parser.add_argument("--scenario", "-s", required=True, help="Scenario name (e.g. pomdp, mdp)")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--timesteps", "-t", type=int, default=None, help="Timesteps per trial (default: config.total_timesteps)")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--study-name", default=None, help="Optuna study name (default: dqn_{scenario})")
    args = parser.parse_args()

    timesteps = args.timesteps or Config().total_timesteps
    study_name = args.study_name or f"dqn_{args.scenario}"

    study = create_optuna_study(study_name)

    study.optimize(
        lambda trial: objective(trial, args.scenario, timesteps, args.seed),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best reward: {study.best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    p = study.best_trial.params
    print(f"\n# Config overrides for best DQN params:")
    print(f"dqn_learning_rate: float = {p['learning_rate']:.6f}")
    print(f"dqn_batch_size: int = {p['batch_size']}")
    print(f"dqn_exploration_fraction: float = {p['exploration_fraction']:.4f}")
    print(f"dqn_exploration_final_eps: float = {p['exploration_final_eps']:.4f}")
    print(f"dqn_tau: float = {p['tau']:.6f}")
    print(f"dqn_gradient_steps: int = {p['gradient_steps']}")
    print(f"dqn_net_arch: List[int] = field(default_factory=lambda: {NET_ARCH_MAP[p['net_arch']]})")


if __name__ == "__main__":
    main()
