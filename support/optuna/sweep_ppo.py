"""Optuna hyperparameter sweep for PPO agents (baseline, framestack, recurrent).

Usage:
    python -m support.optuna.sweep_ppo --agent ppo_baseline --scenario pomdp --n-trials 30
    python -m support.optuna.sweep_ppo --agent ppo_recurrent --scenario mdp --n-trials 3 -t 100000
"""

import argparse
import tempfile
from pathlib import Path
from typing import Dict

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from sb3_contrib import RecurrentPPO

from src.callbacks import StopTrainingOnNoModelImprovementWithDelta
from src.config import Config
from src.scenarios import get_scenario
from src.train import create_eval_env, create_vec_env, linear_schedule
from support.optuna.utils import NET_ARCH_MAP, TrialReportCallback, create_optuna_study

VALID_AGENTS = ("ppo_baseline", "ppo_framestack", "ppo_recurrent")


def suggest_params(trial: optuna.Trial, agent_name: str, config: Config) -> None:
    """Suggest hyperparameters for the given agent and apply them to config."""
    if agent_name == "ppo_recurrent":
        rc = config.recurrent
        rc.ent_coef = trial.suggest_float("ent_coef", 0.005, 0.5, log=True)
        rc.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        rc.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        rc.lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256])
        rc.n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
        rc.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        rc.n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10])
    else:
        ppo = config.ppo
        ppo.ent_coef = trial.suggest_float("ent_coef", 0.005, 0.5, log=True)
        ppo.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ppo.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        ppo.n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        ppo.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        net_arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys()))
        ppo.net_arch = NET_ARCH_MAP[net_arch_key][:]

        if agent_name == "ppo_framestack":
            config.n_stack = trial.suggest_categorical("n_stack", [5, 10, 15, 20, 30, 40, 50])


def objective(
    trial: optuna.Trial,
    agent_name: str,
    scenario: str,
    timesteps: int,
    seed: int,
) -> float:
    """Train PPO agent with trial-suggested hyperparameters and return best eval reward."""
    scenario_config = get_scenario(scenario)
    pomdp_params = scenario_config["pomdp_params"]

    config = Config()
    suggest_params(trial, agent_name, config)

    n_envs = config.n_envs

    with tempfile.TemporaryDirectory() as tmpdir:
        monitor_dir = Path(tmpdir) / "monitor"
        monitor_dir.mkdir()

        env = create_vec_env(config, pomdp_params, seed, n_envs=n_envs, monitor_dir=monitor_dir, agent_name=agent_name)
        eval_env = create_eval_env(config, pomdp_params, seed, agent_name=agent_name)

        is_recurrent = agent_name == "ppo_recurrent"
        agent_config = config.recurrent if is_recurrent else config.ppo
        patience = agent_config.early_stop_patience
        min_evals = agent_config.early_stop_min_evals

        stop_callback = StopTrainingOnNoModelImprovementWithDelta(
            max_no_improvement_evals=patience,
            min_evals=min_evals,
            min_delta=config.early_stop_min_delta,
            verbose=0,
        )
        trial_callback = TrialReportCallback(trial)

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

        if agent_name == "ppo_recurrent":
            rc = config.recurrent
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                verbose=0,
                seed=seed,
                policy_kwargs={"lstm_hidden_size": rc.lstm_hidden_size},
                n_steps=rc.n_steps,
                batch_size=rc.batch_size,
                n_epochs=rc.n_epochs,
                gamma=rc.gamma,
                clip_range=rc.clip_range,
                ent_coef=rc.ent_coef,
                learning_rate=linear_schedule(rc.learning_rate, timesteps, config.lr_decay_steps),
            )
        else:
            ppo = config.ppo
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                policy_kwargs={"net_arch": ppo.net_arch},
                n_steps=ppo.n_steps,
                batch_size=ppo.batch_size,
                gamma=ppo.gamma,
                clip_range=ppo.clip_range,
                ent_coef=ppo.ent_coef,
                learning_rate=linear_schedule(ppo.learning_rate, timesteps, config.lr_decay_steps),
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


def _print_config_overrides(params: Dict, agent_name: str) -> None:
    """Print best hyperparameters as config.py field overrides."""
    if agent_name == "ppo_recurrent":
        print(f"\n# RecurrentPPOConfig overrides:")
        print(f"ent_coef: float = {params['ent_coef']:.4f}")
        print(f"learning_rate: float = {params['learning_rate']:.6f}")
        print(f"gamma: float = {params['gamma']:.4f}")
        print(f"lstm_hidden_size: int = {params['lstm_hidden_size']}")
        print(f"n_steps: int = {params['n_steps']}")
        print(f"batch_size: int = {params['batch_size']}")
        print(f"n_epochs: int = {params['n_epochs']}")
    else:
        print(f"\n# PPOConfig overrides:")
        print(f"ent_coef: float = {params['ent_coef']:.4f}")
        print(f"learning_rate: float = {params['learning_rate']:.6f}")
        print(f"gamma: float = {params['gamma']:.4f}")
        print(f"n_steps: int = {params['n_steps']}")
        print(f"batch_size: int = {params['batch_size']}")
        print(f"net_arch: List[int] = field(default_factory=lambda: {NET_ARCH_MAP[params['net_arch']]})")

        if agent_name == "ppo_framestack":
            print(f"\n# Config override:")
            print(f"n_stack: int = {params['n_stack']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for PPO agents")
    parser.add_argument("--agent", "-a", required=True, choices=VALID_AGENTS, help="PPO agent variant")
    parser.add_argument("--scenario", "-s", required=True, help="Scenario name (e.g. pomdp, mdp)")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--timesteps", "-t", type=int, default=None, help="Timesteps per trial (default: config.total_timesteps)")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--study-name", default=None, help="Optuna study name (default: {agent}_{scenario})")
    args = parser.parse_args()

    timesteps = args.timesteps or Config().total_timesteps
    study_name = args.study_name or f"{args.agent}_{args.scenario}"

    study = create_optuna_study(study_name)

    study.optimize(
        lambda trial: objective(trial, args.agent, args.scenario, timesteps, args.seed),
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

    _print_config_overrides(study.best_trial.params, args.agent)


if __name__ == "__main__":
    main()
