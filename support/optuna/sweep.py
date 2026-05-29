"""Optuna hyperparameter sweep for all RL agents.

Multi-seed evaluation: each trial trains N seeds sequentially.
Seed 0 uses within-seed Optuna pruning; seeds 1+ use only early stopping.
Cross-seed pruning kills trials whose running mean is below median after any seed.

Usage:
    python -m support.optuna.sweep --agent ppo_baseline --n-trials 30
    python -m support.optuna.sweep --agent dqn --n-trials 30
    python -m support.optuna.sweep --agent ppo_recurrent -t 1000 --num-seeds 2
"""

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from sb3_contrib import RecurrentPPO

from src.callbacks import StopTrainingOnNoModelImprovementWithDelta
from src.config import Config
from src.scenarios import get_scenario
from src.train import create_eval_env, create_vec_env, linear_schedule
from support.optuna.utils import (
    CROSS_SEED_BASE,
    NET_ARCH_MAP,
    TrialReportCallback,
    create_optuna_study,
)

VALID_AGENTS = ("ppo_baseline", "ppo_framestack", "ppo_recurrent", "dqn")


def suggest_params(trial: optuna.Trial, agent_name: str, config: Config) -> None:
    """Suggest hyperparameters for the given agent and apply them to config."""
    if agent_name == "ppo_recurrent":
        rc = config.ppo_recurrent
        rc.ent_coef = trial.suggest_float("ent_coef", 0.005, 0.5, log=True)
        rc.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        rc.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        rc.lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128])
        rc.n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
        rc.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    elif agent_name == "dqn":
        dqn = config.dqn
        dqn.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        dqn.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dqn.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        dqn.exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        dqn.exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.3)
        dqn.tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
        dqn.gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
        net_arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys()))
        dqn.net_arch = NET_ARCH_MAP[net_arch_key][:]
    else:
        ac = config.ppo_framestack if agent_name == "ppo_framestack" else config.ppo_baseline
        ac.ent_coef = trial.suggest_float("ent_coef", 0.005, 0.5, log=True)
        ac.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ac.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        ac.n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_sizes = [32, 64, 128, 256, 512] if agent_name == "ppo_framestack" else [32, 64, 128, 256]
        ac.batch_size = trial.suggest_categorical("batch_size", batch_sizes)
        net_arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys()))
        ac.net_arch = NET_ARCH_MAP[net_arch_key][:]

        if agent_name == "ppo_framestack":
            config.n_stack = trial.suggest_categorical("n_stack", [5, 10, 15, 20, 30, 40, 50])


def _create_model(
    agent_name: str,
    config: Config,
    env: VecEnv,
    seed: int,
    timesteps: int,
) -> BaseAlgorithm:
    """Create an RL model with the hyperparameters currently set in config."""
    if agent_name == "ppo_recurrent":
        rc = config.ppo_recurrent
        return RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=0,
            seed=seed,
            policy_kwargs={"lstm_hidden_size": rc.lstm_hidden_size},
            n_steps=rc.n_steps,
            batch_size=rc.batch_size,
            n_epochs=rc.n_epochs,
            gamma=rc.gamma,
            ent_coef=rc.ent_coef,
            learning_rate=linear_schedule(rc.learning_rate, timesteps, config.lr_decay_steps),
        )
    elif agent_name == "dqn":
        dqn = config.dqn
        return DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            policy_kwargs={"net_arch": dqn.net_arch},
            learning_rate=dqn.learning_rate,
            buffer_size=dqn.buffer_size,
            learning_starts=dqn.learning_starts,
            batch_size=dqn.batch_size,
            tau=dqn.tau,
            gradient_steps=dqn.gradient_steps,
            gamma=dqn.gamma,
            train_freq=4,
            target_update_interval=dqn.target_update_interval,
            exploration_fraction=dqn.exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=dqn.exploration_final_eps,
        )
    else:
        ac = config.ppo_framestack if agent_name == "ppo_framestack" else config.ppo_baseline
        return PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            policy_kwargs={"net_arch": ac.net_arch},
            n_steps=ac.n_steps,
            batch_size=ac.batch_size,
            gamma=ac.gamma,
            ent_coef=ac.ent_coef,
            learning_rate=linear_schedule(ac.learning_rate, timesteps, config.lr_decay_steps),
        )


def _get_agent_config(agent_name: str, config: Config):
    """Return the agent-specific config dataclass."""
    if agent_name == "ppo_recurrent":
        return config.ppo_recurrent
    elif agent_name == "ppo_framestack":
        return config.ppo_framestack
    elif agent_name == "dqn":
        return config.dqn
    else:
        return config.ppo_baseline


def _train_seed(
    agent_name: str,
    config: Config,
    pomdp_params: Dict,
    seed: int,
    timesteps: int,
    trial: Optional[optuna.Trial] = None,
) -> float:
    """Train one seed and return best mean eval reward.

    Args:
        agent_name: Agent type.
        config: Config with hyperparameters already set by suggest_params.
        pomdp_params: POMDP wrapper parameters from scenario.
        seed: Training seed.
        timesteps: Total training timesteps.
        trial: Optuna trial for within-seed pruning (seed 0 only). None disables pruning.

    Returns:
        Best mean evaluation reward achieved during training.
    """
    n_envs = 1 if agent_name == "dqn" else config.n_envs
    agent_config = _get_agent_config(agent_name, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        monitor_dir = Path(tmpdir) / "monitor"
        monitor_dir.mkdir()

        env = create_vec_env(config, pomdp_params, seed, n_envs=n_envs, monitor_dir=monitor_dir, agent_name=agent_name)
        eval_env = create_eval_env(config, pomdp_params, seed, agent_name=agent_name)

        stop_callback = StopTrainingOnNoModelImprovementWithDelta(
            max_no_improvement_evals=agent_config.early_stop_patience,
            min_evals=agent_config.early_stop_min_evals,
            min_delta=config.early_stop_min_delta,
            verbose=0,
        )

        after_eval_callbacks = [stop_callback]
        if trial is not None:
            after_eval_callbacks.append(TrialReportCallback(trial))

        adjusted_eval_freq = max(1, config.eval_freq // n_envs)
        eval_callback = EvalCallback(
            eval_env,
            callback_after_eval=CallbackList(after_eval_callbacks),
            n_eval_episodes=config.n_eval_episodes,
            eval_freq=adjusted_eval_freq,
            best_model_save_path=None,
            log_path=None,
            deterministic=True,
            verbose=0,
        )

        model = _create_model(agent_name, config, env, seed, timesteps)

        try:
            model.learn(
                total_timesteps=timesteps,
                callback=CallbackList([eval_callback]),
                progress_bar=False,
            )
        finally:
            env.close()
            eval_env.close()

        return eval_callback.best_mean_reward


def objective(
    trial: optuna.Trial,
    agent_name: str,
    timesteps: int,
    seeds: List[int],
) -> float:
    """Multi-seed objective: train across seeds, return mean best reward.

    Seed 0 uses within-seed Optuna pruning via TrialReportCallback.
    Seeds 1+ train with early stopping only; cross-seed pruning checks
    the running mean after each additional seed completes.
    """
    scenario_config = get_scenario("pomdp")
    pomdp_params = scenario_config["pomdp_params"]

    config = Config()
    suggest_params(trial, agent_name, config)

    rewards: List[float] = []
    for i, seed in enumerate(seeds):
        reward = _train_seed(
            agent_name, config, pomdp_params, seed, timesteps,
            trial=trial if i == 0 else None,
        )
        rewards.append(reward)
        trial.set_user_attr(f"reward_seed_{seed}", reward)

        if i > 0:
            mean_reward = float(np.mean(rewards))
            trial.report(mean_reward, CROSS_SEED_BASE + i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(rewards))


def _print_config_overrides(params: Dict, agent_name: str) -> None:
    """Print best hyperparameters as config.py field overrides."""
    config_names = {
        "ppo_baseline": "PPOBaselineConfig",
        "ppo_framestack": "PPOFrameStackConfig",
        "ppo_recurrent": "PPORecurrentConfig",
        "dqn": "DQNConfig",
    }
    print(f"\n# {config_names[agent_name]} overrides:")

    if agent_name == "ppo_recurrent":
        print(f"ent_coef: float = {params['ent_coef']:.4f}")
        print(f"learning_rate: float = {params['learning_rate']:.6f}")
        print(f"gamma: float = {params['gamma']:.4f}")
        print(f"lstm_hidden_size: int = {params['lstm_hidden_size']}")
        print(f"n_steps: int = {params['n_steps']}")
        print(f"batch_size: int = {params['batch_size']}")
    elif agent_name == "dqn":
        print(f"learning_rate: float = {params['learning_rate']:.6f}")
        print(f"gamma: float = {params['gamma']:.4f}")
        print(f"batch_size: int = {params['batch_size']}")
        print(f"exploration_fraction: float = {params['exploration_fraction']:.4f}")
        print(f"exploration_final_eps: float = {params['exploration_final_eps']:.4f}")
        print(f"tau: float = {params['tau']:.6f}")
        print(f"gradient_steps: int = {params['gradient_steps']}")
        print(f"net_arch: List[int] = field(default_factory=lambda: {NET_ARCH_MAP[params['net_arch']]})")
    else:
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
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for RL agents")
    parser.add_argument("--agent", "-a", required=True, choices=VALID_AGENTS, help="Agent type")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--timesteps", "-t", type=int, default=None, help="Timesteps per trial (default: config.total_timesteps)")
    parser.add_argument("--base-seed", type=int, default=42, help="First training seed")
    parser.add_argument("--num-seeds", "-ns", type=int, default=3, help="Seeds per trial")
    parser.add_argument("--study-name", default=None, help="Optuna study name (default: agent name)")
    args = parser.parse_args()

    timesteps = args.timesteps or Config().total_timesteps
    study_name = args.study_name or args.agent
    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))

    print(f"Sweep: agent={args.agent}, "
          f"seeds={seeds}, timesteps={timesteps}, n_trials={args.n_trials}")

    study = create_optuna_study(study_name)

    study.optimize(
        lambda trial: objective(trial, args.agent, timesteps, seeds),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    best = study.best_trial
    print(f"Best trial: #{best.number}")
    print(f"Best mean reward: {best.value:.4f}")

    seed_rewards = {k: v for k, v in best.user_attrs.items() if k.startswith("reward_seed_")}
    if seed_rewards:
        vals = list(seed_rewards.values())
        print(f"Per-seed rewards: {[f'{v:.4f}' for v in vals]}  (std={np.std(vals, ddof=1):.4f})")

    print(f"\nBest hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    _print_config_overrides(best.params, args.agent)


if __name__ == "__main__":
    main()
