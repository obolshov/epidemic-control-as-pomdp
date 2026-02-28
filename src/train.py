from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from sb3_contrib import RecurrentPPO

from src.config import Config
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.utils import plot_learning_curve
from src.wrappers import create_environment


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule for SB3.

    Args:
        initial_value: Initial learning rate.

    Returns:
        Schedule function: progress_remaining -> current lr.
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(
    config: Config,
    pomdp_params: Dict[str, Any],
    seed: int,
    rank: int,
) -> Callable[[], gym.Env]:
    """Factory function for creating environments with unique seeds.

    Args:
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        seed: Base random seed.
        rank: Environment index (added to seed for uniqueness).

    Returns:
        Callable that creates a seeded environment.
    """

    def _init() -> gym.Env:
        env = create_environment(config, pomdp_params, seed=seed + rank)
        env.reset(seed=seed + rank)
        return env

    return _init


def create_vec_env(
    config: Config,
    pomdp_params: Dict[str, Any],
    seed: int,
    n_envs: int,
    monitor_dir: Path,
    agent_name: str,
) -> VecEnv:
    """Create a vectorized environment with VecNormalize.

    Pipeline: DummyVecEnv(n_envs) -> VecMonitor -> VecNormalize -> [VecFrameStack]

    Args:
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        seed: Base random seed.
        n_envs: Number of parallel environments.
        monitor_dir: Directory for monitor logs.
        agent_name: Agent name (determines if VecFrameStack is applied).

    Returns:
        Vectorized environment with normalization (and optional frame stacking).
    """
    env = DummyVecEnv(
        [make_env(config, pomdp_params, seed, rank=i) for i in range(n_envs)]
    )
    env = VecMonitor(env, filename=str(monitor_dir))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    if agent_name == "ppo_framestack":
        env = VecFrameStack(env, n_stack=config.n_stack)
        print(f"Applied VecFrameStack with n_stack={config.n_stack}")

    print(f"Observation space: {env.observation_space}")
    return env


def create_eval_env(
    config: Config,
    pomdp_params: Dict[str, Any],
    seed: int,
    agent_name: str,
) -> VecEnv:
    """Create a separate evaluation environment with its own VecNormalize.

    EvalCallback will sync normalization stats from training env automatically.

    Args:
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        seed: Random seed (offset from training seed).
        agent_name: Agent name (determines if VecFrameStack is applied).

    Returns:
        Evaluation VecEnv with VecNormalize (stats synced during training).
    """
    env = DummyVecEnv([make_env(config, pomdp_params, seed + 1000, rank=0)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)

    if agent_name == "ppo_framestack":
        env = VecFrameStack(env, n_stack=config.n_stack)

    return env


def create_training_callbacks(
    eval_env: VecEnv,
    experiment_dir: "ExperimentDirectory",
    agent_name: str,
    seed: int,
    n_envs: int = 1,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10,
    patience: int = 10,
    min_evals: int = 5,
) -> CallbackList:
    """Create callback stack for training with early stopping.

    Args:
        eval_env: Separate evaluation environment.
        experiment_dir: Experiment directory for saving outputs.
        agent_name: Agent name for file naming.
        seed: Training seed for file naming.
        n_envs: Number of parallel training envs (eval_freq is divided by this).
        eval_freq: Evaluate every N total timesteps (adjusted for n_envs internally).
        n_eval_episodes: Episodes per evaluation.
        patience: Stop after N evals without improvement.
        min_evals: Minimum evals before early stopping can trigger.

    Returns:
        Composed callback list.
    """
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        min_evals=min_evals,
        verbose=1,
    )

    # best_model is saved per seed
    best_model_dir = experiment_dir.weights_dir / f"best_{agent_name}_seed{seed}"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = experiment_dir.logs_dir / f"{agent_name}_seed{seed}_eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    # EvalCallback's eval_freq counts per-environment steps, so divide by n_envs
    adjusted_eval_freq = max(1, eval_freq // n_envs)

    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=stop_callback,
        n_eval_episodes=n_eval_episodes,
        eval_freq=adjusted_eval_freq,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        deterministic=True,
        verbose=1,
    )

    return CallbackList([eval_callback])


def train_ppo_agent(
    config: Config,
    experiment_dir: "ExperimentDirectory",
    agent_name: str,
    total_timesteps: int,
    seed: int = 42,
    pomdp_params: Optional[Dict[str, Any]] = None,
) -> Union[PPO, RecurrentPPO]:
    """Train a PPO agent with VecNormalize, callbacks, and early stopping.

    Args:
        config: Configuration for the environment and RL hyperparameters.
        experiment_dir: ExperimentDirectory for saving outputs.
        agent_name: Name of the agent (e.g., "ppo_baseline", "ppo_framestack", "ppo_recurrent").
        total_timesteps: Maximum number of timesteps to train.
        seed: Random seed for reproducibility.
        pomdp_params: POMDP parameters for applying wrappers.

    Returns:
        Trained PPO or RecurrentPPO model (best checkpoint).
    """
    if pomdp_params is None:
        pomdp_params = {}

    # Create training environment (per-seed monitor dir to avoid mixing monitor.csv data)
    monitor_dir = experiment_dir.tensorboard_dir / f"{agent_name}_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = create_vec_env(
        config, pomdp_params, seed, config.n_envs, monitor_dir, agent_name
    )

    # Create separate eval environment
    eval_env = create_eval_env(config, pomdp_params, seed, agent_name)

    # Create callbacks
    callbacks = create_training_callbacks(
        eval_env, experiment_dir, agent_name, seed,
        n_envs=config.n_envs,
        n_eval_episodes=config.n_eval_episodes,
        patience=config.early_stop_patience,
        min_evals=config.early_stop_min_evals,
    )

    # Create model
    if agent_name == "ppo_recurrent":
        print(f"Using RecurrentPPO with MlpLstmPolicy")
        print(
            f"LSTM config: hidden_size={config.lstm_hidden_size}, "
            f"n_layers={config.n_lstm_layers}"
        )
        print(
            f"Training config: n_steps={config.recurrent_n_steps}, "
            f"batch_size={config.recurrent_batch_size}, "
            f"n_epochs={config.recurrent_n_epochs}, "
            f"ent_coef={config.ent_coef}"
        )

        policy_kwargs = {
            "lstm_hidden_size": config.lstm_hidden_size,
            "n_lstm_layers": config.n_lstm_layers,
        }

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            seed=seed,
            policy_kwargs=policy_kwargs,
            n_steps=config.recurrent_n_steps,
            batch_size=config.recurrent_batch_size,
            n_epochs=config.recurrent_n_epochs,
            ent_coef=config.ent_coef,
            learning_rate=linear_schedule(3e-4),
            tensorboard_log=str(experiment_dir.tensorboard_dir),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            ent_coef=config.ent_coef,
            learning_rate=linear_schedule(3e-4),
            tensorboard_log=str(experiment_dir.tensorboard_dir),
        )

    print(f"Training {agent_name} (seed={seed}) for up to {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{agent_name}_seed{seed}",
        progress_bar=False,
    )
    print("Training finished.")

    # Save VecNormalize statistics
    vecnorm_path = experiment_dir.weights_dir / f"{agent_name}_seed{seed}_vecnormalize.pkl"
    # Find the VecNormalize layer in the env stack
    vec_normalize = _find_vec_normalize(env)
    if vec_normalize is not None:
        vec_normalize.save(str(vecnorm_path))
        print(f"VecNormalize stats saved to {vecnorm_path}")

    # Load best model from EvalCallback (if it saved one)
    best_model_dir = experiment_dir.weights_dir / f"best_{agent_name}_seed{seed}"
    best_model_path = best_model_dir / "best_model.zip"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        if agent_name == "ppo_recurrent":
            model = RecurrentPPO.load(str(best_model_path))
        else:
            model = PPO.load(str(best_model_path))

    # Save final/best model to standard weight path
    weight_path = experiment_dir.get_weight_path(agent_name, seed)
    model.save(str(weight_path.with_suffix("")))
    print(f"Model weights saved to {weight_path}")

    # Cleanup
    eval_env.close()
    env.close()

    return model


def prepare_rl_agents(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    agents_to_skip: set,
) -> Dict[str, List[Union[PPO, RecurrentPPO]]]:
    """Train or load RL agents across multiple seeds.

    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory for saving/loading.
        agents_to_skip: Set of agent names to skip training (or {"all"} to skip all).

    Returns:
        Dict mapping agent_name -> list of models (one per seed).
    """
    rl_agent_names = [
        name for name in exp_config.target_agents
        if name.startswith("ppo_")
    ]

    if not rl_agent_names:
        return {}

    skip_all = "all" in agents_to_skip

    print("\n" + "=" * 80)
    print("PREPARING RL AGENTS")
    print("=" * 80)

    models_by_agent: Dict[str, List[Union[PPO, RecurrentPPO]]] = {}

    for agent_name in rl_agent_names:
        should_skip = skip_all or agent_name in agents_to_skip
        models = []

        for seed in exp_config.training_seeds:
            if should_skip:
                weight_path = experiment_dir.get_weight_path(agent_name, seed)
                if weight_path.exists():
                    print(f"\nLoading {agent_name} (seed={seed}) from {weight_path}...")
                    if agent_name == "ppo_recurrent":
                        model = RecurrentPPO.load(str(weight_path.with_suffix("")))
                    else:
                        model = PPO.load(str(weight_path.with_suffix("")))
                    models.append(model)
                else:
                    raise FileNotFoundError(
                        f"Weights for {agent_name} (seed={seed}) not found at {weight_path}"
                    )
            else:
                print(f"\nTraining {agent_name} (seed={seed})...")
                model = train_ppo_agent(
                    config=exp_config.base_config,
                    experiment_dir=experiment_dir,
                    agent_name=agent_name,
                    total_timesteps=exp_config.total_timesteps,
                    seed=seed,
                    pomdp_params=exp_config.pomdp_params,
                )
                models.append(model)

                monitor_dir = experiment_dir.tensorboard_dir / f"{agent_name}_seed{seed}"
                if monitor_dir.exists():
                    plot_path = experiment_dir.get_plot_path(
                        f"{agent_name}_seed{seed}_learning.png"
                    )
                    plot_learning_curve(
                        log_folder=str(monitor_dir),
                        title=f"{agent_name} (seed={seed}) Learning Curve",
                        save_path=str(plot_path),
                    )

        models_by_agent[agent_name] = models

    return models_by_agent


def _find_vec_normalize(env: VecEnv) -> Optional[VecNormalize]:
    """Find VecNormalize layer in a wrapped VecEnv stack.

    Args:
        env: Potentially wrapped vectorized environment.

    Returns:
        VecNormalize instance if found, else None.
    """
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    return None
