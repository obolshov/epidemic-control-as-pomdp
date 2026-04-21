import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from sb3_contrib import RecurrentPPO

from src.callbacks import StopTrainingOnNoModelImprovementWithDelta
from src.config import Config
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.utils import plot_learning_curve
from src.wrappers import create_environment


def _load_model(path: str, agent_name: str) -> Union[PPO, RecurrentPPO]:
    """Load a saved PPO or RecurrentPPO model based on agent name.

    Args:
        path: Path to model file (.zip extension optional).
        agent_name: Agent name (determines model class).

    Returns:
        Loaded model.
    """
    cls = RecurrentPPO if agent_name.startswith("ppo_recurrent") else PPO
    return cls.load(path)


def _read_source_timesteps(scenario_dir: Path) -> Optional[int]:
    """Read total_timesteps from the first config.json found in a scenario directory.

    Args:
        scenario_dir: Path to the scenario directory (e.g. experiments/pomdp_t500000/).

    Returns:
        The total_timesteps value, or None if no config.json found.
    """
    for config_path in scenario_dir.glob("*/config.json"):
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            return cfg.get("base_config", {}).get("total_timesteps", cfg.get("total_timesteps"))
        except (json.JSONDecodeError, KeyError):
            continue
    return None


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
    vecnormalize_path: Optional[Path] = None,
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
        vecnormalize_path: Path to saved VecNormalize stats to load (for resume training).
            If provided and exists, loads stats instead of creating fresh VecNormalize.

    Returns:
        Vectorized environment with normalization (and optional frame stacking).
    """
    env = DummyVecEnv(
        [make_env(config, pomdp_params, seed, rank=i) for i in range(n_envs)]
    )
    env = VecMonitor(env, filename=str(monitor_dir))

    if vecnormalize_path is not None and vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = True
        env.norm_reward = True
        print(f"Loaded VecNormalize stats from {vecnormalize_path}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    if agent_name.startswith("ppo_framestack"):
        env = VecFrameStack(env, n_stack=config.n_stack)
        print(f"Applied VecFrameStack with n_stack={config.n_stack}")

    print(f"Observation space: {env.observation_space}")
    return env


def create_eval_env(
    config: Config,
    pomdp_params: Dict[str, Any],
    seed: int,
    agent_name: str,
    vecnormalize_path: Optional[Path] = None,
) -> VecEnv:
    """Create a separate evaluation environment with its own VecNormalize.

    EvalCallback will sync normalization stats from training env automatically.

    Args:
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        seed: Random seed (offset from training seed).
        agent_name: Agent name (determines if VecFrameStack is applied).
        vecnormalize_path: Path to saved VecNormalize stats to load (for resume training).
            If provided and exists, loads stats with training=False, norm_reward=False.

    Returns:
        Evaluation VecEnv with VecNormalize (stats synced during training).
    """
    env = DummyVecEnv([make_env(config, pomdp_params, seed + 1000, rank=0)])
    env = VecMonitor(env)

    if vecnormalize_path is not None and vecnormalize_path.exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)

    if agent_name.startswith("ppo_framestack"):
        env = VecFrameStack(env, n_stack=config.n_stack)

    return env


def create_training_callbacks(
    eval_env: VecEnv,
    experiment_dir: "ExperimentDirectory",
    agent_name: str,
    seed: int,
    *,
    n_envs: int,
    eval_freq: int,
    n_eval_episodes: int,
    patience: int,
    min_evals: int,
    min_delta: float,
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
        patience: Stop after N evals without significant improvement.
        min_evals: Minimum evals before early stopping can trigger.
        min_delta: Minimum reward delta to count as a significant improvement.

    Returns:
        Composed callback list.
    """
    stop_callback = StopTrainingOnNoModelImprovementWithDelta(
        max_no_improvement_evals=patience,
        min_evals=min_evals,
        min_delta=min_delta,
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
    resume_from_weights_dir: Optional[Path] = None,
) -> Tuple[Union[PPO, RecurrentPPO], int]:
    """Train a PPO agent with VecNormalize, callbacks, and early stopping.

    Args:
        config: Configuration for the environment and RL hyperparameters.
        experiment_dir: ExperimentDirectory for saving outputs.
        agent_name: Name of the agent (e.g., "ppo_baseline", "ppo_framestack", "ppo_recurrent").
        total_timesteps: Maximum number of timesteps to train.
        seed: Random seed for reproducibility.
        pomdp_params: POMDP parameters for applying wrappers.
        resume_from_weights_dir: Path to existing weights directory to resume training from.
            If provided and weights exist, loads model + VecNormalize and continues training.

    Returns:
        Tuple of (trained model, initial_num_timesteps). initial_num_timesteps is 0 for
        fresh training or the loaded model's timestep count for resumed training.
    """
    if pomdp_params is None:
        pomdp_params = {}

    # Check if we can resume from existing weights
    resuming = False
    source_weight_path = None
    source_vecnorm_path = None
    if resume_from_weights_dir is not None:
        source_weight_path = resume_from_weights_dir / f"{agent_name}_seed{seed}.zip"
        source_vecnorm_path = resume_from_weights_dir / f"{agent_name}_seed{seed}_vecnormalize.pkl"
        if source_weight_path.exists():
            resuming = True
            print(f"Resuming {agent_name} (seed={seed}) from {source_weight_path}")
        else:
            print(f"No weights found for {agent_name} (seed={seed}) at {source_weight_path}, training from scratch")

    # Create training environment (per-seed monitor dir to avoid mixing monitor.csv data)
    monitor_dir = experiment_dir.tensorboard_dir / f"{agent_name}_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = create_vec_env(
        config, pomdp_params, seed, config.n_envs, monitor_dir, agent_name,
        vecnormalize_path=source_vecnorm_path if resuming else None,
    )

    # Create separate eval environment
    eval_env = create_eval_env(
        config, pomdp_params, seed, agent_name,
        vecnormalize_path=source_vecnorm_path if resuming else None,
    )

    # Create callbacks (recurrent agents use separate early-stop params)
    is_recurrent = agent_name.startswith("ppo_recurrent")
    callbacks = create_training_callbacks(
        eval_env, experiment_dir, agent_name, seed,
        n_envs=config.n_envs,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        patience=config.recurrent_early_stop_patience if is_recurrent else config.early_stop_patience,
        min_evals=config.recurrent_early_stop_min_evals if is_recurrent else config.early_stop_min_evals,
        min_delta=config.early_stop_min_delta,
    )

    # Create or load model
    initial_num_timesteps = 0
    if resuming:
        is_recurrent = agent_name.startswith("ppo_recurrent")
        cls = RecurrentPPO if is_recurrent else PPO
        model = cls.load(
            str(source_weight_path),
            env=env,
            custom_objects={"learning_rate": linear_schedule(config.learning_rate)},
            tensorboard_log=str(experiment_dir.tensorboard_dir),
        )

        initial_num_timesteps = _read_source_timesteps(resume_from_weights_dir.parent)
        if initial_num_timesteps is None:
            initial_num_timesteps = model.num_timesteps
        model.num_timesteps = initial_num_timesteps
        print(f"Resuming from {initial_num_timesteps} timesteps, continuing to {total_timesteps}")
    elif agent_name.startswith("ppo_recurrent"):
        print(f"Using RecurrentPPO with MlpLstmPolicy")
        print(f"LSTM config: hidden_size={config.lstm_hidden_size}")
        print(
            f"Training config: n_steps={config.recurrent_n_steps}, "
            f"batch_size={config.recurrent_batch_size}, "
            f"n_epochs={config.recurrent_n_epochs}, "
            f"ent_coef={config.ent_coef}, "
            f"lr={config.learning_rate}"
        )

        policy_kwargs = {
            "lstm_hidden_size": config.lstm_hidden_size,
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
            learning_rate=linear_schedule(config.learning_rate),
            tensorboard_log=str(experiment_dir.tensorboard_dir),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            ent_coef=config.ent_coef,
            learning_rate=linear_schedule(config.learning_rate),
            tensorboard_log=str(experiment_dir.tensorboard_dir),
        )

    print(f"Training {agent_name} (seed={seed}) for up to {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{agent_name}_seed{seed}",
        progress_bar=False,
        reset_num_timesteps=not resuming,
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
        model = _load_model(str(best_model_path), agent_name)

    # Save final/best model to standard weight path
    weight_path = experiment_dir.get_weight_path(agent_name, seed)
    model.save(str(weight_path))
    print(f"Model weights saved to {weight_path}")

    # Cleanup
    eval_env.close()
    env.close()

    return model, initial_num_timesteps


def prepare_rl_agents(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    agents_to_skip: set,
    resume_from_weights_dir: Optional[Path] = None,
) -> Dict[str, List[Union[PPO, RecurrentPPO]]]:
    """Train or load RL agents across multiple seeds.

    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory for saving/loading.
        agents_to_skip: Set of agent names to skip training (or {"all"} to skip all).
        resume_from_weights_dir: Path to existing weights directory to resume training from.
            Agents with matching weights will be resumed; others train from scratch.

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
        should_skip = skip_all or any(
            agent_name == s or agent_name.startswith(s + "_") for s in agents_to_skip
        )
        models = []

        for seed in exp_config.training_seeds:
            if should_skip:
                weight_path = experiment_dir.get_weight_path(agent_name, seed)
                if weight_path.exists():
                    print(f"\nLoading {agent_name} (seed={seed}) from {weight_path}...")
                    model = _load_model(str(weight_path), agent_name)
                    models.append(model)
                else:
                    raise FileNotFoundError(
                        f"Weights for {agent_name} (seed={seed}) not found at {weight_path}"
                    )
            else:
                print(f"\nTraining {agent_name} (seed={seed})...")
                model, initial_timesteps = train_ppo_agent(
                    config=exp_config.base_config,
                    experiment_dir=experiment_dir,
                    agent_name=agent_name,
                    total_timesteps=exp_config.total_timesteps,
                    seed=seed,
                    pomdp_params=exp_config.pomdp_params,
                    resume_from_weights_dir=resume_from_weights_dir,
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
                        timestep_offset=initial_timesteps,
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
