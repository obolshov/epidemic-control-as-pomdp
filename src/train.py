import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
from stable_baselines3 import DQN, PPO
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

from src.callbacks import SaveVecNormalizeOnBestCallback, StopTrainingOnNoModelImprovementWithDelta
from src.config import Config
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.scenarios import is_off_policy, is_rl_agent
from src.utils import plot_learning_curve
from src.wrappers import FixedNormalizeWrapper, create_environment


def linear_schedule(
    initial_value: float,
    total_timesteps: int,
    lr_decay_steps: int,
) -> Callable[[float], float]:
    """Linear learning rate schedule decoupled from training budget.

    LR decays linearly from initial_value to 10% of initial_value over
    lr_decay_steps, then holds at the floor. This makes the LR at any
    given training step independent of total_timesteps.

    Args:
        initial_value: Peak learning rate.
        total_timesteps: Training budget passed to model.learn() — used only
            to convert SB3's progress_remaining back to absolute steps.
        lr_decay_steps: Number of steps over which LR decays to the floor.
    """
    min_lr = initial_value * 0.1

    def func(progress_remaining: float) -> float:
        current_step = (1.0 - progress_remaining) * total_timesteps
        if current_step >= lr_decay_steps:
            return min_lr
        return min_lr + (initial_value - min_lr) * (1.0 - current_step / lr_decay_steps)

    return func


def _load_model(path: str, agent_name: str) -> Union[DQN, PPO, RecurrentPPO]:
    """Load a saved model based on agent name.

    Args:
        path: Path to model file (.zip extension optional).
        agent_name: Agent name (determines model class).

    Returns:
        Loaded model.
    """
    if agent_name.startswith("dqn"):
        return DQN.load(path)
    if agent_name.startswith("ppo_recurrent"):
        return RecurrentPPO.load(path)
    return PPO.load(path)





def make_env(
    config: Config,
    pomdp_params: Dict[str, Any],
    seed: int,
    rank: int,
    fixed_normalize: bool = False,
) -> Callable[[], gym.Env]:
    """Factory function for creating environments with unique seeds.

    Args:
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        seed: Base random seed.
        rank: Environment index (added to seed for uniqueness).
        fixed_normalize: If True, apply FixedNormalizeWrapper (for off-policy algorithms).

    Returns:
        Callable that creates a seeded environment.
    """

    def _init() -> gym.Env:
        env = create_environment(config, pomdp_params, seed=seed + rank)
        if fixed_normalize:
            env = FixedNormalizeWrapper(env)
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
    is_dqn = is_off_policy(agent_name)
    env = DummyVecEnv(
        [make_env(config, pomdp_params, seed, rank=i, fixed_normalize=is_dqn)
         for i in range(n_envs)]
    )
    env = VecMonitor(env, filename=str(monitor_dir))
    env = VecNormalize(env, norm_obs=not is_dqn, norm_reward=not is_dqn, gamma=0.99)

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
    is_dqn = is_off_policy(agent_name)
    env = DummyVecEnv([make_env(config, pomdp_params, seed + 1000, rank=0, fixed_normalize=is_dqn)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=not is_dqn, norm_reward=False, gamma=0.99)

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

    save_vecnorm_callback = SaveVecNormalizeOnBestCallback(
        save_path=str(best_model_dir / "vecnormalize.pkl"),
    )

    # EvalCallback's eval_freq counts per-environment steps, so divide by n_envs
    adjusted_eval_freq = max(1, eval_freq // n_envs)

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=save_vecnorm_callback,
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
        Trained model.
    """
    if pomdp_params is None:
        pomdp_params = {}

    # Create training environment (per-seed monitor dir to avoid mixing monitor.csv data)
    monitor_dir = experiment_dir.tensorboard_dir / f"{agent_name}_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = create_vec_env(
        config, pomdp_params, seed, config.n_envs, monitor_dir, agent_name,
    )

    # Create separate eval environment
    eval_env = create_eval_env(
        config, pomdp_params, seed, agent_name,
    )

    # Create callbacks (each agent type has its own early-stop params)
    is_recurrent = agent_name.startswith("ppo_recurrent")
    if is_recurrent:
        agent_config = config.ppo_recurrent
    elif agent_name.startswith("ppo_framestack"):
        agent_config = config.ppo_framestack
    else:
        agent_config = config.ppo_baseline
    callbacks = create_training_callbacks(
        eval_env, experiment_dir, agent_name, seed,
        n_envs=config.n_envs,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        patience=agent_config.early_stop_patience,
        min_evals=agent_config.early_stop_min_evals,
        min_delta=config.early_stop_min_delta,
    )

    # Create model
    if is_recurrent:
        rc = config.ppo_recurrent
        print(f"Using RecurrentPPO with MlpLstmPolicy")
        print(f"LSTM config: hidden_size={rc.lstm_hidden_size}")
        print(
            f"Training config: n_steps={rc.n_steps}, "
            f"batch_size={rc.batch_size}, "
            f"n_epochs={rc.n_epochs}, "
            f"ent_coef={rc.ent_coef}, "
            f"lr={rc.learning_rate}"
        )

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            seed=seed,
            policy_kwargs={"lstm_hidden_size": rc.lstm_hidden_size},
            n_steps=rc.n_steps,
            batch_size=rc.batch_size,
            n_epochs=rc.n_epochs,
            gamma=rc.gamma,
            ent_coef=rc.ent_coef,
            learning_rate=linear_schedule(rc.learning_rate, total_timesteps, config.lr_decay_steps),
            tensorboard_log=str(experiment_dir.tensorboard_dir),
        )
    else:
        ac = agent_config
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            policy_kwargs={"net_arch": ac.net_arch},
            n_steps=ac.n_steps,
            batch_size=ac.batch_size,
            gamma=ac.gamma,
            ent_coef=ac.ent_coef,
            learning_rate=linear_schedule(ac.learning_rate, total_timesteps, config.lr_decay_steps),
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

    vecnorm_path = experiment_dir.weights_dir / f"{agent_name}_seed{seed}_vecnormalize.pkl"
    best_model_dir = experiment_dir.weights_dir / f"best_{agent_name}_seed{seed}"
    best_model_path = best_model_dir / "best_model.zip"
    best_vecnorm_path = best_model_dir / "vecnormalize.pkl"

    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model = _load_model(str(best_model_path), agent_name)
        if best_vecnorm_path.exists():
            shutil.copy(str(best_vecnorm_path), str(vecnorm_path))
            print(f"VecNormalize stats saved to {vecnorm_path} (from best checkpoint)")
        else:
            vec_normalize = _find_vec_normalize(env)
            if vec_normalize is not None:
                vec_normalize.save(str(vecnorm_path))
                print(f"VecNormalize stats saved to {vecnorm_path} (end-of-training fallback)")
    else:
        vec_normalize = _find_vec_normalize(env)
        if vec_normalize is not None:
            vec_normalize.save(str(vecnorm_path))
            print(f"VecNormalize stats saved to {vecnorm_path}")

    weight_path = experiment_dir.get_weight_path(agent_name, seed)
    model.save(str(weight_path))
    print(f"Model weights saved to {weight_path}")

    eval_env.close()
    env.close()

    return model


def train_dqn_agent(
    config: Config,
    experiment_dir: "ExperimentDirectory",
    agent_name: str,
    total_timesteps: int,
    seed: int = 42,
    pomdp_params: Optional[Dict[str, Any]] = None,
) -> DQN:
    """Train a DQN agent with VecNormalize, callbacks, and early stopping.

    Uses n_envs=1 (standard for off-policy). No VecFrameStack.

    Args:
        config: Configuration for the environment and DQN hyperparameters.
        experiment_dir: ExperimentDirectory for saving outputs.
        agent_name: Name of the agent (e.g., "dqn").
        total_timesteps: Maximum number of timesteps to train.
        seed: Random seed for reproducibility.
        pomdp_params: POMDP parameters for applying wrappers.

    Returns:
        Trained DQN model.
    """
    if pomdp_params is None:
        pomdp_params = {}

    n_envs = 1

    monitor_dir = experiment_dir.tensorboard_dir / f"{agent_name}_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = create_vec_env(
        config, pomdp_params, seed, n_envs, monitor_dir, agent_name,
    )

    eval_env = create_eval_env(
        config, pomdp_params, seed, agent_name,
    )

    dqn = config.dqn
    callbacks = create_training_callbacks(
        eval_env, experiment_dir, agent_name, seed,
        n_envs=n_envs,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        patience=dqn.early_stop_patience,
        min_evals=dqn.early_stop_min_evals,
        min_delta=config.early_stop_min_delta,
    )

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
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
        tensorboard_log=str(experiment_dir.tensorboard_dir),
    )

    print(f"Training {agent_name} (seed={seed}) for up to {total_timesteps} timesteps...")
    print(
        f"DQN config: net_arch={dqn.net_arch}, "
        f"tau={dqn.tau}, "
        f"buffer_size={dqn.buffer_size}, "
        f"learning_starts={dqn.learning_starts}, "
        f"batch_size={dqn.batch_size}, "
        f"exploration_fraction={dqn.exploration_fraction}, "
        f"lr={dqn.learning_rate}"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{agent_name}_seed{seed}",
        progress_bar=False,
    )
    print("Training finished.")

    vecnorm_path = experiment_dir.weights_dir / f"{agent_name}_seed{seed}_vecnormalize.pkl"
    best_model_dir = experiment_dir.weights_dir / f"best_{agent_name}_seed{seed}"
    best_model_path = best_model_dir / "best_model.zip"
    best_vecnorm_path = best_model_dir / "vecnormalize.pkl"

    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model = _load_model(str(best_model_path), agent_name)
        if best_vecnorm_path.exists():
            shutil.copy(str(best_vecnorm_path), str(vecnorm_path))
            print(f"VecNormalize stats saved to {vecnorm_path} (from best checkpoint)")
        else:
            vec_normalize = _find_vec_normalize(env)
            if vec_normalize is not None:
                vec_normalize.save(str(vecnorm_path))
                print(f"VecNormalize stats saved to {vecnorm_path} (end-of-training fallback)")
    else:
        vec_normalize = _find_vec_normalize(env)
        if vec_normalize is not None:
            vec_normalize.save(str(vecnorm_path))
            print(f"VecNormalize stats saved to {vecnorm_path}")

    weight_path = experiment_dir.get_weight_path(agent_name, seed)
    model.save(str(weight_path))
    print(f"Model weights saved to {weight_path}")

    eval_env.close()
    env.close()

    return model


def prepare_rl_agents(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    agents_to_skip: set,
) -> Dict[str, List[Union[DQN, PPO, RecurrentPPO]]]:
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
        if is_rl_agent(name)
    ]

    if not rl_agent_names:
        return {}

    skip_all = "all" in agents_to_skip

    print("\n" + "=" * 80)
    print("PREPARING RL AGENTS")
    print("=" * 80)

    models_by_agent: Dict[str, List[Union[DQN, PPO, RecurrentPPO]]] = {}

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
                    print(
                        f"WARNING: Weights not found for {agent_name} (seed={seed}), "
                        f"skipping remaining seeds."
                    )
                    break
            else:
                print(f"\nTraining {agent_name} (seed={seed})...")
                train_fn = train_dqn_agent if agent_name.startswith("dqn") else train_ppo_agent
                model = train_fn(
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

        if models:
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
