"""
Evaluation module for epidemic control agents.

Provides:
- run_agent(): Single-episode trajectory evaluation for SEIR plots.
- evaluate_agent_statistics(): Multi-episode reward statistics using SB3's evaluate_policy.
- evaluate_multi_seed(): Aggregate statistics across multiple independently-trained models.
"""

import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional, Union

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecMonitor, VecNormalize
from sb3_contrib import RecurrentPPO

from src.agents import Agent, create_baseline_agents
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.config import Config
from src.env import EpidemicEnv, SimulationResult
from src.utils import log_results, plot_single_result
from src.wrappers import create_environment


def create_eval_vec_env(
    config: Config,
    pomdp_params: Dict[str, Any],
    agent_name: str,
    vecnorm_path: Optional[str] = None,
    seed: int = 0,
) -> VecEnv:
    """Create evaluation VecEnv with frozen VecNormalize stats.

    Args:
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        agent_name: Agent name (determines if VecFrameStack is applied).
        vecnorm_path: Path to saved VecNormalize stats. If None, no normalization.
        seed: Random seed for the environment.

    Returns:
        Evaluation VecEnv ready for inference.
    """
    env = create_environment(config, pomdp_params, seed=seed)
    env.reset(seed=seed)
    venv = DummyVecEnv([lambda: env])
    venv = VecMonitor(venv)

    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)

    if agent_name == "ppo_framestack":
        venv = VecFrameStack(venv, n_stack=config.n_stack)

    return venv


def evaluate_agent_statistics(
    model: Union[PPO, RecurrentPPO],
    eval_env: VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate agent over multiple episodes and return reward statistics.

    Args:
        model: Trained RL model.
        eval_env: Evaluation environment (with frozen VecNormalize).
        n_eval_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.

    Returns:
        Dict with mean_reward, std_reward, ci_low, ci_high, episode_rewards, episode_lengths.
    """
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        return_episode_rewards=True,
    )

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    n = len(episode_rewards)
    ci_margin = 1.96 * std_reward / np.sqrt(n) if n > 1 else 0.0

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "ci_low": mean_reward - ci_margin,
        "ci_high": mean_reward + ci_margin,
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(el) for el in episode_lengths],
    }


def evaluate_multi_seed(
    models: List[Union[PPO, RecurrentPPO]],
    seeds: List[int],
    config: Config,
    pomdp_params: Dict[str, Any],
    agent_name: str,
    experiment_dir: "ExperimentDirectory",
    n_eval_episodes_per_seed: int = 10,
) -> Dict[str, Any]:
    """Evaluate multiple independently-trained models and aggregate statistics.

    Args:
        models: List of trained models (one per seed).
        seeds: List of training seeds corresponding to models.
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        agent_name: Agent name.
        experiment_dir: Experiment directory for VecNormalize paths.
        n_eval_episodes_per_seed: Episodes per seed evaluation.

    Returns:
        Dict with overall_mean, overall_std, ci_low, ci_high, per_seed stats, best_seed_idx.
    """
    all_rewards = []
    per_seed_stats = []

    for model, seed in zip(models, seeds):
        vecnorm_path = str(experiment_dir.get_vecnormalize_path(agent_name, seed))
        eval_env = create_eval_vec_env(
            config, pomdp_params, agent_name, vecnorm_path, seed=seed
        )

        stats = evaluate_agent_statistics(
            model, eval_env, n_eval_episodes_per_seed
        )
        stats["seed"] = seed
        per_seed_stats.append(stats)
        all_rewards.extend(stats["episode_rewards"])

        eval_env.close()

    overall_mean = float(np.mean(all_rewards))
    overall_std = float(np.std(all_rewards))
    n = len(all_rewards)
    ci_margin = 1.96 * overall_std / np.sqrt(n) if n > 1 else 0.0

    best_seed_idx = int(np.argmax([s["mean_reward"] for s in per_seed_stats]))

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "ci_low": overall_mean - ci_margin,
        "ci_high": overall_mean + ci_margin,
        "per_seed": per_seed_stats,
        "best_seed_idx": best_seed_idx,
        "best_seed": seeds[best_seed_idx],
    }


def run_agent(
    agent: Agent,
    env: EpidemicEnv,
    experiment_dir: Optional["ExperimentDirectory"] = None,
    agent_name: Optional[str] = None,
) -> SimulationResult:
    """
    Runs a simulation for a single agent (single-episode trajectory).

    Used for generating SEIR trajectory plots. For statistical evaluation,
    use evaluate_agent_statistics() or evaluate_multi_seed() instead.

    Args:
        agent: The agent to evaluate.
        env: The environment to run the simulation in (can be VecEnv or regular Env).
        experiment_dir: ExperimentDirectory for saving outputs. If None, saves to default locations.
        agent_name: Override name for saving files. If None, uses agent's class name.

    Returns:
        A SimulationResult object containing the results of the simulation.
    """
    # Check if this is a VecEnv (used for framestack)
    is_vec_env = isinstance(env, VecEnv)

    obs = env.reset()

    # VecEnv returns obs without info dict in reset
    if is_vec_env:
        obs_flat = obs[0] if len(obs.shape) > 1 else obs
    else:
        obs, _ = obs if isinstance(obs, tuple) else (obs, {})
        obs_flat = obs

    done = False

    # Get unwrapped environment to access true state
    if is_vec_env:
        unwrapped_env = env.envs[0]
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
    else:
        unwrapped_env = env.unwrapped

    # Get initial state from unwrapped env
    if hasattr(unwrapped_env, 'current_state'):
        state = unwrapped_env.current_state
        S_init = state.S
        E_init = state.E
        I_init = state.I
        R_init = state.R
    else:
        if len(obs_flat) == 3:
            S_init, I_init, R_init = obs_flat[-3:]
            E_init = 0.0
        elif len(obs_flat) == 4:
            S_init, E_init, I_init, R_init = obs_flat[-4:]
        else:
            if len(obs_flat) % 4 == 0:
                S_init, E_init, I_init, R_init = obs_flat[-4:]
            else:
                S_init, I_init, R_init = obs_flat[-3:]
                E_init = 0.0

    all_S = [S_init]
    all_E = [E_init]
    all_I = [I_init]
    all_R = [R_init]

    actions_taken = []
    timesteps = []
    rewards = []
    observations = []

    current_timestep = 0

    # Initialize state for recurrent policies (LSTM)
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        observations.append(obs_flat if is_vec_env else obs)
        timesteps.append(current_timestep)

        action_idx, lstm_state = agent.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
        if is_vec_env:
            action_idx = int(action_idx[0]) if hasattr(action_idx, '__len__') else int(action_idx)

        episode_start = np.zeros((1,), dtype=bool)

        if is_vec_env:
            obs, reward, done_array, info_array = env.step([action_idx])
            obs_flat = obs[0]
            reward = float(reward[0])
            done = bool(done_array[0])
            info = info_array[0]
        else:
            obs, reward, done, truncated, info = env.step(action_idx)

        S = info.get("S", [])
        E = info.get("E", [])
        I = info.get("I", [])
        R = info.get("R", [])

        if len(S) > 0:
            all_S.extend(S)
            all_E.extend(E)
            all_I.extend(I)
            all_R.extend(R)

        current_timestep += len(S)

        action_enum = unwrapped_env.action_map[action_idx]
        actions_taken.append(action_enum)
        rewards.append(reward)

    t = np.arange(len(all_S))

    save_name = agent_name if agent_name else None

    result = SimulationResult(
        agent=agent,
        t=t,
        S=np.array(all_S),
        E=np.array(all_E),
        I=np.array(all_I),
        R=np.array(all_R),
        actions=actions_taken,
        timesteps=timesteps,
        rewards=rewards,
        observations=observations,
        custom_name=save_name,
    )

    if save_name is None:
        save_name = result.agent_name.lower().replace(" ", "_").replace("-", "_")

    # Save logs and plots
    if experiment_dir is not None:
        log_path = experiment_dir.get_log_path(save_name)
        plot_path = experiment_dir.get_plot_path(f"{save_name}_seir.png")
    else:
        log_path = os.path.join("logs", f"{save_name}.txt")
        os.makedirs("results", exist_ok=True)
        plot_path = os.path.join("results", f"{save_name}_seir.png")

    log_results(result, log_path=str(log_path))
    plot_single_result(result, save_path=str(plot_path))

    return result


def run_evaluation(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    rl_models: Dict[str, List[Union[PPO, RecurrentPPO]]],
) -> tuple[List[SimulationResult], Dict[str, dict]]:
    """Run multi-seed statistical evaluation and best-seed trajectory evaluation.

    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory for saving results.
        rl_models: Dict mapping agent_name -> list of models (one per seed).

    Returns:
        Tuple of (trajectory results for best seeds, multi-seed statistics dict).
    """
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    results: List[SimulationResult] = []
    multi_seed_stats: Dict[str, dict] = {}

    # Evaluate non-RL agents (single-episode trajectory)
    env = create_environment(exp_config.base_config, exp_config.pomdp_params, seed=exp_config.training_seeds[0])
    agents = create_baseline_agents(exp_config.base_config, exp_config.target_agents)

    for agent in agents:
        agent_name = agent.__class__.__name__.lower()
        print(f"\nEvaluating {agent.__class__.__name__}...")
        result = run_agent(agent, env, experiment_dir=experiment_dir, agent_name=agent_name)
        results.append(result)

    # Evaluate RL agents: multi-seed statistics + best-seed trajectory
    for agent_name, models in rl_models.items():
        print(f"\nEvaluating {agent_name} ({len(models)} seeds)...")

        # Phase 1: Multi-seed statistical evaluation
        stats = evaluate_multi_seed(
            models=models,
            seeds=exp_config.training_seeds[: len(models)],
            config=exp_config.base_config,
            pomdp_params=exp_config.pomdp_params,
            agent_name=agent_name,
            experiment_dir=experiment_dir,
            n_eval_episodes_per_seed=exp_config.num_eval_episodes,
        )
        multi_seed_stats[agent_name] = stats

        print(
            f"  {agent_name}: mean={stats['overall_mean']:.2f} "
            f"± {stats['overall_std']:.2f} "
            f"(95% CI: [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}])"
        )
        print(f"  Best seed: {stats['best_seed']} (idx={stats['best_seed_idx']})")

        # Phase 2: Trajectory evaluation for the best seed model
        best_model = models[stats["best_seed_idx"]]
        best_seed = stats["best_seed"]
        vecnorm_path = str(
            experiment_dir.get_vecnormalize_path(agent_name, best_seed)
        )

        eval_env = create_eval_vec_env(
            exp_config.base_config,
            exp_config.pomdp_params,
            agent_name,
            vecnorm_path,
            seed=best_seed,
        )
        result = run_agent(
            best_model, eval_env, experiment_dir=experiment_dir, agent_name=agent_name
        )
        results.append(result)
        eval_env.close()

    return results, multi_seed_stats
