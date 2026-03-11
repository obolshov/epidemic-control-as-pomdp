"""
Evaluation module for epidemic control agents.

Provides:
- _collect_trajectory(): Single-episode trajectory collection (no I/O).
- evaluate_agent(): Multi-episode evaluation for ANY agent, returns AggregatedResult.
- select_best_model(): Quick reward-only model selection across training seeds.
- run_evaluation(): Unified evaluation pipeline for baselines and RL agents.
"""

import numpy as np
import os
from typing import Any, Dict, List, Optional, Union

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecMonitor, VecNormalize
from sb3_contrib import RecurrentPPO

from src.agents import Agent, RandomAgent, create_baseline_agents
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.config import Config
from src.env import AggregatedResult, EpidemicEnv, SimulationResult
from src.utils import log_results, plot_single_aggregated
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


def _collect_trajectory(
    agent: Agent,
    env: Union[EpidemicEnv, VecEnv],
) -> SimulationResult:
    """Run a single episode and collect the full SEIR trajectory.

    Pure computation — no I/O (no log/plot saving).

    Args:
        agent: Agent (baseline or RL model) with SB3-compatible predict().
        env: Environment (VecEnv for RL agents, raw Gymnasium env for baselines).

    Returns:
        SimulationResult with full SEIR trajectory arrays.
    """
    is_vec_env = isinstance(env, VecEnv)

    obs = env.reset()

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
        if len(obs_flat) == 6 or (len(obs_flat) > 6 and len(obs_flat) % 6 == 0):
            S_init, E_init, I_init, R_init = obs_flat[-6:-2]
        elif len(obs_flat) == 5 or (len(obs_flat) > 5 and len(obs_flat) % 5 == 0):
            S_init, I_init, R_init = obs_flat[-5:-2]
            E_init = 0.0
        elif len(obs_flat) == 4:
            S_init, E_init, I_init, R_init = obs_flat[-4:]
        elif len(obs_flat) == 3:
            S_init, I_init, R_init = obs_flat[-3:]
            E_init = 0.0
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

    return SimulationResult(
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
    )


def evaluate_agent(
    agent: Agent,
    config: Config,
    pomdp_params: Dict[str, Any],
    agent_name: str,
    eval_seeds: List[int],
    vecnorm_path: Optional[str] = None,
) -> tuple[AggregatedResult, List[SimulationResult]]:
    """Unified multi-episode evaluation for ANY agent type.

    For each eval_seed, creates a fresh environment, collects a full SEIR
    trajectory, and aggregates statistics (mean +/- SD) across episodes.

    Args:
        agent: Agent to evaluate (baseline or RL model).
        config: Base SEIR configuration.
        pomdp_params: POMDP wrapper parameters.
        agent_name: Name of the agent (also determines VecFrameStack for RL).
        eval_seeds: List of evaluation seeds (one episode per seed).
        vecnorm_path: Path to frozen VecNormalize stats (RL agents only).

    Returns:
        Tuple of (AggregatedResult, list of per-episode SimulationResults).
    """
    is_rl = isinstance(agent, (PPO, RecurrentPPO))

    all_results: List[SimulationResult] = []

    for seed in eval_seeds:
        if is_rl:
            env = create_eval_vec_env(config, pomdp_params, agent_name, vecnorm_path, seed=seed)
        else:
            env = create_environment(config, pomdp_params, seed=seed)

        result = _collect_trajectory(agent, env)
        result.custom_name = agent_name
        all_results.append(result)

        if is_rl:
            env.close()

    all_S = [r.S for r in all_results]
    all_E = [r.E for r in all_results]
    all_I = [r.I for r in all_results]
    all_R = [r.R for r in all_results]

    # Truncate to shortest trajectory length (stochastic envs may differ slightly)
    min_len = min(len(s) for s in all_S)
    S_stack = np.stack([s[:min_len] for s in all_S], axis=0)
    E_stack = np.stack([e[:min_len] for e in all_E], axis=0)
    I_stack = np.stack([i[:min_len] for i in all_I], axis=0)
    R_stack = np.stack([r[:min_len] for r in all_R], axis=0)

    t = np.arange(min_len)

    agg = AggregatedResult(
        agent_name=agent_name,
        t=t,
        S_mean=np.mean(S_stack, axis=0),
        S_std=np.std(S_stack, axis=0),
        E_mean=np.mean(E_stack, axis=0),
        E_std=np.std(E_stack, axis=0),
        I_mean=np.mean(I_stack, axis=0),
        I_std=np.std(I_stack, axis=0),
        R_mean=np.mean(R_stack, axis=0),
        R_std=np.std(R_stack, axis=0),
        episode_rewards=[r.total_reward for r in all_results],
        peak_infected_per_episode=[float(r.peak_infected) for r in all_results],
        total_infected_per_episode=[float(r.total_infected) for r in all_results],
        n_episodes=len(eval_seeds),
    )

    return agg, all_results


def select_best_model(
    models: List[Union[PPO, RecurrentPPO]],
    seeds: List[int],
    config: Config,
    pomdp_params: Dict[str, Any],
    agent_name: str,
    experiment_dir: "ExperimentDirectory",
    n_eval_episodes: int = 10,
) -> tuple:
    """Quick reward-only evaluation to select the best training seed model.

    Uses SB3's evaluate_policy for fast reward-only evaluation (no trajectory
    collection). Full trajectory evaluation is done afterwards by evaluate_agent().

    Args:
        models: List of trained models (one per training seed).
        seeds: List of training seeds corresponding to models.
        config: Base configuration.
        pomdp_params: POMDP wrapper parameters.
        agent_name: Agent name.
        experiment_dir: Experiment directory for VecNormalize paths.
        n_eval_episodes: Episodes for model selection evaluation.

    Returns:
        Tuple of (best_model, best_seed, per_seed_mean_rewards).
    """
    per_seed_mean_rewards: List[Dict[str, Any]] = []

    for model, seed in zip(models, seeds):
        vecnorm_path = str(experiment_dir.get_vecnormalize_path(agent_name, seed))
        eval_env = create_eval_vec_env(
            config, pomdp_params, agent_name, vecnorm_path, seed=seed
        )

        episode_rewards, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        per_seed_mean_rewards.append({
            "seed": seed,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episode_rewards": [float(r) for r in episode_rewards],
        })

        eval_env.close()

    best_idx = int(np.argmax([s["mean_reward"] for s in per_seed_mean_rewards]))
    best_model = models[best_idx]
    best_seed = seeds[best_idx]

    return best_model, best_seed, per_seed_mean_rewards


def _save_episode_logs(
    experiment_dir: ExperimentDirectory,
    agent_name: str,
    eval_seeds: List[int],
    episode_results: List[SimulationResult],
) -> None:
    """Save per-episode action logs to logs/{agent_name}/seed_{seed}.txt."""
    agent_log_dir = experiment_dir.logs_dir / agent_name
    agent_log_dir.mkdir(parents=True, exist_ok=True)

    for seed, result in zip(eval_seeds, episode_results):
        log_path = agent_log_dir / f"seed_{seed}.txt"
        log_results(result, log_path=str(log_path))


def run_evaluation(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    rl_models: Dict[str, List[Union[PPO, RecurrentPPO]]],
) -> tuple[Dict[str, AggregatedResult], Dict[str, Any]]:
    """Run unified multi-episode evaluation for all agents.

    Phase 1: Baselines — evaluate_agent() on eval_seeds.
    Phase 2: RL agents — select_best_model(), then evaluate_agent() on same eval_seeds.

    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory for saving results.
        rl_models: Dict mapping agent_name -> list of models (one per seed).

    Returns:
        Tuple of (Dict[agent_name -> AggregatedResult], Dict[rl_agent_name -> per_seed_stats]).
    """
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    aggregated_results: Dict[str, AggregatedResult] = {}
    per_seed_stats: Dict[str, Any] = {}
    eval_seeds = exp_config.eval_seeds

    # Phase 1: Evaluate baselines
    agents = create_baseline_agents(exp_config.base_config, exp_config.target_agents)

    for agent in agents:
        agent_name = agent.__class__.__name__.lower()
        print(f"\nEvaluating {agent.__class__.__name__} ({len(eval_seeds)} episodes)...")

        # Re-seed RandomAgent per evaluation for reproducibility
        if isinstance(agent, RandomAgent):
            agent = RandomAgent(seed=eval_seeds[0])

        agg, episode_results = evaluate_agent(
            agent=agent,
            config=exp_config.base_config,
            pomdp_params=exp_config.pomdp_params,
            agent_name=agent_name,
            eval_seeds=eval_seeds,
        )
        aggregated_results[agent_name] = agg

        # Save per-agent aggregated SEIR plot
        plot_path = experiment_dir.get_plot_path(f"{agent_name}_seir.png")
        plot_single_aggregated(agg, save_path=str(plot_path))

        # Save per-episode logs
        _save_episode_logs(experiment_dir, agent_name, eval_seeds, episode_results)

        print(
            f"  {agent_name}: reward = {agg.mean_reward:.2f} ± {agg.std_reward:.2f}, "
            f"peak I = {agg.mean_peak_infected:.1f} ± {agg.std_peak_infected:.1f}"
        )

    # Phase 2: Evaluate RL agents
    for agent_name, models in rl_models.items():
        print(f"\nEvaluating {agent_name} ({len(models)} seeds)...")

        # Select best model across training seeds
        best_model, best_seed, seed_stats = select_best_model(
            models=models,
            seeds=exp_config.training_seeds[:len(models)],
            config=exp_config.base_config,
            pomdp_params=exp_config.pomdp_params,
            agent_name=agent_name,
            experiment_dir=experiment_dir,
            n_eval_episodes=exp_config.num_eval_episodes,
        )
        per_seed_stats[agent_name] = seed_stats
        print(f"  Best seed: {best_seed}")

        # Full trajectory evaluation on eval_seeds
        vecnorm_path = str(experiment_dir.get_vecnormalize_path(agent_name, best_seed))
        agg, episode_results = evaluate_agent(
            agent=best_model,
            config=exp_config.base_config,
            pomdp_params=exp_config.pomdp_params,
            agent_name=agent_name,
            eval_seeds=eval_seeds,
            vecnorm_path=vecnorm_path,
        )
        aggregated_results[agent_name] = agg

        # Save per-agent aggregated SEIR plot
        plot_path = experiment_dir.get_plot_path(f"{agent_name}_seir.png")
        plot_single_aggregated(agg, save_path=str(plot_path))

        # Save per-episode logs
        _save_episode_logs(experiment_dir, agent_name, eval_seeds, episode_results)

        print(
            f"  {agent_name}: reward = {agg.mean_reward:.2f} ± {agg.std_reward:.2f}, "
            f"peak I = {agg.mean_peak_infected:.1f} ± {agg.std_peak_infected:.1f}"
        )

    return aggregated_results, per_seed_stats
