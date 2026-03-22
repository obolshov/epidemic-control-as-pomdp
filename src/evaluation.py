"""
Evaluation module for epidemic control agents.

Provides:
- _collect_trajectory(): Single-episode trajectory collection (no I/O).
- evaluate_agent(): Multi-episode evaluation for ANY agent, returns AggregatedResult.
- evaluate_all_seeds(): Full trajectory evaluation across all training seeds.
- aggregate_across_seeds(): Aggregate per-seed results into cross-seed summary.
- run_evaluation(): Unified evaluation pipeline for baselines and RL agents.
"""

import numpy as np
import os
from typing import Any, Dict, List, Optional, Union

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecMonitor, VecNormalize
from sb3_contrib import RecurrentPPO

from src.agents import Agent, RandomAgent, ThresholdAgent, create_baseline_agents
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.config import Config
from src.env import EpidemicEnv
from src.results import AggregatedResult, SimulationResult
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

    # Get initial state from unwrapped env (always EpidemicEnv with current_state)
    if not hasattr(unwrapped_env, 'current_state'):
        raise RuntimeError(
            f"Unwrapped env {type(unwrapped_env).__name__} has no current_state attribute. "
            "Expected EpidemicEnv at the bottom of the wrapper stack."
        )
    state = unwrapped_env.current_state
    S_init = state.S
    E_init = state.E
    I_init = state.I
    R_init = state.R

    all_S = [S_init]
    all_E = [E_init]
    all_I = [I_init]
    all_R = [R_init]

    actions_taken = []
    timesteps = []
    rewards = []
    observations = []
    reward_components = []

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
        reward_components.append({
            "reward_infection": info.get("reward_infection", 0.0),
            "reward_stringency": info.get("reward_stringency", 0.0),
            "reward_switching": info.get("reward_switching", 0.0),
        })

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
        reward_components=reward_components,
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

    # Truncate to shortest trajectory length (stochastic envs may differ slightly)
    min_len = min(len(r.S) for r in all_results)
    t = np.arange(min_len)

    # Stack and compute mean/std for each compartment
    stats = {}
    for comp in ("S", "E", "I", "R"):
        stacked = np.stack([getattr(r, comp)[:min_len] for r in all_results], axis=0)
        stats[f"{comp}_mean"] = np.mean(stacked, axis=0)
        stats[f"{comp}_std"] = np.std(stacked, axis=0)

    agg = AggregatedResult(
        agent_name=agent_name,
        t=t,
        **stats,
        episode_rewards=[r.total_reward for r in all_results],
        peak_infected_per_episode=[float(r.peak_infected) for r in all_results],
        total_infected_per_episode=[float(r.total_infected) for r in all_results],
        total_stringency_per_episode=[float(r.total_stringency) for r in all_results],
        n_episodes=len(eval_seeds),
    )

    return agg, all_results



def evaluate_all_seeds(
    models: List[Union[PPO, RecurrentPPO]],
    seeds: List[int],
    config: Config,
    pomdp_params: Dict[str, Any],
    agent_name: str,
    experiment_dir: "ExperimentDirectory",
    eval_seeds: List[int],
) -> List[AggregatedResult]:
    """Full trajectory evaluation for each training seed.

    Args:
        models: Trained models, one per training seed.
        seeds: Training seeds corresponding to models.
        config: Base SEIR configuration.
        pomdp_params: POMDP wrapper parameters.
        agent_name: Agent name (determines VecFrameStack).
        experiment_dir: For VecNormalize paths.
        eval_seeds: Evaluation seeds (each model evaluated on all of them).

    Returns:
        List of AggregatedResult, one per training seed.
    """
    per_seed_results: List[AggregatedResult] = []

    for model, seed in zip(models, seeds):
        vecnorm_path = str(experiment_dir.get_vecnormalize_path(agent_name, seed))
        agg, _ = evaluate_agent(
            agent=model,
            config=config,
            pomdp_params=pomdp_params,
            agent_name=agent_name,
            eval_seeds=eval_seeds,
            vecnorm_path=vecnorm_path,
        )
        print(f"    seed {seed}: reward = {agg.mean_reward:.4f} ± {agg.std_reward:.4f}")
        per_seed_results.append(agg)

    return per_seed_results


def aggregate_across_seeds(
    per_seed_results: List[AggregatedResult],
    seeds: List[int],
    agent_name: str,
) -> tuple[AggregatedResult, List[Dict[str, Any]]]:
    """Aggregate per-seed evaluation results into a cross-seed summary.

    Computes mean-of-seed-means for all metrics. The returned AggregatedResult
    has episode_rewards = all episodes from all seeds (concatenated), and
    per-seed stats are returned separately for summary.json.

    Args:
        per_seed_results: One AggregatedResult per training seed.
        seeds: Training seeds (same order as per_seed_results).
        agent_name: Agent name for the aggregated result.

    Returns:
        Tuple of (aggregated AggregatedResult, per_seed_stats list).
    """
    n_seeds = len(per_seed_results)

    # Collect seed-level means
    seed_mean_rewards = [agg.mean_reward for agg in per_seed_results]
    seed_mean_peak = [agg.mean_peak_infected for agg in per_seed_results]
    seed_mean_infected = [agg.mean_total_infected for agg in per_seed_results]
    seed_mean_stringency = [agg.mean_total_stringency for agg in per_seed_results]

    # Concatenate all episodes across seeds (for Wilcoxon tests etc.)
    all_episode_rewards: List[float] = []
    all_peak_infected: List[float] = []
    all_total_infected: List[float] = []
    all_total_stringency: List[float] = []

    per_seed_stats: List[Dict[str, Any]] = []
    for seed, agg in zip(seeds, per_seed_results):
        all_episode_rewards.extend(agg.episode_rewards)
        all_peak_infected.extend(agg.peak_infected_per_episode)
        all_total_infected.extend(agg.total_infected_per_episode)
        all_total_stringency.extend(agg.total_stringency_per_episode)

        per_seed_stats.append({
            "seed": seed,
            "mean_reward": agg.mean_reward,
            "std_reward": agg.std_reward,
            "episode_rewards": [float(r) for r in agg.episode_rewards],
        })

    # Aggregate SEIR timeseries: mean of per-seed means
    min_len = min(len(agg.t) for agg in per_seed_results)
    t = np.arange(min_len)
    stats = {}
    for comp in ("S", "E", "I", "R"):
        seed_means = np.stack(
            [getattr(agg, f"{comp}_mean")[:min_len] for agg in per_seed_results],
            axis=0,
        )
        stats[f"{comp}_mean"] = np.mean(seed_means, axis=0)
        stats[f"{comp}_std"] = np.std(seed_means, axis=0)

    combined_agg = AggregatedResult(
        agent_name=agent_name,
        t=t,
        **stats,
        episode_rewards=all_episode_rewards,
        peak_infected_per_episode=all_peak_infected,
        total_infected_per_episode=all_total_infected,
        total_stringency_per_episode=all_total_stringency,
        n_episodes=len(all_episode_rewards),
    )

    return combined_agg, per_seed_stats


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
    Phase 2: RL agents — evaluate_all_seeds() on all training seeds, then
        aggregate_across_seeds() for cross-seed mean ± SE.

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
    agents = create_baseline_agents(
        exp_config.base_config,
        exp_config.target_agents,
        pomdp_params=exp_config.pomdp_params,
    )

    for agent in agents:
        agent_name = getattr(agent, 'name', agent.__class__.__name__.lower())
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

        # Save per-episode logs (only for threshold agent — static/random logs are not useful)
        if isinstance(agent, ThresholdAgent):
            _save_episode_logs(experiment_dir, agent_name, eval_seeds, episode_results)

        print(
            f"  {agent_name}: reward = {agg.mean_reward:.2f} ± {agg.std_reward:.2f}, "
            f"peak I = {agg.mean_peak_infected:.1f} ± {agg.std_peak_infected:.1f}"
        )

    # Phase 2: Evaluate RL agents (all seeds, not just best)
    for agent_name, models in rl_models.items():
        training_seeds = exp_config.training_seeds[:len(models)]
        print(f"\nEvaluating {agent_name} ({len(models)} seeds × {len(eval_seeds)} episodes)...")

        # Full trajectory evaluation on all training seeds
        seed_results = evaluate_all_seeds(
            models=models,
            seeds=training_seeds,
            config=exp_config.base_config,
            pomdp_params=exp_config.pomdp_params,
            agent_name=agent_name,
            experiment_dir=experiment_dir,
            eval_seeds=eval_seeds,
        )

        # Aggregate across seeds
        agg, seed_stats = aggregate_across_seeds(
            seed_results, training_seeds, agent_name,
        )
        aggregated_results[agent_name] = agg
        per_seed_stats[agent_name] = seed_stats

        # SEIR plot from median seed (representative, not cherry-picked)
        seed_means = [r.mean_reward for r in seed_results]
        median_idx = int(np.argsort(seed_means)[len(seed_means) // 2])
        median_seed = training_seeds[median_idx]
        plot_path = experiment_dir.get_plot_path(f"{agent_name}_seir.png")
        plot_single_aggregated(seed_results[median_idx], save_path=str(plot_path))

        # Compute cross-seed stats for display
        seed_mean_arr = np.array(seed_means)
        cross_seed_mean = float(np.mean(seed_mean_arr))
        cross_seed_se = float(np.std(seed_mean_arr, ddof=1) / np.sqrt(len(seed_mean_arr)))
        best_seed = training_seeds[int(np.argmax(seed_mean_arr))]

        print(
            f"  {agent_name}: reward = {cross_seed_mean:.2f} ± {cross_seed_se:.2f} (SE), "
            f"best seed = {best_seed} ({max(seed_means):.2f}), "
            f"SEIR plot from median seed {median_seed}"
        )

    return aggregated_results, per_seed_stats
