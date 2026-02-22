"""
Main entry point for epidemic control experiments.

Supports multiple modes:
- Predefined scenarios (--scenario mdp, --scenario no_exposed)
- Custom POMDP configurations (--no-exposed, --detection-rate, etc.)
- Multi-seed training with statistical evaluation
"""

from typing import Dict, List, Optional, Union

import numpy as np
import typer
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from src.agents import (
    Agent,
    RandomAgent,
    ThresholdAgent,
)
from src.config import Config
from src.env import EpidemicEnv, SimulationResult
from src.evaluation import (
    create_eval_vec_env,
    evaluate_multi_seed,
    run_agent,
)
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.scenarios import (
    get_scenario,
    create_custom_scenario_name,
    list_scenarios,
    TARGET_AGENTS,
)
from src.train import train_ppo_agent
from src.wrappers import create_environment
from src.utils import plot_all_results, plot_evaluation_curves, plot_learning_curve


app = typer.Typer(help="Epidemic Control as POMDP - Experiment Runner")


def setup_target_agents(config: Config, agent_names: List[str]) -> List[Agent]:
    """
    Initialize non-RL agents for evaluation.

    Args:
        config: Configuration for agents.
        agent_names: List of agent names to initialize.

    Returns:
        List of initialized Agent objects.
    """
    agents = []

    if "random" in agent_names:
        agents.append(RandomAgent())

    if "threshold" in agent_names:
        agents.append(ThresholdAgent(config))

    return agents


def prepare_rl_agents(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    agents_to_skip: set,
) -> Dict[str, List[Union[PPO, RecurrentPPO]]]:
    """
    Train or load RL agents across multiple seeds.

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
                # Load existing weights
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
                # Train agent
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

                # Plot monitor-based learning curve for this seed (must match per-seed monitor_dir in train.py)
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


def run_evaluation(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    rl_models: Dict[str, List[Union[PPO, RecurrentPPO]]],
) -> tuple[List[SimulationResult], Dict[str, dict]]:
    """
    Run multi-seed statistical evaluation and best-seed trajectory evaluation.

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

    results = []
    multi_seed_stats = {}

    # Evaluate non-RL agents (single-episode trajectory)
    env = create_environment(exp_config.base_config, exp_config.pomdp_params)
    agents = setup_target_agents(exp_config.base_config, exp_config.target_agents)

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


@app.command()
def main(
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        "-s",
        help=f"Predefined scenario to run. Available: {', '.join(list_scenarios())}",
    ),
    skip_training: Optional[str] = typer.Option(
        None,
        "--skip-training",
        help="Skip training for specified agents (comma-separated) or 'all'.",
    ),
    total_timesteps: int = typer.Option(
        200_000,
        "--timesteps",
        "-t",
        help="Maximum timesteps for RL training (early stopping may stop sooner)",
    ),
    num_seeds: int = typer.Option(
        5,
        "--num-seeds",
        "-n",
        help="Number of independent training seeds per agent",
    ),
    # POMDP parameters (for custom scenarios)
    no_exposed: bool = typer.Option(
        False,
        "--no-exposed",
        help="Mask E (Exposed) compartment from observations",
    ),
    detection_rate: float = typer.Option(
        1.0,
        "--detection-rate",
        help="Fraction of true I and R observed (1.0=full, 0.3=COVID-realistic).",
    ),
):
    """
    Run epidemic control experiment with multi-seed training and evaluation.

    Examples:
        python main.py --scenario mdp
        python main.py --scenario mdp --skip-training all
        python main.py --scenario no_exposed_underreporting -t 200000 --num-seeds 5
        python main.py --no-exposed --detection-rate 0.3
    """

    # Parse skip_training argument
    agents_to_skip = set()
    if skip_training is not None:
        if skip_training.lower() == "all" or skip_training == "":
            agents_to_skip = {"all"}
        else:
            agents_to_skip = set(agent.strip() for agent in skip_training.split(","))

    # Generate training seeds
    training_seeds = _generate_seeds(num_seeds)

    # Mode 1: Predefined scenario
    if scenario:
        print(f"Running predefined scenario: {scenario}")
        scenario_config = get_scenario(scenario)

        exp_config = ExperimentConfig(
            base_config=Config(),
            pomdp_params=scenario_config["pomdp_params"],
            scenario_name=scenario,
            is_custom=False,
            target_agents=scenario_config["target_agents"],
            total_timesteps=total_timesteps,
            num_training_seeds=num_seeds,
            training_seeds=training_seeds,
        )

    # Mode 2: Custom configuration
    else:
        print("Running custom experiment")
        pomdp_params = {
            "include_exposed": not no_exposed,
            "detection_rate": detection_rate,
        }

        exp_config = ExperimentConfig(
            base_config=Config(),
            pomdp_params=pomdp_params,
            scenario_name=create_custom_scenario_name(pomdp_params),
            is_custom=True,
            target_agents=TARGET_AGENTS.copy(),
            total_timesteps=total_timesteps,
            num_training_seeds=num_seeds,
            training_seeds=training_seeds,
        )

    experiment_dir = ExperimentDirectory(exp_config)
    print(f"Results will be saved to: {experiment_dir.root}")
    print(f"Training seeds: {exp_config.training_seeds}")

    # Save experiment configuration
    experiment_dir.save_config()

    # Prepare RL agents (train or load, multi-seed)
    rl_models = prepare_rl_agents(exp_config, experiment_dir, agents_to_skip)

    # Run evaluation (multi-seed stats + best-seed trajectory)
    results, multi_seed_stats = run_evaluation(exp_config, experiment_dir, rl_models)

    # Create comparison plot (best-seed trajectories)
    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    comparison_path = experiment_dir.get_plot_path("comparison_all_agents.png")
    plot_all_results(results, save_path=str(comparison_path))

    # Create evaluation curves plot: one curve per agent, aggregated across seeds
    eval_log_paths_by_agent: Dict[str, List[str]] = {}
    for agent_name in rl_models:
        seed_dirs = []
        for seed in exp_config.training_seeds:
            eval_dir = experiment_dir.logs_dir / f"{agent_name}_seed{seed}_eval"
            seed_dirs.append(str(eval_dir))
        eval_log_paths_by_agent[agent_name] = seed_dirs

    if eval_log_paths_by_agent:
        eval_curve_path = experiment_dir.get_plot_path("evaluation_curves.png")
        plot_evaluation_curves(
            eval_log_paths_by_agent,
            title=f"Evaluation During Training — {exp_config.scenario_name}",
            save_path=str(eval_curve_path),
        )

    # Save summary with multi-seed statistics
    experiment_dir.save_summary(results, multi_seed_stats)

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nScenario: {exp_config.scenario_name}")
    print(f"Training seeds: {exp_config.training_seeds}")
    print(f"Results directory: {experiment_dir.root}")

    print(f"\nNon-RL agents:")
    for result in results:
        if not result.agent_name.startswith("ppo_"):
            print(
                f"  - {result.agent_name}: "
                f"Peak I = {result.peak_infected:.1f}, "
                f"Total Reward = {result.total_reward:.2f}"
            )

    if multi_seed_stats:
        print(f"\nRL agents (multi-seed evaluation):")
        for agent_name, stats in multi_seed_stats.items():
            print(
                f"  - {agent_name}: "
                f"mean={stats['overall_mean']:.2f} ± {stats['overall_std']:.2f} "
                f"(95% CI: [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}])"
            )
            for seed_stat in stats["per_seed"]:
                print(
                    f"      seed={seed_stat['seed']}: "
                    f"mean={seed_stat['mean_reward']:.2f} ± {seed_stat['std_reward']:.2f}"
                )

    print("\n" + "=" * 80)


def _generate_seeds(num_seeds: int) -> List[int]:
    """Generate a deterministic list of training seeds.

    Args:
        num_seeds: Number of seeds to generate.

    Returns:
        List of integer seeds.
    """
    base_seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999]
    if num_seeds <= len(base_seeds):
        return base_seeds[:num_seeds]
    # Extend with deterministic RNG if more seeds needed
    rng = np.random.default_rng(42)
    extra = rng.integers(0, 10000, size=num_seeds - len(base_seeds)).tolist()
    return base_seeds + extra


if __name__ == "__main__":
    app()
