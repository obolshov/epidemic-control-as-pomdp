"""
Main entry point for epidemic control experiments.

Supports multiple modes:
- Predefined scenarios (--scenario mdp, --scenario no_exposed)
- Custom POMDP configurations (--no-exposed, --detection-rate, etc.)
- Multi-seed training with statistical evaluation
"""

from typing import Any, Dict, List, Optional

import typer

from src.config import Config
from src.evaluation import run_evaluation
from src.experiment import ExperimentConfig, ExperimentDirectory, generate_seeds
from src.scenarios import (
    get_scenario,
    create_custom_scenario_name,
    list_scenarios,
    TARGET_AGENTS,
)
from src.train import prepare_rl_agents
from src.utils import plot_all_results, plot_evaluation_curves


app = typer.Typer(help="Epidemic Control as POMDP - Experiment Runner")


def _parse_skip_training(skip_training: Optional[str]) -> set:
    """Parse --skip-training CLI argument into a set of agent names.

    Args:
        skip_training: Raw CLI string (comma-separated names, "all", or None).

    Returns:
        Set of agent names to skip, or {"all"} to skip all, or empty set.
    """
    if skip_training is None:
        return set()
    if skip_training.lower() == "all" or skip_training == "":
        return {"all"}
    return set(agent.strip() for agent in skip_training.split(","))


def _build_experiment_config(
    scenario: Optional[str],
    no_exposed: bool,
    detection_rate: float,
    noise_stds: Optional[List[float]],
    total_timesteps: int,
    num_seeds: int,
    training_seeds: List[int],
    deterministic: bool = False,
) -> ExperimentConfig:
    """Build ExperimentConfig for either a predefined or custom scenario.

    Args:
        scenario: Predefined scenario name, or None for custom.
        no_exposed: Whether to mask the E compartment.
        detection_rate: Fraction of true I and R observed.
        noise_stds: Per-compartment multiplicative noise stds, or None to disable.
        total_timesteps: RL training budget.
        num_seeds: Number of training seeds.
        training_seeds: Deterministic seed list.
        deterministic: If True, use deterministic ODE dynamics (stochastic=False in Config).

    Returns:
        Fully populated ExperimentConfig.
    """
    base_config = Config(stochastic=not deterministic)
    det_suffix = "_det" if deterministic else ""

    if scenario:
        print(f"Running predefined scenario: {scenario}")
        scenario_config = get_scenario(scenario)
        return ExperimentConfig(
            base_config=base_config,
            pomdp_params=scenario_config["pomdp_params"],
            scenario_name=scenario + det_suffix + f"_t{total_timesteps}",
            is_custom=False,
            target_agents=scenario_config["target_agents"],
            total_timesteps=total_timesteps,
            num_training_seeds=num_seeds,
            training_seeds=training_seeds,
        )

    print("Running custom experiment")
    pomdp_params: Dict[str, Any] = {
        "include_exposed": not no_exposed,
        "detection_rate": detection_rate,
        "noise_stds": noise_stds,
    }
    return ExperimentConfig(
        base_config=base_config,
        pomdp_params=pomdp_params,
        scenario_name=create_custom_scenario_name(pomdp_params, total_timesteps=total_timesteps, deterministic=deterministic),
        is_custom=True,
        target_agents=TARGET_AGENTS.copy(),
        total_timesteps=total_timesteps,
        num_training_seeds=num_seeds,
        training_seeds=training_seeds,
    )


def _create_plots(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    results: list,
    rl_models: dict,
) -> None:
    """Generate comparison and evaluation-curve plots.

    Args:
        exp_config: Experiment configuration (for scenario name and seeds).
        experiment_dir: Experiment directory (for plot paths and logs dir).
        results: Best-seed trajectory SimulationResult list.
        rl_models: Dict mapping agent_name -> list of models.
    """
    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    comparison_path = experiment_dir.get_plot_path("comparison_all_agents.png")
    plot_all_results(results, save_path=str(comparison_path))

    eval_log_paths_by_agent: Dict[str, List[str]] = {
        agent_name: [
            str(experiment_dir.logs_dir / f"{agent_name}_seed{seed}_eval")
            for seed in exp_config.training_seeds
        ]
        for agent_name in rl_models
    }

    if eval_log_paths_by_agent:
        eval_curve_path = experiment_dir.get_plot_path("evaluation_curves.png")
        plot_evaluation_curves(
            eval_log_paths_by_agent,
            title=f"Evaluation During Training — {exp_config.scenario_name}",
            save_path=str(eval_curve_path),
        )


def _print_summary(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    results: list,
    multi_seed_stats: dict,
) -> None:
    """Print final experiment summary to stdout.

    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory (for root path).
        results: All SimulationResult objects.
        multi_seed_stats: Multi-seed statistics dict from run_evaluation.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nScenario: {exp_config.scenario_name}")
    print(f"Training seeds: {exp_config.training_seeds}")
    print(f"Results directory: {experiment_dir.root}")

    print("\nNon-RL agents:")
    for result in results:
        if not result.agent_name.startswith("ppo_"):
            print(
                f"  - {result.agent_name}: "
                f"Peak I = {result.peak_infected:.1f}, "
                f"Total Infected = {result.total_infected:.1f}, "
                f"Total Reward = {result.total_reward:.2f}"
            )

    if multi_seed_stats:
        print("\nRL agents (multi-seed evaluation):")
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
    noise_stds: Optional[List[float]] = typer.Option(
        None,
        "--noise-stds",
        help=(
            "Per-compartment multiplicative noise stds matching current obs shape. "
            "Pass once per compartment: e.g. --noise-stds 0.05 --noise-stds 0.3 --noise-stds 0.15 "
            "for [S, I, R] when --no-exposed is set. None = disabled."
        ),
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Use deterministic ODE dynamics instead of stochastic Binomial transitions.",
    ),
):
    """
    Run epidemic control experiment with multi-seed training and evaluation
    """
    agents_to_skip = _parse_skip_training(skip_training)
    training_seeds = generate_seeds(num_seeds)

    exp_config = _build_experiment_config(
        scenario, no_exposed, detection_rate, noise_stds,
        total_timesteps, num_seeds, training_seeds,
        deterministic=deterministic,
    )

    experiment_dir = ExperimentDirectory(exp_config)
    print(f"Results will be saved to: {experiment_dir.root}")
    print(f"Training seeds: {exp_config.training_seeds}")
    experiment_dir.save_config()

    rl_models = prepare_rl_agents(exp_config, experiment_dir, agents_to_skip)
    results, multi_seed_stats = run_evaluation(exp_config, experiment_dir, rl_models)

    _create_plots(exp_config, experiment_dir, results, rl_models)
    experiment_dir.save_summary(results, multi_seed_stats)
    _print_summary(exp_config, experiment_dir, results, multi_seed_stats)


if __name__ == "__main__":
    app()
