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
from src.results import AggregatedResult
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


def _parse_csv_floats(raw: Optional[str], name: str) -> Optional[List[float]]:
    """Parse a comma-separated string of floats (e.g. '0.05,0.3,0.15').

    Args:
        raw: Raw CLI string or None.
        name: Argument name for error messages.

    Returns:
        List of floats, or None if raw is None.
    """
    if raw is None:
        return None
    try:
        return [float(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise typer.BadParameter(f"--{name} must be comma-separated numbers, got: '{raw}'")


def _parse_csv_ints(raw: Optional[str], name: str) -> Optional[List[int]]:
    """Parse a comma-separated string of ints (e.g. '5,14').

    Args:
        raw: Raw CLI string or None.
        name: Argument name for error messages.

    Returns:
        List of ints, or None if raw is None.
    """
    if raw is None:
        return None
    try:
        return [int(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise typer.BadParameter(f"--{name} must be comma-separated integers, got: '{raw}'")


def _build_experiment_config(
    scenario: Optional[str],
    no_exposed: bool,
    detection_rate: float,
    noise_stds: Optional[List[float]],
    total_timesteps: int,
    num_seeds: int,
    training_seeds: List[int],
    deterministic: bool = False,
    lag: Optional[List[int]] = None,
    testing_capacity: Optional[float] = None,
    action_delay: Optional[int] = None,
    noise_rho: float = 0.0,
    lstm_hidden_size: Optional[int] = None,
    n_stack: Optional[int] = None,
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
        lag: Lag range [min_lag, max_lag] in days, or None to disable.
        testing_capacity: Fraction of population testable per day, or None to disable.
        action_delay: Action implementation delay in days, or None to disable.
        noise_rho: AR(1) autocorrelation coefficient for multiplicative noise.
        lstm_hidden_size: LSTM hidden size override for RecurrentPPO, or None for default.
        n_stack: FrameStack depth override for ppo_framestack, or None for default (10).

    Returns:
        Fully populated ExperimentConfig.
    """
    base_config = Config(stochastic=not deterministic)
    if lstm_hidden_size is not None:
        base_config.lstm_hidden_size = lstm_hidden_size
    if n_stack is not None:
        base_config.n_stack = n_stack
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
        "noise_rho": noise_rho,
        "lag": lag,
        "testing_capacity": testing_capacity,
        "action_delay": action_delay,
    }
    return ExperimentConfig(
        base_config=base_config,
        pomdp_params=pomdp_params,
        scenario_name=create_custom_scenario_name(
            pomdp_params, total_timesteps=total_timesteps, deterministic=deterministic,
            lstm_hidden_size=lstm_hidden_size,
            n_stack=n_stack,
        ),
        is_custom=True,
        target_agents=TARGET_AGENTS.copy(),
        total_timesteps=total_timesteps,
        num_training_seeds=num_seeds,
        training_seeds=training_seeds,
    )


def _create_plots(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    aggregated_results: Dict[str, AggregatedResult],
    rl_models: dict,
) -> None:
    """Generate comparison and evaluation-curve plots.

    Args:
        exp_config: Experiment configuration (for scenario name and seeds).
        experiment_dir: Experiment directory (for plot paths and logs dir).
        aggregated_results: Dict mapping agent_name -> AggregatedResult.
        rl_models: Dict mapping agent_name -> list of models.
    """
    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    comparison_path = experiment_dir.get_plot_path("comparison_all_agents.png")
    plot_all_results(aggregated_results, save_path=str(comparison_path))

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
    aggregated_results: Dict[str, AggregatedResult],
) -> None:
    """Print final experiment summary to stdout.

    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory (for root path).
        aggregated_results: Dict mapping agent_name -> AggregatedResult.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nScenario: {exp_config.scenario_name}")
    print(f"Training seeds: {exp_config.training_seeds}")
    print(f"Eval seeds: {exp_config.eval_seeds}")
    print(f"Results directory: {experiment_dir.root}")

    print("\nAll agents:")
    for agent_name, agg in aggregated_results.items():
        print(
            f"  - {agent_name}: "
            f"reward = {agg.mean_reward:.2f} ± {agg.std_reward:.2f}, "
            f"peak I = {agg.mean_peak_infected:.1f} ± {agg.std_peak_infected:.1f}, "
            f"total inf = {agg.mean_total_infected:.1f} ± {agg.std_total_infected:.1f}, "
            f"stringency = {agg.mean_total_stringency:.2f} ± {agg.std_total_stringency:.2f} "
            f"(n={agg.n_episodes})"
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
        300_000,
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
    noise_stds: Optional[str] = typer.Option(
        None,
        "--noise-stds",
        help=(
            "Per-compartment multiplicative noise stds (comma-separated). "
            "E.g. --noise-stds 0.05,0.3,0.15 for [S, I, R] when --no-exposed is set. "
            "None = disabled."
        ),
    ),
    noise_rho: float = typer.Option(
        0.0,
        "--noise-rho",
        help=(
            "AR(1) autocorrelation coefficient for multiplicative noise in [0, 1). "
            "0.0 = iid noise (default), 0.7 = persistent measurement bias "
            "(decorrelation half-life ≈ 2 steps / 10 days)."
        ),
    ),
    testing_capacity: Optional[float] = typer.Option(
        None,
        "--testing-capacity",
        help=(
            "Fraction of population testable per day. When set, detection rate "
            "drops during surges (Michaelis-Menten saturation). E.g. 0.015 = 1.5%%/day."
        ),
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Use deterministic ODE dynamics instead of stochastic Binomial transitions.",
    ),
    lag: Optional[str] = typer.Option(
        None,
        "--lag",
        help=(
            "Temporal lag range [min,max] in days (comma-separated). "
            "E.g. --lag 5,14. None = disabled."
        ),
    ),
    action_delay: Optional[int] = typer.Option(
        None,
        "--action-delay",
        help="Action implementation delay in days. None = disabled.",
    ),
    lstm_hidden_size: Optional[int] = typer.Option(
        None,
        "--lstm-hidden-size",
        help="LSTM hidden size for RecurrentPPO (default: 32).",
    ),
    n_stack: Optional[int] = typer.Option(
        None,
        "--n-stack",
        help="FrameStack depth for ppo_framestack (default: 10).",
    ),
):
    """
    Run epidemic control experiment with multi-seed training and evaluation
    """
    agents_to_skip = _parse_skip_training(skip_training)
    training_seeds = generate_seeds(num_seeds)
    parsed_noise_stds = _parse_csv_floats(noise_stds, "noise-stds")
    parsed_lag = _parse_csv_ints(lag, "lag")

    if parsed_lag is not None and len(parsed_lag) != 2:
        raise typer.BadParameter(f"--lag requires exactly 2 values (min,max), got {len(parsed_lag)}")

    exp_config = _build_experiment_config(
        scenario, no_exposed, detection_rate, parsed_noise_stds,
        total_timesteps, num_seeds, training_seeds,
        deterministic=deterministic,
        lag=parsed_lag,
        testing_capacity=testing_capacity,
        action_delay=action_delay,
        noise_rho=noise_rho,
        lstm_hidden_size=lstm_hidden_size,
        n_stack=n_stack,
    )

    experiment_dir = ExperimentDirectory(exp_config)
    print(f"Results will be saved to: {experiment_dir.root}")
    print(f"Training seeds: {exp_config.training_seeds}")
    experiment_dir.save_config()

    rl_models = prepare_rl_agents(exp_config, experiment_dir, agents_to_skip)
    aggregated_results, per_seed_stats = run_evaluation(exp_config, experiment_dir, rl_models)

    _create_plots(exp_config, experiment_dir, aggregated_results, rl_models)
    experiment_dir.save_summary(aggregated_results, per_seed_stats)
    _print_summary(exp_config, experiment_dir, aggregated_results)


if __name__ == "__main__":
    app()
