"""
Main entry point for epidemic control experiments.

Runs predefined scenarios with multi-seed training and evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import typer

from src.results import cross_seed_se

from src.config import Config
from src.results import AggregatedResult
from src.evaluation import run_evaluation
from src.experiment import ExperimentConfig, ExperimentDirectory, generate_seeds
from src.scenarios import (
    get_scenario,
    get_agent_variant_name,
    list_scenarios,
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
    scenario: str,
    total_timesteps: int,
    num_seeds: int,
    training_seeds: List[int],
    deterministic: bool = False,
    lstm_hidden_size: Optional[int] = None,
    n_stack: Optional[int] = None,
    ent_coef: Optional[float] = None,
    recurrent_ent_coef: Optional[float] = None,
    recurrent_n_steps: Optional[int] = None,
    run_name: Optional[str] = None,
) -> ExperimentConfig:
    """Build ExperimentConfig from a predefined scenario.

    Args:
        scenario: Predefined scenario name.
        total_timesteps: RL training budget.
        num_seeds: Number of training seeds.
        training_seeds: Deterministic seed list.
        deterministic: If True, use deterministic ODE dynamics (stochastic=False in Config).
        lstm_hidden_size: LSTM hidden size override for RecurrentPPO, or None for default.
        n_stack: FrameStack depth override for ppo_framestack, or None for default (10).
        ent_coef: Entropy bonus override for PPO/FrameStack agents, or None for default (0.01).
        recurrent_ent_coef: Entropy bonus override for RecurrentPPO, or None for default (0.05).
        recurrent_n_steps: Rollout length override for RecurrentPPO, or None for default (256).
        run_name: Custom subfolder name for the run, or None to use the timestamp.

    Returns:
        Fully populated ExperimentConfig.
    """
    base_config = Config(stochastic=not deterministic)
    if lstm_hidden_size is not None:
        base_config.lstm_hidden_size = lstm_hidden_size
    if n_stack is not None:
        base_config.n_stack = n_stack
    if ent_coef is not None:
        base_config.ent_coef = ent_coef
    if recurrent_ent_coef is not None:
        base_config.recurrent_ent_coef = recurrent_ent_coef
    if recurrent_n_steps is not None:
        base_config.recurrent_n_steps = recurrent_n_steps
    det_suffix = "_det" if deterministic else ""

    print(f"Running predefined scenario: {scenario}")
    scenario_config = get_scenario(scenario)
    return ExperimentConfig(
        base_config=base_config,
        pomdp_params=scenario_config["pomdp_params"],
        scenario_name=scenario + det_suffix + f"_t{total_timesteps}",
        target_agents=[get_agent_variant_name(a, base_config) for a in scenario_config["target_agents"]],
        total_timesteps=total_timesteps,
        num_training_seeds=num_seeds,
        training_seeds=training_seeds,
        run_name=run_name,
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
            f"reward = {np.mean(agg.seed_mean_rewards):.2f} ± {cross_seed_se(agg.seed_mean_rewards):.2f} (SE), "
            f"peak I = {np.mean(agg.seed_mean_peak):.1f} ± {cross_seed_se(agg.seed_mean_peak):.1f}, "
            f"total inf = {np.mean(agg.seed_mean_infected):.1f} ± {cross_seed_se(agg.seed_mean_infected):.1f}, "
            f"stringency = {np.mean(agg.seed_mean_stringency):.2f} ± {cross_seed_se(agg.seed_mean_stringency):.2f} "
            f"(n={agg.n_seeds} seeds × {agg.n_episodes // agg.n_seeds} ep)"
        )

    print("\n" + "=" * 80)


@app.command()
def main(
    scenario: str = typer.Option(
        ...,
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
        500_000,
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
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Use deterministic ODE dynamics instead of stochastic Binomial transitions.",
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
    ent_coef: Optional[float] = typer.Option(
        None,
        "--ent-coef",
        help="Entropy bonus for PPO and FrameStack agents (default: 0.01).",
    ),
    recurrent_ent_coef: Optional[float] = typer.Option(
        None,
        "--recurrent-ent-coef",
        help="Entropy bonus for RecurrentPPO (default: 0.05).",
    ),
    recurrent_n_steps: Optional[int] = typer.Option(
        None,
        "--recurrent-n-steps",
        help="Rollout length per env for RecurrentPPO (default: 256).",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Custom subfolder name for the run (default: auto-generated timestamp).",
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Scenario folder name to resume training from (e.g. 'pomdp_t500000'). "
             "Loads weights from experiments/{name}/weights/. "
             "Agents without matching weights train from scratch.",
    ),
):
    """Run epidemic control experiment with multi-seed training and evaluation."""
    agents_to_skip = _parse_skip_training(skip_training)
    training_seeds = generate_seeds(num_seeds)

    # Resolve resume-from weights directory
    resume_from_weights_dir = None
    if resume_from is not None:
        resume_from_weights_dir = Path("experiments") / resume_from / "weights"
        if not resume_from_weights_dir.exists():
            print(f"ERROR: Resume weights directory not found: {resume_from_weights_dir}")
            raise typer.Exit(code=1)
        print(f"Resuming training from: {resume_from_weights_dir}")

    exp_config = _build_experiment_config(
        scenario,
        total_timesteps, num_seeds, training_seeds,
        deterministic=deterministic,
        lstm_hidden_size=lstm_hidden_size,
        n_stack=n_stack,
        ent_coef=ent_coef,
        recurrent_ent_coef=recurrent_ent_coef,
        recurrent_n_steps=recurrent_n_steps,
        run_name=run_name,
    )
    if resume_from is not None:
        exp_config.resumed_from = resume_from

    experiment_dir = ExperimentDirectory(exp_config)
    print(f"Results will be saved to: {experiment_dir.root}")
    print(f"Training seeds: {exp_config.training_seeds}")
    experiment_dir.save_config()

    rl_models = prepare_rl_agents(exp_config, experiment_dir, agents_to_skip, resume_from_weights_dir)
    aggregated_results, per_seed_stats = run_evaluation(exp_config, experiment_dir, rl_models)

    _create_plots(exp_config, experiment_dir, aggregated_results, rl_models)
    experiment_dir.save_evaluation(per_seed_stats)
    experiment_dir.save_summary(aggregated_results)
    _print_summary(exp_config, experiment_dir, aggregated_results)


if __name__ == "__main__":
    app()
