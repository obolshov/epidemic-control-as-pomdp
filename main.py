"""
Main entry point for epidemic control experiments.

Supports multiple modes:
- Predefined scenarios (--scenario mdp, --scenario no_exposed)
- Custom POMDP configurations (--no-exposed, --delay, --noise, etc.)
- Loading and rerunning experiments (--load-experiment)
"""

from typing import List, Optional

import typer
from stable_baselines3 import PPO

from src.agents import (
    Agent,
    RandomAgent,
    ThresholdAgent,
)
from src.config import DefaultConfig, get_config
from src.env import EpidemicEnv, SimulationResult
from src.evaluation import run_agent
from src.experiment import ExperimentConfig, ExperimentDirectory
from src.scenarios import (
    get_scenario,
    create_custom_scenario_name,
    list_scenarios,
    TARGET_AGENTS,
)
from src.train import train_ppo_agent
from src.wrappers import EpidemicObservationWrapper, UnderReportingWrapper
from src.utils import plot_all_results, plot_learning_curve


app = typer.Typer(help="Epidemic Control as POMDP - Experiment Runner")


def create_environment(config: DefaultConfig, pomdp_params: dict) -> EpidemicEnv:
    """
    Create environment with appropriate POMDP wrappers.
    
    Args:
        config: Base configuration.
        pomdp_params: POMDP parameters for wrappers.
        
    Returns:
        Environment with wrappers applied.
    """
    env = EpidemicEnv(config)
    
    # Apply POMDP wrapper if partial observability is enabled
    if not pomdp_params.get("include_exposed", True):
        env = EpidemicObservationWrapper(env, include_exposed=False)

    # Apply under-reporting wrapper if detection_rate < 1.0
    detection_rate = pomdp_params.get("detection_rate", 1.0)
    if detection_rate < 1.0:
        env = UnderReportingWrapper(env, detection_rate=detection_rate)

    # Future wrappers can be added here:
    # if pomdp_params.get("noise_std", 0) > 0:
    #     env = NoiseWrapper(env, noise_std=pomdp_params["noise_std"])
    # if pomdp_params.get("delay", 0) > 0:
    #     env = DelayWrapper(env, delay=pomdp_params["delay"])

    return env


def setup_target_agents(config: DefaultConfig, agent_names: List[str]) -> List[Agent]:
    """
    Initialize target agents (non-static agents for evaluation).
    
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
    
    # PPO agents will be loaded or trained separately
    # They are not added here to avoid duplication
    
    return agents


def prepare_rl_agents(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    agents_to_skip: set,
) -> List[PPO]:
    """
    Train or load RL agents based on skip list.
    
    By default, trains all agents. Agents in skip list are loaded from weights.
    
    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory for saving/loading.
        agents_to_skip: Set of agent names to skip training (or {"all"} to skip all).
        
    Returns:
        List of PPO models (trained or loaded).
    """
    rl_agent_names = [
        name for name in exp_config.target_agents 
        if name.startswith("ppo_")
    ]
    
    if not rl_agent_names:
        return []
    
    skip_all = "all" in agents_to_skip
    
    print("\n" + "=" * 80)
    print("PREPARING RL AGENTS")
    print("=" * 80)
    
    models = []
    for agent_name in rl_agent_names:
        should_skip = skip_all or agent_name in agents_to_skip
        
        if should_skip:
            # Load existing weights
            weight_path = experiment_dir.get_weight_path(agent_name)
            if weight_path.exists():
                print(f"\nLoading {agent_name} from {weight_path}...")
                model = PPO.load(str(weight_path.with_suffix("")))
                models.append(model)
            else:
                raise FileNotFoundError(f"Weights for {agent_name} not found at {weight_path}")
        else:
            # Train agent
            print(f"\nTraining {agent_name}...")
            model = train_ppo_agent(
                env_cls=EpidemicEnv,
                config=exp_config.base_config,
                experiment_dir=experiment_dir,
                agent_name=agent_name,
                total_timesteps=exp_config.total_timesteps,
                pomdp_params=exp_config.pomdp_params,
            )
            models.append(model)
            
            # Plot learning curve
            tensorboard_log = experiment_dir.tensorboard_dir / agent_name
            if tensorboard_log.exists():
                plot_path = experiment_dir.get_plot_path(f"{agent_name}_learning.png")
                plot_learning_curve(
                    log_folder=str(tensorboard_log),
                    title=f"{agent_name} Learning Curve",
                    save_path=str(plot_path),
                )
    
    return models


def run_evaluation(
    exp_config: ExperimentConfig,
    experiment_dir: ExperimentDirectory,
    rl_models: List[PPO],
) -> List[SimulationResult]:
    """
    Run evaluation for all agents.
    
    Args:
        exp_config: Experiment configuration.
        experiment_dir: Experiment directory for saving results.
        rl_models: List of trained/loaded RL models.
        
    Returns:
        List of SimulationResult objects.
    """
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    
    # Create environment for evaluation (non-RL agents)
    env = create_environment(exp_config.base_config, exp_config.pomdp_params)
    
    # Setup non-RL agents
    agents = setup_target_agents(exp_config.base_config, exp_config.target_agents)
    
    results = []
    
    # Evaluate non-RL agents
    for agent in agents:
        agent_name = agent.__class__.__name__.lower()
        print(f"\nEvaluating {agent.__class__.__name__}...")
        result = run_agent(agent, env, experiment_dir=experiment_dir, agent_name=agent_name)
        results.append(result)
    
    # Evaluate RL agents
    rl_agent_names = [
        name for name in exp_config.target_agents 
        if name.startswith("ppo_")
    ]
    for model, agent_name in zip(rl_models, rl_agent_names):
        print(f"\nEvaluating {agent_name}...")
        
        # Create appropriate environment for this agent
        if agent_name == "ppo_framestack":
            # FrameStack needs VecFrameStack wrapper
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
            eval_env = create_environment(exp_config.base_config, exp_config.pomdp_params)
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecFrameStack(eval_env, n_stack=exp_config.base_config.n_stack)
        elif agent_name == "ppo_recurrent":
            # Recurrent agent uses raw environment (no stacking, LSTM handles temporal info)
            # Important: RecurrentPPO manages LSTM state internally
            eval_env = create_environment(exp_config.base_config, exp_config.pomdp_params)
        else:
            # Standard environment for baseline PPO
            eval_env = env  # Reuse standard environment
        
        result = run_agent(model, eval_env, experiment_dir=experiment_dir, agent_name=agent_name)
        results.append(result)
    
    return results


@app.command()
def main(
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        "-s",
        help=f"Predefined scenario to run. Available: {', '.join(list_scenarios())}",
    ),
    config_name: str = typer.Option(
        "default",
        "--config",
        "-c",
        help="Base configuration to use",
    ),
    skip_training: Optional[str] = typer.Option(
        None,
        "--skip-training",
        help="Skip training for specified agents (comma-separated) or all if no argument provided. Use 'all' to skip all agents.",
    ),
    total_timesteps: int = typer.Option(
        50000,
        "--timesteps",
        "-t",
        help="Total timesteps for RL training",
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
        help="Fraction of true I and R observed (1.0=full, 0.3=COVID-realistic). Custom mode only.",
    ),
    # Future parameters can be easily added here:
    # delay: int = typer.Option(0, "--delay", help="Observation delay in days"),
    # noise: float = typer.Option(0.0, "--noise", help="Observation noise std"),
):
    """
    Run epidemic control experiment.
    
    By default, trains all RL agents. Use --skip-training to load existing weights.
    
    Examples:
        python main.py --scenario mdp
        python main.py --scenario mdp --skip-training all
        python main.py --scenario mdp --skip-training ppo_baseline
        python main.py --no-exposed
    """
    
    # Parse skip_training argument
    agents_to_skip = set()
    if skip_training is not None:
        if skip_training.lower() == "all" or skip_training == "":
            agents_to_skip = {"all"}
        else:
            agents_to_skip = set(agent.strip() for agent in skip_training.split(","))
    
    # Mode 1: Predefined scenario
    if scenario:
        print(f"Running predefined scenario: {scenario}")
        scenario_config = get_scenario(scenario)
        
        base_config = get_config(config_name)
        
        exp_config = ExperimentConfig(
            base_config=base_config,
            pomdp_params=scenario_config["pomdp_params"],
            scenario_name=scenario,
            is_custom=False,
            target_agents=scenario_config["target_agents"],
            train_rl=True,  # This field is deprecated but kept for backward compatibility
            total_timesteps=total_timesteps,
        )
        
        experiment_dir = ExperimentDirectory(exp_config)
        print(f"Results will be saved to: {experiment_dir.root}")
        
    # Mode 2: Custom configuration
    else:
        print("Running custom experiment")
        
        base_config = get_config(config_name)
        
        # Build POMDP parameters from CLI flags
        pomdp_params = {
            "include_exposed": not no_exposed,
            "detection_rate": detection_rate,
            # Future parameters:
            # "delay": delay,
            # "noise_std": noise,
        }

        # Generate scenario name
        scenario_name = create_custom_scenario_name(pomdp_params)
        
        exp_config = ExperimentConfig(
            base_config=base_config,
            pomdp_params=pomdp_params,
            scenario_name=scenario_name,
            is_custom=True,
            target_agents=TARGET_AGENTS.copy(),
            train_rl=True,  # This field is deprecated but kept for backward compatibility
            total_timesteps=total_timesteps,
        )
        
        experiment_dir = ExperimentDirectory(exp_config)
        print(f"Results will be saved to: {experiment_dir.root}")
    
    # Save experiment configuration
    experiment_dir.save_config()
    
    # Prepare RL agents (train or load based on skip list)
    rl_models = prepare_rl_agents(exp_config, experiment_dir, agents_to_skip)
    
    # Run evaluation
    results = run_evaluation(exp_config, experiment_dir, rl_models)
    
    # Create comparison plot
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOT")
    print("=" * 80)
    comparison_path = experiment_dir.get_plot_path("comparison_all_agents.png")
    plot_all_results(results, save_path=str(comparison_path))
    
    # Save summary
    experiment_dir.save_summary(results)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nScenario: {exp_config.scenario_name}")
    print(f"Results directory: {experiment_dir.root}")
    print(f"\nEvaluated {len(results)} agents:")
    for result in results:
        print(f"  - {result.agent_name}: "
              f"Peak I = {result.peak_infected:.1f}, "
              f"Total Reward = {result.total_reward:.2f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    app()
