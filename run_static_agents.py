"""
Script for running static agents to verify the epidemic model.

This script runs all four StaticAgent configurations (NO, MILD, MODERATE, SEVERE)
and creates a single comparison plot showing SEIR curves for each intervention level.
Used for sanity-checking the epidemic model, not for experiments.

Results are saved to experiments/static_agents/{timestamp}/
with only config.json and static_agents_comparison.png (no subdirectories).
"""

import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.agents import StaticAgent, InterventionAction
from src.config import Config
from src.env import EpidemicEnv, SimulationResult


def run_static_agent(agent: StaticAgent, env: EpidemicEnv) -> SimulationResult:
    """
    Run a single static agent simulation.
    
    Args:
        agent: StaticAgent to evaluate.
        env: EpidemicEnv to run simulation in.
        
    Returns:
        SimulationResult with complete trajectory.
    """
    obs, _ = env.reset(seed=42)
    done = False
    
    # Extract initial state
    S_init, E_init, I_init, R_init = obs
    
    all_S = [S_init]
    all_E = [E_init]
    all_I = [I_init]
    all_R = [R_init]
    
    actions_taken = []
    timesteps = []
    rewards = []
    observations = []
    
    current_timestep = 0
    
    while not done:
        observations.append(obs)
        timesteps.append(current_timestep)
        
        action_idx, _ = agent.predict(obs, deterministic=True)
        
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
        
        action_enum = env.action_map[action_idx]
        actions_taken.append(action_enum)
        rewards.append(reward)
    
    t = np.arange(len(all_S))
    
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
    )
    
    return result


def plot_static_agents_comparison(results: List[SimulationResult], save_path: str) -> None:
    """
    Create comparison plot for all static agents.
    
    Args:
        results: List of SimulationResult objects (one per static agent).
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {"S": "blue", "E": "orange", "I": "red", "R": "green"}
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Plot SEIR curves
        ax.plot(result.t, result.S, color=colors["S"], label="Susceptible (S)", linewidth=2)
        ax.plot(result.t, result.E, color=colors["E"], label="Exposed (E)", linewidth=2)
        ax.plot(result.t, result.I, color=colors["I"], label="Infected (I)", linewidth=2)
        ax.plot(result.t, result.R, color=colors["R"], label="Recovered (R)", linewidth=2)
        
        # Title and labels
        ax.set_title(result.agent_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Number of people")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        info_text = f"Peak I: {result.peak_infected:.1f}\n"
        info_text += f"Total infected: {result.total_infected:.1f}\n"
        info_text += f"Total Reward: {result.total_reward:.2f}"
        
        ax.text(
            0.98,
            0.98,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )
        
        # Add action decision lines
        for timestep in result.timesteps[1:]:
            ax.axvline(timestep, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Static agents comparison plot saved to: {save_path}")


def main():
    """Main function to run static agents verification."""
    print("=" * 80)
    print("STATIC AGENTS VERIFICATION")
    print("=" * 80)
    print("\nRunning all four static intervention strategies to verify epidemic model...\n")
    
    # Load default configuration
    config = Config()
    
    # Create output directory under experiments/static_agents/
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("experiments", "static_agents", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define static agents
    static_agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.MILD),
        StaticAgent(InterventionAction.MODERATE),
        StaticAgent(InterventionAction.SEVERE),
    ]
    
    # Run simulations
    results = []
    for agent in static_agents:
        env = EpidemicEnv(config)
        print(f"Running simulation: {agent.__class__.__name__} - "
              f"{list(InterventionAction)[agent.action_idx].name}")
        result = run_static_agent(agent, env)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Agent':<30} {'Peak I':<12} {'Total Infected':<15} {'Total Reward':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result.agent_name:<30} {result.peak_infected:<12.1f} "
              f"{result.total_infected:<15.1f} {result.total_reward:<12.2f}")
    
    # Save comparison plot
    plot_path = os.path.join(output_dir, "static_agents_comparison.png")
    plot_static_agents_comparison(results, plot_path)
    
    # Save configuration for reference
    import json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        config_dict = {
            "N": config.N,
            "E0": config.E0,
            "I0": config.I0,
            "beta_0": config.beta_0,
            "sigma": config.sigma,
            "gamma": config.gamma,
            "stochastic": config.stochastic,
            "days": config.days,
            "action_interval": config.action_interval,
            "w_I": config.w_I,
            "w_S": config.w_S,
        }
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
