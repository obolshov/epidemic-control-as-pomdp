import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.simulation import Simulation, calculate_reward
from src.agents import StaticAgent, RandomAgent, MyopicMaximizer, InterventionAction
from src.sir import EpidemicState
from src.config import get_config
import os


def calculate_reward_components(
    I_t: float, I_t1: float, action: InterventionAction, action_interval: int
):
    """
    Calculate reward components separately for visualization.
    """
    infection_ratio = I_t1 / I_t if I_t > 0 else 0.0
    log_infection_ratio = (
        np.log(infection_ratio) if I_t > 0 and infection_ratio > 0 else 0.0
    )
    infection_penalty = max(0.0, log_infection_ratio) if I_t > 0 else 0.0

    reward = calculate_reward(I_t, I_t1, action, action_interval)

    return (
        infection_penalty,
        reward,
        infection_ratio,
        log_infection_ratio,
    )


def run_simulation_with_reward_tracking(simulation: Simulation):
    """
    Runs simulation and tracks reward components for each action.
    """
    current_state = simulation.initial_state

    all_S = [current_state.S]
    all_I = [current_state.I]
    all_R = [current_state.R]

    actions_taken = []
    action_timesteps = []
    rewards = []
    max_log_infection_ratios = []
    infection_ratios = []
    log_infection_ratios = []
    action_states = []

    current_day = 0

    from src.sir import SIR

    sir = SIR()

    while current_day < simulation.total_days:
        # Store state at decision point
        action_states.append(current_state)

        action = simulation.agent.select_action(current_state)
        actions_taken.append(action)
        action_timesteps.append(current_day)

        beta = simulation.apply_action_to_beta(action)

        days_to_simulate = min(
            simulation.action_interval, simulation.total_days - current_day
        )

        I_before = current_state.I

        S, I, R = sir.run_interval(
            current_state, beta, simulation.gamma, days_to_simulate
        )

        all_S.extend(S[1:])
        all_I.extend(I[1:])
        all_R.extend(R[1:])

        current_state = EpidemicState(N=current_state.N, S=S[-1], I=I[-1], R=R[-1])

        I_after = current_state.I

        inf_penalty, reward, inf_ratio, log_inf_ratio = calculate_reward_components(
            I_before, I_after, action, simulation.action_interval
        )

        rewards.append(reward)
        max_log_infection_ratios.append(inf_penalty)
        infection_ratios.append(inf_ratio)
        log_infection_ratios.append(log_inf_ratio)

        current_day += days_to_simulate

    t = np.arange(len(all_S))

    return {
        "t": t,
        "S": np.array(all_S),
        "I": np.array(all_I),
        "R": np.array(all_R),
        "actions": actions_taken,
        "action_timesteps": action_timesteps,
        "rewards": rewards,
        "max_log_infection_ratios": max_log_infection_ratios,
        "infection_ratios": infection_ratios,
        "log_infection_ratios": log_infection_ratios,
        "action_states": action_states,
    }


if __name__ == "__main__":
    config = get_config("default")

    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.SEVERE),
        MyopicMaximizer(config),
    ]

    agent_names = ["No action", "Severe action", "Myopic Maximizer"]

    results = []
    for agent in agents:
        simulation = Simulation(agent=agent, config=config)
        result = run_simulation_with_reward_tracking(simulation)
        results.append(result)

    all_values = []
    for result in results:
        max_log_infection_ratios = result["max_log_infection_ratios"]
        rewards = result["rewards"]
        infection_ratios = result["infection_ratios"]
        log_infection_ratios = result["log_infection_ratios"]
        all_values.extend(max_log_infection_ratios)
        all_values.extend(rewards)
        all_values.extend(infection_ratios)
        all_values.extend(log_infection_ratios)

    global_min = min(all_values)
    global_max = max(all_values)
    padding = (global_max - global_min) * 0.1  # 10% padding

    # Create figure with subplots for all agents (3 rows x 2 columns)
    fig, axes = plt.subplots(1, len(agents), figsize=(7 * len(agents), 6))
    axes = axes.flatten()

    for idx, (result, agent_name) in enumerate(zip(results, agent_names)):
        ax = axes[idx]

        # Extract timesteps and reward components
        action_timesteps = result["action_timesteps"]
        max_log_infection_ratios = result["max_log_infection_ratios"]
        rewards = result["rewards"]
        infection_ratios = result["infection_ratios"]
        log_infection_ratios = result["log_infection_ratios"]

        # Plot the three components
        ax.plot(
            action_timesteps,
            rewards,
            "g-",
            label="Total Reward",
            linewidth=2,
            marker="^",
            markersize=4,
        )
        ax.plot(
            action_timesteps,
            infection_ratios,
            "m-",
            label="I_t1 / I_t",
            linewidth=2,
            marker="d",
            markersize=4,
        )
        ax.plot(
            action_timesteps,
            log_infection_ratios,
            "c-",
            label="log(I_t1 / I_t)",
            linewidth=2,
            marker="v",
            markersize=4,
        )
        ax.plot(
            action_timesteps,
            max_log_infection_ratios,
            "r-",
            label="max(0, log(I_t1 / I_t))",
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # Add zero line
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=1)

        # Set same y-axis limits for all subplots
        ax.set_ylim(global_min - padding, global_max + padding)

        # Set y-axis tick step to 1
        ax.yaxis.set_major_locator(MultipleLocator(1))

        title = agent_name

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Reward component value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add stats in a box

        total_infection_ratios = sum(infection_ratios)
        total_log_infection_ratios = sum(log_infection_ratios)
        total_max_log_infection_ratios = sum(max_log_infection_ratios)
        total_reward = sum(rewards)

        info_text = f"I_t1 / I_t: {total_infection_ratios:.2f}\n"
        info_text += f"log(I_t1 / I_t): {total_log_infection_ratios:.2f}\n"
        info_text += f"max(0, log(I_t1 / I_t)): {total_max_log_infection_ratios:.2f}\n"
        info_text += f"Total Reward: {total_reward:.2f}"

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

    R_0 = config.beta_0 / config.gamma

    plt.tight_layout()
    plt.suptitle(
        f"Reward Components Over Time, No stringency penalty, R_0 = {R_0:.2f}",
        fontsize=14,
        fontweight="bold",
        y=1.001,
    )

    os.makedirs("results", exist_ok=True)
    save_path = "results/reward_components_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
