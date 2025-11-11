import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.simulation import Simulation, calculate_reward
from src.agents import StaticAgent, RandomAgent, MyopicMaximizer, InterventionAction
from src.sir import EpidemicState
from src.config import get_config, DefaultConfig
import os


def calculate_reward_components(
    I_t: float, action: InterventionAction, config: DefaultConfig
):
    """
    Calculate reward components separately for visualization.
    """
    infection_ratio = (I_t / config.N) - config.infection_peak
    infection_penalty = config.w_I * max(0, infection_ratio)
    stringency_penalty = config.w_S * (1 - action.value)

    reward = calculate_reward(I_t, action, config)

    return (
        reward,
        infection_penalty,
        stringency_penalty,
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
    infection_penalties = []
    stringency_penalties = []
    action_states = []

    current_day = 0

    from src.sir import SIR

    sir = SIR()

    while current_day < simulation.config.days:
        # Store state at decision point
        action_states.append(current_state)

        action = simulation.agent.select_action(current_state)
        actions_taken.append(action)
        action_timesteps.append(current_day)

        beta = simulation.apply_action_to_beta(action)

        days_to_simulate = min(
            simulation.config.action_interval, simulation.config.days - current_day
        )

        S, I, R = sir.run_interval(
            current_state, beta, simulation.config.gamma, days_to_simulate
        )

        all_S.extend(S[1:])
        all_I.extend(I[1:])
        all_R.extend(R[1:])

        current_state = EpidemicState(N=current_state.N, S=S[-1], I=I[-1], R=R[-1])

        reward, infection_penalty, stringency_penalty = calculate_reward_components(
            current_state.I, action, simulation.config
        )

        rewards.append(reward)
        infection_penalties.append(infection_penalty)
        stringency_penalties.append(stringency_penalty)

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
        "infection_penalties": infection_penalties,
        "stringency_penalties": stringency_penalties,
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
        rewards = result["rewards"]
        infection_penalties = result["infection_penalties"]
        stringency_penalties = result["stringency_penalties"]
        all_values.extend(rewards)
        all_values.extend(infection_penalties)
        all_values.extend(stringency_penalties)

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
        rewards = result["rewards"]
        infection_penalties = result["infection_penalties"]
        stringency_penalties = result["stringency_penalties"]

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
            infection_penalties,
            "m-",
            label="Infection Penalty",
            linewidth=2,
            marker="d",
            markersize=4,
        )
        ax.plot(
            action_timesteps,
            stringency_penalties,
            "c-",
            label="Stringency Penalty",
            linewidth=2,
            marker="v",
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

        total_infection_penalties = sum(infection_penalties)
        total_stringency_penalties = sum(stringency_penalties)
        total_reward = sum(rewards)

        info_text = f"Infection Penalty: {total_infection_penalties:.2f}\n"
        info_text += f"Stringency Penalty: {total_stringency_penalties:.2f}\n"
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
        f"Reward Components Over Time, R_0 = {R_0:.2f}",
        fontsize=14,
        fontweight="bold",
        y=1.001,
    )

    os.makedirs("results", exist_ok=True)
    save_path = "results/reward_components_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
