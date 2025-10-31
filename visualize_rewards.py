import numpy as np
import matplotlib.pyplot as plt
from src.simulation import Simulation
from src.agents import StaticAgent, InterventionAction
from src.sir import EpidemicState
from src.config import get_config


def calculate_reward_components(
    I_t: float,
    I_t1: float,
    action: InterventionAction,
    w_S: float,
):
    """
    Calculate reward components separately for visualization.
    """
    if I_t > 0:
        infection_penalty = max(0.0, np.log(I_t1 / I_t))
    else:
        infection_penalty = 0.0

    stringency_penalty = w_S * ((1 - action.value) ** 2)
    reward = -(infection_penalty + stringency_penalty)

    return infection_penalty, stringency_penalty, reward


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

    current_day = 0

    from src.sir import SIR

    sir = SIR()

    while current_day < simulation.total_days:
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

        inf_penalty, str_penalty, reward = calculate_reward_components(
            I_before,
            I_after,
            action,
            simulation.w_S,
        )

        rewards.append(reward)
        infection_penalties.append(inf_penalty)
        stringency_penalties.append(str_penalty)

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
    }


def plot_reward_components():
    """
    Visualize reward components (infection_penalty, stringency_penalty, reward)
    for all 4 actions.
    """
    config = get_config("default")

    config.beta_0 = 0.4
    config.gamma = 0.1

    static_agents = [StaticAgent(action) for action in InterventionAction]

    results = []
    for static_agent in static_agents:
        simulation = Simulation(agent=static_agent, config=config)
        result = run_simulation_with_reward_tracking(simulation)
        results.append(result)

    # Collect all values to find global min and max
    all_values = []
    for result in results:
        infection_penalties = result["infection_penalties"]
        stringency_penalties = result["stringency_penalties"]
        rewards = result["rewards"]
        all_values.extend(infection_penalties)
        all_values.extend(stringency_penalties)
        all_values.extend(rewards)

    global_min = min(all_values)
    global_max = max(all_values)
    padding = (global_max - global_min) * 0.1  # 10% padding

    # Create figure with 4 subplots, one for each action
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (result, action) in enumerate(zip(results, InterventionAction)):
        ax = axes[idx]

        # Extract timesteps and reward components
        action_timesteps = result["action_timesteps"]
        infection_penalties = result["infection_penalties"]
        stringency_penalties = result["stringency_penalties"]
        rewards = result["rewards"]

        # Plot the three components
        ax.plot(
            action_timesteps,
            infection_penalties,
            "r-",
            label="Infection Penalty",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax.plot(
            action_timesteps,
            stringency_penalties,
            "b-",
            label="Stringency Penalty",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        ax.plot(
            action_timesteps,
            rewards,
            "g-",
            label="Total Reward",
            linewidth=2,
            marker="^",
            markersize=4,
        )

        # Add zero line
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=1)

        # Set same y-axis limits for all subplots
        ax.set_ylim(global_min - padding, global_max + padding)

        title = f"Action: {action.name}\n(beta multiplier = {action.value})"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Reward component value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add stats in a box
        total_inf_penalty = sum(infection_penalties)
        total_str_penalty = sum(stringency_penalties)
        total_reward = sum(rewards)

        info_text = f"Total Infection P: {total_inf_penalty:.2f}\n"
        info_text += f"Total Stringency P: {total_str_penalty:.2f}\n"
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
        f"Reward Components Over Time for All Actions, R_0 = {R_0:.2f}",
        fontsize=14,
        fontweight="bold",
        y=1.001,
    )

    plt.show()


if __name__ == "__main__":
    plot_reward_components()
