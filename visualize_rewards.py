import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.env import EpidemicEnv, calculate_reward
from src.agents import StaticAgent, InterventionAction
from src.seir import EpidemicState
from src.config import Config
import os
from datetime import datetime


def calculate_reward_components(
    I_t: float, action: InterventionAction, config: Config, prev_action_idx: int = 0
):
    """
    Calculate reward components separately for visualization.

    Args:
        I_t: Infected count.
        action: Current action.
        config: Experiment config.
        prev_action_idx: Index of the previous action (for switching penalty).

    Returns:
        Tuple of (total_reward, infection_penalty, stringency_penalty, switching_penalty).
    """
    infection_ratio = (I_t / config.N) ** 2
    infection_penalty = config.w_I * max(0, infection_ratio)
    stringency_penalty = config.w_S * (1 - action.value)
    action_idx = list(InterventionAction).index(action)
    delta = action_idx - prev_action_idx
    switching_penalty = config.w_switch * delta ** 2

    reward = calculate_reward(I_t, action, config, prev_action_idx)

    return (
        reward,
        infection_penalty,
        stringency_penalty,
        switching_penalty,
    )


def run_simulation_with_reward_tracking(env: EpidemicEnv, agent):
    """
    Runs simulation and tracks reward components for each action.
    """
    obs, _ = env.reset()

    # Reconstruct initial state from observation (6-element: S, E, I, R, prev_action_idx, day_frac)
    S_curr, E_curr, I_curr, R_curr = obs[0], obs[1], obs[2], obs[3]
    current_state = EpidemicState(
        N=env.config.N, S=S_curr, E=E_curr, I=I_curr, R=R_curr
    )

    all_S = [current_state.S]
    all_E = [current_state.E]
    all_I = [current_state.I]
    all_R = [current_state.R]

    actions_taken = []
    action_timesteps = []
    rewards = []
    infection_penalties = []
    stringency_penalties = []
    switching_penalties = []
    action_states = []
    prev_action_idx = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Store state at decision point
        action_states.append(current_state)

        # Get action from agent
        action_idx, _ = agent.predict(obs, deterministic=True)
        action = env.action_map[action_idx]
        actions_taken.append(action)
        action_timesteps.append(env.current_day)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_idx)

        # Extract S, E, I, R trajectories from info
        S, E, I, R = info["S"], info["E"], info["I"], info["R"]

        # Add all intermediate states (skip first as it's already in all_S/I/R)
        all_S.extend(S[1:])
        all_E.extend(E[1:])
        all_I.extend(I[1:])
        all_R.extend(R[1:])

        # Update current state
        current_state = env.current_state

        # Calculate reward components
        reward_components = calculate_reward_components(
            current_state.I, action, env.config, prev_action_idx
        )
        _, infection_penalty, stringency_penalty, switching_penalty = reward_components

        rewards.append(reward)
        infection_penalties.append(infection_penalty)
        stringency_penalties.append(stringency_penalty)
        switching_penalties.append(switching_penalty)

        prev_action_idx = action_idx

    t = np.arange(len(all_S))

    return {
        "t": t,
        "S": np.array(all_S),
        "E": np.array(all_E),
        "I": np.array(all_I),
        "R": np.array(all_R),
        "actions": actions_taken,
        "action_timesteps": action_timesteps,
        "rewards": rewards,
        "infection_penalties": infection_penalties,
        "stringency_penalties": stringency_penalties,
        "switching_penalties": switching_penalties,
        "action_states": action_states,
    }


if __name__ == "__main__":
    config = Config()

    agents = [
        StaticAgent(InterventionAction.NO),
        StaticAgent(InterventionAction.SEVERE),
    ]

    agent_names = ["No action", "Severe action", "Myopic Maximizer"]

    results = []
    for agent in agents:
        env = EpidemicEnv(config=config)
        result = run_simulation_with_reward_tracking(env, agent)
        results.append(result)

    all_values = []
    for result in results:
        rewards = result["rewards"]
        infection_penalties = result["infection_penalties"]
        stringency_penalties = result["stringency_penalties"]
        switching_penalties = result["switching_penalties"]
        all_values.extend(rewards)
        all_values.extend(infection_penalties)
        all_values.extend(stringency_penalties)
        all_values.extend(switching_penalties)

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
        switching_penalties = result["switching_penalties"]

        # Plot the four components
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
        ax.plot(
            action_timesteps,
            switching_penalties,
            "r-",
            label="Switching Penalty",
            linewidth=2,
            marker="s",
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
        total_switching_penalties = sum(switching_penalties)
        total_reward = sum(rewards)

        info_text = f"Infection Penalty: {total_infection_penalties:.2f}\n"
        info_text += f"Stringency Penalty: {total_stringency_penalties:.2f}\n"
        info_text += f"Switching Penalty: {total_switching_penalties:.2f}\n"
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

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("experiments", "reward_visualization", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "reward_components_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

    plt.show()
