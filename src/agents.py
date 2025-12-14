import random
from src.sir import EpidemicState, run_sir
from enum import Enum
from src.config import DefaultConfig


class InterventionAction(Enum):
    NO = 1.0
    MILD = 0.75
    MODERATE = 0.5
    SEVERE = 0.25


class Agent:
    """
    Base class for epidemic control agents.
    Compatible with Stable Baselines3 interface.
    """

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        :param observation: np.ndarray [S, I, R]
        :return: (action_index, state)
        """
        raise NotImplementedError("Subclasses must implement predict")


class StaticAgent(Agent):
    def __init__(self, action: InterventionAction):
        self.action_idx = list(InterventionAction).index(action)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return self.action_idx, state


class RandomAgent(Agent):
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        action_idx = random.choice(range(len(InterventionAction)))
        return action_idx, state


class MyopicMaximizer(Agent):
    """
    Agent that selects action by maximizing immediate reward.
    """

    def __init__(self, config: DefaultConfig):
        self.config = config
        self.step_counter = 0
        self.action_values = list(InterventionAction)

        # Setup logging
        import os

        self.log_path = "logs/myopic_maximizer_debug.txt"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("MyopicMaximizer Debug Log\n")
            header = (
                f"{'Step':<8}"
                + "".join(f"{a.name:<15}" for a in InterventionAction)
                + f"{'Selected':<15}\n"
            )
            f.write("=" * len(header) + "\n")
            f.write(header)
            f.write("-" * len(header) + "\n")

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        from src.env import calculate_reward

        S, I, R = observation
        # Reconstruct state for internal simulation
        # Note: We assume N = S + I + R
        N = S + I + R
        current_state = EpidemicState(N=N, S=S, I=I, R=R)

        best_action_idx = 0
        best_reward = float("-inf")
        action_rewards = []

        for idx, action in enumerate(self.action_values):
            beta = self.config.beta_0 * action.value

            # Simulate one interval
            # We only need the final state of the interval to calculate reward
            _, I_sim, _ = run_sir(
                current_state, beta, self.config.gamma, self.config.action_interval
            )

            reward = calculate_reward(I_sim[-1], action, self.config)
            action_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_action_idx = idx

        # Logging
        with open(self.log_path, "a", encoding="utf-8") as f:
            row = f"{self.step_counter:<8}"
            for r in action_rewards:
                row += f"{r:<15.4f}"
            selected_name = self.action_values[best_action_idx].name
            row += f"{selected_name:<15}\n"
            f.write(row)

        self.step_counter += 1
        return best_action_idx, state
