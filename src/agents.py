import random
from src.sir import EpidemicState, SIR
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
    """

    def select_action(self, state: EpidemicState) -> InterventionAction:
        raise NotImplementedError("Subclasses must implement select_action")


class StaticAgent(Agent):
    def __init__(self, action: InterventionAction):
        self.action = action

    def select_action(self, state: EpidemicState) -> InterventionAction:
        return self.action


class RandomAgent(Agent):
    def select_action(self, state: EpidemicState) -> InterventionAction:
        return random.choice(list(InterventionAction))


class MyopicMaximizer(Agent):
    """
    Agent that selects action by maximizing immediate reward.
    For each possible action, simulates one step forward and calculates reward,
    then returns the action with the highest reward.
    """

    def __init__(self, config: DefaultConfig):
        self.config = config
        self.sir = SIR()
        self.step_counter = 0

        import os

        self.log_path = "logs/myopic_maximizer_debug.txt"

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("MyopicMaximizer Debug Log - Reward Values for Each Action\n")
            f.write("=" * 80 + "\n\n")
            # Write header
            header = f"{'Step':<8}"
            for action in InterventionAction:
                header += f"{action.name:<15}"
            header += f"{'Selected':<15}\n"
            f.write(header)
            f.write("-" * 80 + "\n")

    def select_action(self, state: EpidemicState) -> InterventionAction:
        from src.simulation import calculate_reward

        best_action = None
        best_reward = float("-inf")
        action_rewards = {}

        for action in InterventionAction:
            beta = self.config.beta_0 * action.value
            I_before = state.I

            S, I, R = self.sir.run_interval(
                state, beta, self.config.gamma, self.config.action_interval
            )

            I_after = I[-1]

            reward = calculate_reward(I_before, I_after, action, self.config)
            action_rewards[action] = reward

            if reward > best_reward:
                best_reward = reward
                best_action = action

        # Log rewards if logging enabled
        with open(self.log_path, "a", encoding="utf-8") as f:
            row = f"{self.step_counter:<8}"
            for action in InterventionAction:
                row += f"{action_rewards[action]:<15.4f}"
            row += f"{best_action.name:<15}\n"
            f.write(row)

        self.step_counter += 1

        return best_action
