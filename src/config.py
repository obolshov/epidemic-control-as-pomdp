from typing import List


class DefaultConfig:
    def __init__(self):
        # Population and epidemic parameters
        self.N = 100000  # Total population
        self.E0 = 200  # Initial exposed
        self.I0 = 50  # Initial infected

        # SEIR model parameters
        self.beta_0 = 0.4  # Base transmission rate
        self.sigma = 0.2  # Incubation rate (1/incubation_period)
        self.gamma = 0.1  # Recovery rate

        # Simulation settings
        self.days = 300  # Simulation days
        self.action_interval = 5  # Days between decisions

        # Reward function parameters
        self.w_I = 10
        self.w_S = 0.1

        # POMDP settings
        self.include_exposed = True  # If False, E compartment is masked from observations
        self.detection_rate: float = 1.0  # Fraction of true I and R observed; 1.0 = full, 0.3 = COVID-realistic

        # ThresholdAgent parameters
        self.thresholds: List[float] = [0.01, 0.05, 0.09]  # Fraction of infected population

        # Frame stacking for temporal awareness
        self.n_stack = 10  # Number of consecutive observations to stack

        # Recurrent agent parameters (LSTM)
        self.lstm_hidden_size = 128  # Hidden state size for LSTM
        self.n_lstm_layers = 1  # Number of LSTM layers


def get_config(name: str):
    if name == "default":
        return DefaultConfig()
    else:
        raise ValueError(f"Unknown config: {name}")
