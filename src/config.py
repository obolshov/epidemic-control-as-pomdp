class DefaultConfig:
    def __init__(self):
        # Population and epidemic parameters
        self.N = 1000  # Total population
        self.I0 = 1  # Initial infected
        self.R0 = 0  # Initial recovered

        # SIR model parameters
        self.beta_0 = 0.4  # Base transmission rate
        self.gamma = 0.1  # Recovery rate

        # Simulation settings
        self.days = 160  # Simulation days
        self.action_interval = 7  # Days between decisions

        # Reward function parameters
        self.w_I = 1.0
        self.w_S = 0.5


def get_config(name: str):
    if name == "default":
        return DefaultConfig()
    else:
        raise ValueError(f"Unknown config: {name}")
