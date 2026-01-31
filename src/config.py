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
        self.days = 200  # Simulation days
        self.action_interval = 5  # Days between decisions

        # Reward function parameters
        self.w_I = 10
        self.w_S = 0.1


def get_config(name: str):
    if name == "default":
        return DefaultConfig()
    else:
        raise ValueError(f"Unknown config: {name}")
