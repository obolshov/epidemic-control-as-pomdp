from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Population and epidemic parameters
    N: int = 100_000 # Total population size
    E0: int = 200 # Initial number of exposed individuals
    I0: int = 50 # Initial number of infected individuals

    # SEIR model parameters
    beta_0: float = 0.4   # Base transmission rate
    sigma: float = 0.2    # Incubation rate (1 / incubation_period)
    gamma: float = 0.1    # Recovery rate

    # Simulation settings
    days: int = 300 # Total simulation length in days
    action_interval: int = 5  # Days between decisions

    # Reward function parameters
    w_I: float = 10.0   # Weight on infection penalty
    w_S: float = 0.1    # Weight on stringency penalty

    # ThresholdAgent parameters
    thresholds: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.09])

    # Frame stacking for temporal awareness
    n_stack: int = 10  # Number of consecutive observations to stack

    # Recurrent agent parameters (LSTM)
    lstm_hidden_size: int = 128
    n_lstm_layers: int = 1
