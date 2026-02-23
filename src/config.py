from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Population and epidemic parameters
    N: int = 200_000 # Total population size
    E0: int = 200 # Initial number of exposed individuals
    I0: int = 50 # Initial number of infected individuals

    # SEIR model parameters
    beta_0: float = 0.4   # Base transmission rate
    sigma: float = 0.2    # Incubation rate (1 / incubation_period)
    gamma: float = 0.1    # Recovery rate
    stochastic: bool = True  # If True, use Binomial transitions; if False, use deterministic ODE

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

    # Training hyperparameters
    n_envs: int = 4  # Number of parallel environments for training
    recurrent_n_steps: int = 256  # Rollout length per env for RecurrentPPO
    recurrent_batch_size: int = 64  # Mini-batch size for RecurrentPPO
    recurrent_n_epochs: int = 5  # Optimization epochs per rollout for RecurrentPPO
    ent_coef: float = 0.01  # Entropy bonus for exploration

    # Early stopping hyperparameters (via EvalCallback)
    # Stopping triggers when no improvement for `early_stop_patience` consecutive evals
    # Each eval = eval_freq timesteps (default 5000). Min warmup = min_evals * eval_freq.
    early_stop_patience: int = 5   # Evals without improvement before stopping (25k steps)
    early_stop_min_evals: int = 3  # Minimum evals before stopping can trigger (15k warmup)
