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
    w_switch: float = 0.05  # Weight on quadratic distance switching penalty

    # ThresholdAgent parameters
    thresholds: List[float] = field(default_factory=lambda: [0.01, 0.04, 0.09])

    # Frame stacking for temporal awareness
    n_stack: int = 30  # Number of consecutive observations to stack

    # Recurrent agent parameters (LSTM)
    lstm_hidden_size: int = 128

    # Training hyperparameters
    n_envs: int = 4  # Number of parallel environments for training
    learning_rate: float = 3e-4  # Peak lr for all RL agents (with linear schedule)
    lr_decay_steps: int = 2_000_000  # LR decays to 10% of peak over this many steps, then holds
    ent_coef: float = 0.2  # Entropy bonus for all RL agents
    recurrent_n_steps: int = 256  # Rollout length per env for RecurrentPPO
    recurrent_batch_size: int = 64  # Mini-batch size for RecurrentPPO
    recurrent_n_epochs: int = 5  # Optimization epochs per rollout for RecurrentPPO

    # Experiment scale
    total_timesteps: int = 3_000_000  # Maximum timesteps for RL training
    num_training_seeds: int = 10  # Number of independent training seeds per agent

    # Evaluation and early stopping hyperparameters
    # Stopping triggers when `best_mean_reward` fails to improve by at least `early_stop_min_delta`
    # for `early_stop_patience` consecutive evals after `early_stop_min_evals` warmup evals.
    # Guaranteed min training = (min_evals + patience) * eval_freq.
    eval_freq: int = 10_000             # Evaluate every N total timesteps
    early_stop_patience: int = 60       # Evals without significant improvement before stopping (600k steps)
    early_stop_min_evals: int = 40      # Warmup evals before stopping can trigger (400k steps)
    early_stop_min_delta: float = 0.02  # Min raw reward delta (vs last significant best) to count as improvement
    n_eval_episodes: int = 20           # Episodes per evaluation (higher = less noise in estimate)

    # RecurrentPPO overrides: LSTM converges faster, so patience/min_evals are halved.
    recurrent_early_stop_patience: int = 30   # 300k steps
    recurrent_early_stop_min_evals: int = 20  # 200k steps
