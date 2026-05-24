from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOBaselineConfig:
    """PPO baseline (no memory) hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.2
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    early_stop_patience: int = 60
    early_stop_min_evals: int = 40


@dataclass
class PPOFrameStackConfig:
    """PPO with FrameStack hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.2
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    early_stop_patience: int = 60
    early_stop_min_evals: int = 40


@dataclass
class PPORecurrentConfig:
    """RecurrentPPO (LSTM) hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 5
    ent_coef: float = 0.2
    lstm_hidden_size: int = 128
    early_stop_patience: int = 30
    early_stop_min_evals: int = 20


@dataclass
class DQNConfig:
    """DQN hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 32
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.2
    target_update_interval: int = 1
    tau: float = 0.03
    gradient_steps: int = 2
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    early_stop_patience: int = 60
    early_stop_min_evals: int = 40


@dataclass
class Config:
    """Master configuration for epidemic control experiments.

    SEIR model, reward, evaluation, and scale parameters are top-level.
    Agent-specific RL hyperparameters live in nested dataclasses:
    ``ppo_baseline``, ``ppo_framestack``, ``ppo_recurrent``, ``dqn``.
    """

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

    # Shared training parameters
    n_envs: int = 4  # Number of parallel environments for training
    lr_decay_steps: int = 2_000_000  # LR decays to 10% of peak over this many steps, then holds

    # Evaluation and early stopping (shared threshold)
    eval_freq: int = 10_000             # Evaluate every N total timesteps
    n_eval_episodes: int = 20           # Episodes per evaluation (higher = less noise in estimate)
    early_stop_min_delta: float = 0.02  # Min raw reward delta (vs last significant best) to count as improvement

    # Experiment scale
    total_timesteps: int = 3_000_000  # Maximum timesteps for RL training
    num_training_seeds: int = 10  # Number of independent training seeds per agent

    # Agent-specific hyperparameters
    ppo_baseline: PPOBaselineConfig = field(default_factory=PPOBaselineConfig)
    ppo_framestack: PPOFrameStackConfig = field(default_factory=PPOFrameStackConfig)
    ppo_recurrent: PPORecurrentConfig = field(default_factory=PPORecurrentConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
