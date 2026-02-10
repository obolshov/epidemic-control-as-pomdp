import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from typing import Type
import gymnasium as gym
from src.config import DefaultConfig
from src.wrappers import EpidemicObservationWrapper


def train_ppo_agent(
    env_cls: Type[gym.Env],
    config: DefaultConfig,
    experiment_dir: "ExperimentDirectory",
    agent_name: str,
    total_timesteps: int,
    pomdp_params: dict = None,
) -> PPO:
    """
    Train a PPO agent and save weights to experiment directory.
    
    Args:
        env_cls: Environment class to instantiate.
        config: Configuration for the environment.
        experiment_dir: ExperimentDirectory for saving outputs.
        agent_name: Name of the agent (e.g., "ppo_baseline").
        total_timesteps: Number of timesteps to train.
        pomdp_params: POMDP parameters for applying wrappers.
        
    Returns:
        Trained PPO model.
    """
    if pomdp_params is None:
        pomdp_params = {}
    
    # Create environment
    env = env_cls(config)
    
    # Apply POMDP wrapper if partial observability is enabled
    if not pomdp_params.get("include_exposed", True):
        env = EpidemicObservationWrapper(env, include_exposed=False)
    
    # Apply frame stacking for ppo_framestack agent
    if agent_name == "ppo_framestack":
        # Create monitor directory
        monitor_dir = experiment_dir.tensorboard_dir / agent_name
        monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Wrap in DummyVecEnv first (required by VecFrameStack)
        env = DummyVecEnv([lambda: env])
        # Add VecMonitor for training metrics (must be before VecFrameStack)
        env = VecMonitor(env, filename=str(monitor_dir))
        # Apply frame stacking
        env = VecFrameStack(env, n_stack=config.n_stack)
        print(f"Applied VecMonitor + VecFrameStack with n_stack={config.n_stack}")
        print(f"Observation space changed to: {env.observation_space}")
    else:
        # Standard Monitor wrapper for non-framestack agents
        monitor_dir = experiment_dir.tensorboard_dir / agent_name
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_dir))

    # Initialize PPO model with TensorBoard logging
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(experiment_dir.tensorboard_dir))

    print(f"Training {agent_name} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=agent_name)
    print("Training finished.")

    # Save model weights
    weight_path = experiment_dir.get_weight_path(agent_name)
    model.save(str(weight_path.with_suffix("")))  # Remove .zip as SB3 adds it
    print(f"Model weights saved to {weight_path}")
    
    return model
