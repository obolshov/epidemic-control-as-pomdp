import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from typing import Type
import gymnasium as gym
from src.config import DefaultConfig


def train_ppo_agent(
    env_cls: Type[gym.Env], config: DefaultConfig, log_dir: str, total_timesteps: int
) -> None:
    os.makedirs(log_dir, exist_ok=True)

    env = env_cls(config)
    env = Monitor(env, log_dir)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.")

    save_path = os.path.join(log_dir, "ppo_model")
    model.save(save_path)
    print(f"Model saved to {save_path}")
