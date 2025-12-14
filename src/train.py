import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def train_ppo_agent(env_cls, config, log_dir="logs/ppo", total_timesteps=50000):
    """
    Trains a PPO agent in the given environment.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = env_cls(config)
    monitor_path = os.path.join(log_dir, "training")
    env = Monitor(env, monitor_path)

    # Create agent
    # Use MlpPolicy as we have vector input
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Train
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.")

    # Save
    save_path = os.path.join(log_dir, "ppo_model")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return save_path
