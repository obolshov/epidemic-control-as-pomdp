# Epidemic Control as Partially Observable Domain

## Prerequisites

*   Python 3.13+
*   pip

## Installation

1.  Clone the repository to your local machine:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required Python packages using `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```

## Agents

The simulation includes the following agents:

- **StaticAgent** - Always takes the same intervention action (NO, MILD, MODERATE, or SEVERE)
- **RandomAgent** - Randomly selects intervention actions
- **MyopicMaximizer** - Selects actions that maximize immediate reward
- **PPO** - Proximal Policy Optimization agent (requires training with `--train_ppo`)

## Usage

### Basic Usage

To run the simulation with different agents (StaticAgent, RandomAgent, MyopicMaximizer, and PPO if available):

```sh
python main.py
```

### Training PPO Agent

To train a PPO agent before running simulations:

```sh
python main.py --train_ppo
```

This will:
- Train the PPO agent for 50,000 timesteps
- Save the trained model to `logs/ppo/ppo_model.zip`
- Generate learning curve plots in `results/ppo_learning_curve_*.png`
- Log training metrics to TensorBoard (in `logs/ppo/`)

**Note**: If a trained PPO model already exists at `logs/ppo/ppo_model.zip`, it will be automatically loaded and included in the simulation without needing to retrain.

### Configuration

You can specify a different configuration:

```sh
python main.py --config <config_name>
```

## Output

### Logs

- **Simulation logs**: `logs/{agent_name}.txt` - Text files containing timesteps, observations, actions, and rewards for each agent
- **PPO training logs**: `logs/ppo/` - TensorBoard event files and training monitor CSV (use `tensorboard --logdir logs/ppo` to view)

### Plots

- **Individual agent plots**: `results/{agent_name}.png` - SIR curves for each agent
- **Comparison plot**: `results/all_results.png` - Side-by-side comparison of all agents
- **PPO learning curves**: `results/ppo_learning_curve_timesteps.png` and `results/ppo_learning_curve_episodes.png` - Training progress visualization (generated when using `--train_ppo`)

## Running Tests

To run the tests:

```sh
python -m pytest
```
