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

## Usage

To run the simulation with different agents, use the following command:

```sh
python main.py
```

## Output

**Logs**: Simulation results are saved to `logs/` directory as text files (one per agent) containing timesteps, observations, actions, and rewards.

**Plots**: 
- Individual agent plots: `results/{agent_name}.png` - SIR curves for each agent
- Comparison plot: `results/all_results.png` - side-by-side comparison of all agents

## Running Tests

To run the tests:

```sh
python -m pytest
```
