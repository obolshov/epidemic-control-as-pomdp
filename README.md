# Epidemic Control as Partially Observable Domain

## Prerequisites

*   Python 3.8+
*   conda

## Installation

1.  Clone the repository to your local machine:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install dependencies using conda:

    ```sh
    conda env create -f environment.yml
    conda activate epidemic-control-as-pomdp
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
