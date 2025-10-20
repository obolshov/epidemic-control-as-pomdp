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

To run the SIR model simulation with different intervention strategies:

```sh
python main.py
```

Or alternatively:

```sh
python -m src
```

This will run simulations comparing four intervention levels (No, Mild, Moderate, Severe) and display plots showing the SIR curves for each strategy.

## Running Tests

The project includes tests for the SIR model and intervention actions:

```sh
python -m pytest
```
