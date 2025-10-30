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

To run the simulation with different intervention strategies, use the following command:

```sh
python main.py
```

This will run simulations comparing four intervention levels (No, Mild, Moderate, Severe) and display plots showing the curves for each strategy.

## Running Tests

To run the tests:

```sh
python -m pytest
```
