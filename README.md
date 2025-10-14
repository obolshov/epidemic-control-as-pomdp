# Epidemic Control as Partiall Observable Domain

## Prerequisites

*   Python 3.13
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

To run the SIR model simulation execute the following command in your terminal:

```sh
python sir_model.py
```

This will run the simulation with the default parameters defined in the script and display a plot showing the number of susceptible, infected, and recovered individuals over time.

## Running Tests

The project includes tests to verify the correctness of the SIR model implementation. To run the tests, execute the following command:

```sh
pytest
```

or

```sh
python -m pytest
```

