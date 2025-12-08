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
