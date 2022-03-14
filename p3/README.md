# Peter Rasmussen, Programming Assignment 3

This Python 3 program trains the decision tree predictor across six datasets, using accuracy for the classification data and mean squared error for the regression data.

## Getting Started

The package is designed to be executed as a module from the command line. The user must specify the
 output directory as illustrated below. 

```shell
python -m path/to/p3  -i path/to/in_dir -o path/to/out_dir/ -k <folds> -v <val frac> -r <random state>
```

As an example:
```shell
python -m path/to/p3  -i path/to/in_dir -o path/to/out_dir/ -k 5 -v 0.2 -r 777
```

A summary of the command line arguments is below.

Positional arguments:

    -i, --src_dir               Input directory
    -o, --dst_dir               Output directory

Optional arguments:    

    -h, --help                 Show this help message and exit
    -k, --k_folds              Number of folds
    -v, --val_frac             Fraction of validation observations
    -r, --random_state         Provide pseudo-random seed

## Key parts of program
* run.py: Executes data loading, preprocessing, training, socring, and output creation.
* preprocessor.py: Preprocesses data: loading, imputation, and numeric data handling.

* classification_decision_tree.py
  * Employs gain ratio to split attributes at each node
  * Scored on the basis of accuracy
* regression_decision_tree.py
  * Employs mean squared error to split attributes at each node
  * Scored on the basis of mean squared error

## Features

* Performance metrics for each run for each dataset.
* Support for edited and condensed mode.
* Outputs provided as CSV files.
* Control over number of folds, validation fraction, and randomization.

## Output Files

See the outputs in the data/ folder or whichever folder the user specifies for outputs.

## Licensing

This project is licensed under the CC0 1.0 Universal license.
