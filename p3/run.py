"""Peter Rasmussen, Programming Assignment 3, run.py

The run function ingests user inputs to train decision tree predictors on six different datasets.

Outputs are saved to the user-specified directory.

"""

# Standard library imports
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import typing as t
import warnings

# Third party imports
import pandas as pd

# Local imports
from p3.algorithms.classification_entropy import get_feature_importances
from p3.algorithms.regression_error import make_thetas
from p3.nodes import ClassificationDecisionNode
from p3.nodes import RegressionDecisionNode
from p3.preprocessing.tree_preprocessor import TreePreprocessor
from p3.preprocessing.split import make_splits
from p3.trees import ClassificationDecisionTree
from p3.trees import RegressionDecisionTree

warnings.filterwarnings('ignore')

THETAS = [0, 0.02, 0.04, 0.06]

def run(
        src_dir: Path,
        dst_dir: Path,
        k_folds: int,
        val_frac: float,
        random_state: int,
):
    """
    Train and score a majority predictor across six datasets.
    :param src_dir: Input directory that provides each dataset and params files
    :param dst_dir: Output directory
    :param k_folds: Number of folds to partition the data into
    :param val_frac: Validation fraction of train-validation set
    :param random_state: Random number seed

    """
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    log_path = dir_path / "p3.log"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)

    logging.debug(f"Begin: src_dir={src_dir.name}, dst_dir={dst_dir.name}, seed={random_state}.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load data catalog
    with open(src_dir / "data_catalog.json", "r") as file:
        data_catalog = json.load(file)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate over each dataset
    classification_results: t.Union[list, pd.DataFrame] = []
    feature_importances: t.Union[list, pd.DataFrame] = []
    regression_results: t.Union[list, pd.DataFrame] = []
    #data_catalog = {k: v for k, v in data_catalog.items() if k in ["breast-cancer-wisconsin", "car", "house-votes-84"]}
    data_catalog = {k: v for k, v in data_catalog.items() if k in ["forestfires", "machine", "abalone"]}
    for dataset_name, dataset_meta in data_catalog.items():

        print(f"Dataset: {dataset_name}")
        logging.debug(f"Dataset: {dataset_name}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Preprocess data
        preprocessor = TreePreprocessor(dataset_name, dataset_meta, src_dir)
        preprocessor.load()
        preprocessor.identify_features_label_id()
        preprocessor.replace()
        preprocessor.log_transform()
        preprocessor.set_data_classes()
        preprocessor.impute()
        preprocessor.drop()
        preprocessor.shuffle()

        # Extract label and data classes
        label = preprocessor.label
        problem_class = dataset_meta["problem_class"]

        # Split data into train-test and validation sets
        data = preprocessor.data.copy()
        val_assignments = make_splits(data.sample(frac=1, random_state=random_state + 1), problem_class, label,
                                      k_folds=None, val_frac=val_frac)
        fold_assignments = make_splits(data, problem_class, label, k_folds=k_folds, val_frac=None)
        assignments = fold_assignments.join(val_assignments).rename(columns={"train": "train_test"})

        # Break off validation from train-test
        val_ix = assignments.copy().query("train_test==0").drop(axis=1, labels=["fold", "train_test"])
        val_data = val_ix.join(data)
        train_test_ix = assignments.copy().query("train_test==1").drop(axis=1, labels="train_test")
        train_test = train_test_ix.join(data)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterate over each fold and save results
        # for fold in range(2, k_folds + 1):
        for fold in list(range(1, k_folds + 1)):
            print(fold)
            test_mask = train_test["fold"] == fold
            test_data = train_test.copy()[test_mask].drop(axis=1, labels="fold")
            train_data = train_test.copy()[~test_mask].drop(axis=1, labels="fold")

            # Instantiate a copy of the preprocessor specific to this fold
            te_tr_preprocessor = deepcopy(preprocessor)
            te_tr_preprocessor.data = train_data

            # Discretize the train data
            te_tr_preprocessor.discretize()
            train = te_tr_preprocessor.data

            # Apply those discretizations to test and validation sets
            test = te_tr_preprocessor.discretize_nontrain(test_data)
            val = te_tr_preprocessor.discretize_nontrain(val_data)

            # Extract updated features and data classes
            features = te_tr_preprocessor.features
            data_classes = te_tr_preprocessor.data_classes

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Classification
            if problem_class == "classification":

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize root node and tree
                root = ClassificationDecisionNode(train, label, features, data_classes)
                tree = ClassificationDecisionTree(verbose=True)
                tree.add_node(root)
                tree.populate_tree_metadata()

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build decision tree
                tree.train()
                tree.set_height()

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Prune
                unpruned_height = tree.height
                tree.rules = tree.unpruned_rules = tree.make_rules(root)
                n_unpruned_nodes = len(tree.nodes)
                unpruned_feat_imps = get_feature_importances(tree, dataset_name, fold, pruned=False)

                tree.prune_nodes(val)
                tree.set_height()
                pruned_height = tree.height
                n_pruned_nodes = len(tree.nodes)
                pruned_feat_imps = get_feature_importances(tree, dataset_name, fold, pruned=True)
                tree.rules = tree.make_rules(root)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Predict
                pruned_pred = tree.predict(test)
                unpruned_pred = tree.predict(test, pruned=False)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Score
                pruned_score = tree.score(pruned_pred, test[label])
                unpruned_score = tree.score(unpruned_pred, test[label])

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Organize outputs
                classification_results.append(dict(problem_class=problem_class, dataset_name=dataset_name, fold=fold,
                                                   pruned_score=pruned_score, unpruned_score=unpruned_score,
                                                   n_pruned_nodes=n_pruned_nodes, n_unpruned_nodes=n_unpruned_nodes,
                                                   pruned_height=pruned_height, npruned_height=unpruned_height))
                feature_importances.append(pruned_feat_imps)
                feature_importances.append(unpruned_feat_imps)
                print(f"{dataset_name}: pruned={pruned_score}, unpruned={unpruned_score}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Regression
            else:

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Tune
                thetas = make_thetas(train, label, THETAS)
                print(f"Thetas: {thetas}")
                for theta_rel, theta_abs in thetas.items():
                    try:
                        print(f"Theta: {theta_rel}")
                        # Initialize root node and tree
                        root = RegressionDecisionNode(train, preprocessor.label, features, data_classes)
                        tree = RegressionDecisionTree(theta=theta_abs, verbose=True)
                        tree.add_node(root)
                        tree.populate_tree_metadata()

                        # Build decision tree
                        tree.train()

                        # Predict
                        tune_pred = tree.predict(val)
                        test_pred = tree.predict(test)

                        # Score
                        tune_score = tree.score(tune_pred, val[label])
                        test_score = tree.score(test_pred, test[label])

                        # Append results
                        regression_results.append([dataset_name, fold, theta_rel, theta_abs, tune_score, test_score])
                    except:
                        print(f"Error with dataset {dataset_name}, fold {fold}, theta_rel {theta_rel}.")
                        logging.error(f"Error with dataset {dataset_name}, fold {fold}, theta_rel {theta_rel}.")

    # Organize results
    cols = ["dataset_name", "fold", "theta_rel", "theta_abs", "tune_mse", "test_mse"]
    regression_results = pd.DataFrame(regression_results, columns=cols).sort_values(by="score")
    classification_results = pd.DataFrame(classification_results)
    feature_importances = pd.concat(feature_importances)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logging.debug("Save outputs.")
    classification_results_dst = dst_dir / "classification_results.csv"
    feature_importances_dst = dst_dir / "feature_importances.csv"
    regression_results_dst = dst_dir / "regression_results.csv"

    classification_results.to_csv(classification_results_dst, index=False)
    feature_importances.to_csv(feature_importances_dst, index=False)
    regression_results.to_csv(regression_results_dst, index=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logging.debug("Finish.\n")
