#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

"""
# Standard library imports
import collections as c
from copy import deepcopy
import json
from pathlib import Path
import typing as t
import warnings

# Third party imports
import pandas as pd

# Local imports
from p3.algorithms.classification_entropy import get_feature_importances
from p3.nodes import ClassificationDecisionNode
from p3.preprocessing.tree_preprocessor import TreePreprocessor
from p3.preprocessing.split import make_splits
from p3.trees import ClassificationDecisionTree

warnings.filterwarnings('ignore')

test_dir = Path(".").absolute()
p3_dir = test_dir.parent
repo_dir = p3_dir.parent
src_dir = repo_dir / "data"

# Load data catalog and tuning params
with open(src_dir / "data_catalog.json", "r") as file:
    data_catalog = json.load(file)
with open(src_dir / "tuning_params.json", "r") as file:
    tuning_params = json.load(file)
data_catalog = {k: v for k, v in data_catalog.items() if k in ["breast-cancer-wisconsin", "car", "house-votes-84"]}
data_catalog = {k: v for k, v in data_catalog.items() if k in ["car"]}
k_folds = 5
val_frac = 0.2
random_state = 777


def test_classification_decision_tree():
    for dataset_name, dataset_meta in data_catalog.items():
        print(f"Dataset: {dataset_name}")
        preprocessor = TreePreprocessor(dataset_name, dataset_meta, src_dir)
        preprocessor.load()
        preprocessor.data = preprocessor.data.sample(frac=0.2, random_state=random_state)
        preprocessor.identify_features_label_id()
        preprocessor.replace()
        preprocessor.log_transform()
        preprocessor.set_data_classes()
        preprocessor.impute()
        preprocessor.drop()
        preprocessor.shuffle()

        # Extract label and data classes
        label = preprocessor.label
        data_classes = preprocessor.data_classes
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

        # Iterate over each fold and save results
        results: t.Union[list, pd.DataFrame] = []
        feature_importances: t.Union[list, pd.DataFrame] = []
        for fold in range(1, k_folds + 1):
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

            # Initialize root node and tree
            root = ClassificationDecisionNode(train, label, features, data_classes)
            tree = ClassificationDecisionTree(verbose=True)
            tree.add_node(root)
            tree.populate_tree_metadata()

            # Build decision tree
            tree.train()
            tree.set_height()
            unpruned_height = tree.height
            tree.rules = tree.unpruned_rules = tree.make_rules(root)
            n_unpruned_nodes = len(tree.nodes)
            unpruned_feat_imps = get_feature_importances(tree, dataset_name, fold, pruned=False)

            # Prune nodes
            tree.prune_nodes(val)
            tree.set_height()
            pruned_height = tree.height
            n_pruned_nodes = len(tree.nodes)
            pruned_feat_imps = get_feature_importances(tree, dataset_name, fold, pruned=True)
            tree.rules = tree.make_rules(root)

            # Predict
            pruned_pred = tree.predict(test)
            unpruned_pred = tree.predict(test, pruned=False)

            # Score
            pruned_score = tree.score(pruned_pred, test[label])
            unpruned_score = tree.score(unpruned_pred, test[label])
            results.append(dict(problem_class=problem_class, dataset_name=dataset_name, fold=fold,
                                pruned_score=pruned_score, unpruned_score=unpruned_score, n_pruned_nodes=n_pruned_nodes,
                                n_unpruned_nodes=n_unpruned_nodes, pruned_height=pruned_height,
                                unpruned_height=unpruned_height))
            print(f"{dataset_name}: pruned={pruned_score}, unpruned={unpruned_score}")

            feature_importances.append(pruned_feat_imps)
            feature_importances.append(unpruned_feat_imps)

    results = pd.DataFrame(results)
    feature_importances = pd.concat(feature_importances)

    # Save outputs
    results.to_csv("classification_results.csv")
    feature_importances.to_csv("feature_importances.csv")


if __name__ == "__main__":
    test_classification_decision_tree()
