#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

"""
# Standard library imports
import collections as c
from copy import deepcopy
import json
from pathlib import Path
import warnings

# Local imports
from p3.nodes import RegressionDecisionNode
from p3.trees import RegressionDecisionTree
from p3.preprocessing.tree_preprocessor import TreePreprocessor

warnings.filterwarnings('ignore')

test_dir = Path(".").absolute()
p3_dir = test_dir.parent
repo_dir = p3_dir.parent
src_dir = repo_dir / "data"

# Load data catalog and tuning params
with open(src_dir / "data_catalog.json", "r") as file:
    data_catalog = json.load(file)
data_catalog = {k: v for k, v in data_catalog.items() if k in ["forestfires", "machine", "abalone"]}


def test_regression_decision_tree():
    for dataset_name, dataset_meta in data_catalog.items():
        print(f"Dataset: {dataset_name}")
        preprocessor = TreePreprocessor(dataset_name, dataset_meta, src_dir)
        preprocessor.load()
        preprocessor.identify_features_label_id()
        preprocessor.replace()
        preprocessor.log_transform()
        preprocessor.set_data_classes()
        preprocessor.impute()
        preprocessor.drop()
        preprocessor.shuffle()
        testing_data = preprocessor.data.copy().sample(frac=0.2, random_state=777)
        preprocessor.discretize()
        testing_data = preprocessor.discretize_nontrain(testing_data)
        data = preprocessor.data.copy().sample(frac=0.8, random_state=777)

        # Extract label, features, and data classes
        label = preprocessor.label
        data_classes = preprocessor.data_classes
        features = preprocessor.features

        # Initialize root node and tree
        root = RegressionDecisionNode(data, preprocessor.label, features, data_classes)
        tree = RegressionDecisionTree(verbose=True)
        tree.add_node(root)
        tree.populate_tree_metadata()

        # Build decision tree
        tree.train()

        # Predict
        pred = tree.predict(testing_data)

        # Score
        score = tree.score(pred, testing_data[label])
        print(f"{dataset_name}: score={score}")


if __name__ == "__main__":
    test_regression_decision_tree()
