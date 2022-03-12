#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

"""
# Standard library imports
import collections as c
import json
from pathlib import Path
import warnings

# Local imports
from p3.nodes import DecisionNode
from p3.trees import ClassificationDecisionTree
from p3.preprocessing.tree_preprocessor import TreePreprocessor

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


def test_classification_decision_tree():
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
        preprocessor.discretize()

        # Extract label, features, and data classes
        label = preprocessor.label
        data_classes = preprocessor.data_classes
        features = preprocessor.features

        data = preprocessor.data.copy().sample(frac=0.2)
        pruning_data = preprocessor.data.copy().sample(frac=0.2)

        # Initialize root node and tree
        root = DecisionNode(data, preprocessor.label, features, data_classes)
        tree = ClassificationDecisionTree(verbose=True)
        tree.add_node(root)
        tree.populate_tree_metadata()

        # Build decision tree
        tree.train()
        tree.rules = tree.unpruned_rules = tree.make_rules(root)

        # Prune nodes
        tree.prune_nodes(pruning_data)
        tree.rules = tree.make_rules(root)

        # Predict
        pruned_pred = tree.predict(preprocessor.data)
        unpruned_pred = tree.predict(preprocessor.data, pruned=False)

        # Score
        pruned_score = tree.score(pruned_pred, preprocessor.data[label])
        unpruned_score = tree.score(unpruned_pred, preprocessor.data[label])
        print(f"{dataset_name}: pruned={pruned_score}, unpruned={unpruned_score}")


if __name__ == "__main__":
    test_classification_decision_tree()
