#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, regression_decision_tree.py

This module provides the decision tree class to solve regression problems.

"""

# Standard library imports
import collections as c
from copy import deepcopy

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from p3.nodes import RegressionDecisionNode
from p3.trees.tree import Tree
from p3.algorithms.regression_error import compute_error, select_best_feature, split_attribute


class RegressionDecisionTree(Tree):
    def __init__(self, theta: float = 0.0, verbose: bool = False):
        super().__init__()
        self.theta = theta
        self.verbose = verbose
        self.label = None
        self.features = None
        self.data_classes = None
        self.id_counter = 0
        self.unused_features = None
        self.means = {}
        self.mean = None

    def __repr__(self):
        return f"Tree rooted at {str(self.root)}."

    def set_id_node(self, node):
        node.id = self.id_counter
        self.id_counter += 1
        return node.id

    def train(self):
        """
        Build a decision tree from the root node down.
        Wrapper function for make_tree_ method.
        """
        self.make_tree_(self.root)

    def make_tree_(self, node: RegressionDecisionNode):
        """
        Recursively build a decision subtree from the given node down.
        :param node: Root of subtree
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute node mean & MSE
        node.compute_mean()
        node.compute_error()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute summary statistics of node attribute candidates
        cols = list(self.unused_features) + [self.label]
        branch_stats = split_attribute(node.data[cols], self.label, self.unused_features)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return if node error <= threshold or no more features to process
        condition = (node.error <= self.theta) or (len(branch_stats) == 0) or self.no_more_features() or isinstance(
            branch_stats, pd.Series)
        if condition:
            # Create leaf labeled by majority class
            if np.isnan(node.mean):
                node.mean = node.parent.mean
            node.name = f"{node.depth}__{node.mean:.3f}"
            node.feature = None
            self.root.leaves.append(node)
            if self.verbose:
                space = node.depth * "\t"
                print(f"{space}{node.parent_category}==>{node.name}")
            return

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find the best feature
        best_feature = select_best_feature(branch_stats)
        node.name = f"{node.depth}_{best_feature}"
        node.feature = best_feature
        self.unused_features.remove(best_feature)
        branches = branch_stats.loc[best_feature].set_index("branch_cat")
        if self.verbose:
            space = node.depth * "\t"
            print(f"{space}{node.parent_category}==>{node.name}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterate over each category of best feature
        for branch_cat, branch_cat_data in branches.iterrows():
            mask = node.data[best_feature] == branch_cat
            data = node.data.copy()[mask]
            child_node = RegressionDecisionNode(data, self.label, self.features, self.data_classes)

            self.add_node(child_node, node)
            child_node.parent_category = branch_cat
            child_node.assign_root(self.root)
            child_node.make_rule()
            self.make_tree_(child_node)

    def no_more_features(self):
        return len(self.unused_features) == 0

    def populate_tree_metadata(self):
        """
        Populate label and feature attributes using corresponding root node attributes.
        """
        if self.is_not_empty():
            self.label = self.root.label
            self.features = self.root.features
            self.data_classes = self.root.data_classes
            self.unused_features = set(deepcopy(self.features))

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict classes.
        :param pruned: True to use pruned ruleset
        :param data: Prediction dataset
        :return: Predicted classes
        """
        predicted = pd.Series([None for x in range(len(data))], index=data.index)
        for child in self.root.children:
            mask = data.eval(child.rule)
            predicted.loc[mask] = child.mean

        # Make sure tree functions properly
        if predicted.isnull().sum() > 0:
            raise ValueError("Some of data is unpredicted.")

        return predicted

    def score(self, pred: pd.Series, truth: pd.Series) -> float:
        """
        Score on the basis of accuracy.
        :return: Accuracy score
        """
        return compute_error(truth, pred)
