#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, regression_decision_node.py

This module provides the decision node used for regression problems.

"""
# Standard library imports
import collections as c

# Third party imports
import pandas as pd

# Local imports
from p3.nodes.decision_node import DecisionNode
from p3.algorithms.regression_error import compute_error


class RegressionDecisionNode(DecisionNode):
    """
    Classification decision node.
    """

    def __init__(self, data: pd.DataFrame, label: str, features: list, data_classes: c.OrderedDict, name=None,
                 parent=None, children=None,
                 quantiles=None):
        """
        Set name and optionally set data attributes.
        :param data: Data associated with node
        :param label: Label name
        :param features: List of features
        :param data_classes: Data classes for each feature
        :param name: Node name
        :param parent: Parent of node
        :param children: Node children
        :param quantiles: Quantiles to split if node is numeric
        """
        super().__init__(data, label, features, data_classes, name, parent, children, quantiles)
        self.data = data
        self.label = label
        self.features = [x for x in features if x != label]
        self.data_classes = data_classes
        self.name = name
        self.parent = parent
        self.quantiles = quantiles
        if quantiles is None:
            self.quantiles = [0.25, 0.5, 0.75, 1]
        self.error = None
        self.parent_category = None
        self.feature = None
        self.rule = ""
        self.mean = None
        self.feature_branches = None

    def compute_mean(self) -> float:
        """
        Compute node mean.
        :return: Node mean
        """
        self.mean = self.data[self.label].mean()
        return self.mean

    def compute_error(self):
        """
        Compute node entropy
        :return: Compute mean squared error
        """
        if self.mean is None:
            raise ValueError("You must run compute_mean method first.")
        y_true = self.data[self.label]
        self.error = compute_error(y_true, self.mean)
        return self.error
