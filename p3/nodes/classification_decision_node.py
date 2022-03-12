#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, classification_decision_node.py

This module provides base node class.

"""
# Standard library imports
import collections as c

# Third party imports
import pandas as pd

# Local imports
from p3.nodes.decision_node import DecisionNode
from p3.algorithms.classification_entropy import compute_entropy, get_feature_counts


class ClassificationDecisionNode(DecisionNode):
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
        self.entropy = None
        self.majority_label = None
        self.parent_category = None
        self.feature = None
        self.rule = ""

    def get_majority_label(self):
        """
        Set the majority label of a leaf node.

        """
        self.majority_label = self.data[self.label].mode().iloc[0]
        return self.majority_label

    def compute_entropy(self):
        """
        Compute node entropy
        :return:
        """
        self.entropy = compute_entropy(get_feature_counts(self.data, self.label))
        return self.entropy

