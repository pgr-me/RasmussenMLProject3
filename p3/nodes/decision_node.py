#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, decision_node.py

This module provides base node class.

"""
# Third party imports
import pandas as pd
import numpy as np

# Local imports
from p3.nodes import Node


class DecisionNode(Node):
    """
    Base node for singly-linked list.
    """
    def __init__(self, mask: pd.Series, name: str, label: str, features: list, data_classes: list, children=None,
                 quantiles=None):
        """
        Set name and optionally set data attributes.
        :param mask: Mask used to subset original data to get node data
        :param name: Node name
        :param label: Label name
        :param features: List of features
        :param data_classes: Data classes for each feature
        :param children: Node children
        :param quantiles: Quantiles to split if node is numeric
        """
        # function arguments
        super().__init__(name, children)
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75, 1]
        self.mask = mask
        self.label = label
        self.features = [x for x in features if x != label]
        self.data_classes = data_classes
        self.quantiles = quantiles
        self.features_data_classes = dict(zip(features, data_classes))
        self.entropy = None
        self.majority_label = None

    def find_best_numeric_split(self, node_data: pd.DataFrame, feature: str, quantiles=None) -> tuple:
        """
        Find best numeric split.
        :param node_data:
        :param feature: Numeric feature to split
        :param quantiles: Quantiles to generate candidate splits for
        :return: Tuple of best entropy, mask, and split value for entropy minimizing numeric split
        Used as a helper function for split_entropy method, which is based on Equation 9.8 of Alpaydin.
        """
        if quantiles is None:
            quantiles = self.quantiles
        candidate_split_vals: pd.DataFrame = node_data[feature].quantile(quantiles).drop_duplicates()
        best_split_val = None
        best_mask = None
        best_ent = float("inf")
        for candidate_split_val in candidate_split_vals:
            candidate_mask: pd.Series = node_data[feature] <= candidate_split_val
            candidate_entropy = self.split_entropy_(node_data, candidate_mask)
            if candidate_entropy < best_ent:
                best_ent = candidate_entropy
                best_split_val = candidate_split_val
                best_mask = candidate_mask
        return best_ent, best_mask, best_split_val

    def node_entropy(self, data: pd.DataFrame) -> float:
        """
        Compute the entropy of a class.
        :param data: Data of node or one of its branches or candidate branches
        :return: Entropy of a node
        Based on Equation 9.3 of Alpaydin's Intro to Machine Learning, 4th Ed.
        """
        class_counts = data[self.label].value_counts()
        instances = class_counts.sum()
        class_fracs = (class_counts / instances)
        return -1 * (class_fracs * np.log2(class_fracs)).fillna(0).sum()

    def split_attribute(self, data: pd.DataFrame)->dict:
        """
        Find best attribute to split.
        :param data: Root dataset
        :return: Dictionary of entropy, feature, mask, and split value for child splits
        Based on algorithm provided in Figure 9.3 of Alpaydin.
        """
        node_data = data.copy()[self.mask]
        best = dict(entropy=float("inf"), feature=None, mask=None, split_value=None)
        for feature, data_class in self.features_data_classes.items():
            import pdb; pdb.set_trace()
            entropy, mask, split_value = self.split_entropy(node_data, feature, data_class)
            if entropy < best["entropy"]:
                best = dict(entropy=entropy, feature=feature, mask=mask, split_value=split_value)
        return best

    def split_entropy(self, data: pd.DataFrame, feature: str, data_class: str) -> tuple:
        """
        Compute split entropy / total impurity.
        :param data: Node data
        :param feature: Feature to split entropy for
        :param data_class: Data class - ordinal, numeric, categorical - of feature
        :return: Split entropy / total impurity
        Per Equation 9.8 of Alpaydin.
        """
        split_val = None
        if data_class == "categorical":
            split_mask = data[feature].isin([False, 0])
            split_ent = self.split_entropy_(data, split_mask)
        else:
            split_ent, split_mask, split_val = self.find_best_numeric_split(data, feature)
        return split_ent, split_mask, split_val

    def split_entropy_(self, data: pd.DataFrame, mask: pd.Series) -> float:
        """
        Split entropy for two-branch node.
        :param data: Data to split entropy for
        :param mask: Mask used to split entropy into left and right branches
        Helper function for split_entropy, which is based on Equation 9.8 of Alpaydin.
        Left branch is the mask and right branch is that mask's complement.
        """
        left_branch = data.copy()[mask]
        right_branch = data.copy()[~mask]
        left_branch_entropy = self.node_entropy(left_branch)
        right_branch_entropy = self.node_entropy(right_branch)
        left_branch_frac = len(left_branch) / len(data)
        right_branch_frac = len(right_branch) / len(data)
        return left_branch_frac * left_branch_entropy + right_branch_frac * right_branch_entropy

    def set_majority_label(self, data: pd.DataFrame):
        """
        Set the majority label of a leaf node.

        """
        self.majority_label = data[self.label].mode().iloc[0]
        return self.majority_label