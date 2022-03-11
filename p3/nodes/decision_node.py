#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, decision_node.py

This module provides base node class.

"""
# Standard library imports
import collections as c

# Third party imports
import pandas as pd

# Local imports
from p3.nodes import Node
from p3.algorithms.classification_entropy import compute_entropy, get_feature_counts


class DecisionNode(Node):
    """
    Classification decision node.
    """

    def __init__(self, data: pd.DataFrame, label: str, features: list, data_classes: c.OrderedDict, name=None,
                 parent=None, children=None,
                 quantiles=None):
        """
        Set name and optionally set data attributes.
        :param data: Data associated with node
        :param name: Node name
        :param label: Label name
        :param features: List of features
        :param data_classes: Data classes for each feature
        :param children: Node children
        :param quantiles: Quantiles to split if node is numeric
        """
        super().__init__(name, children)
        self.data = data
        self.label = label
        self.features = [x for x in features if x != label]
        self.data_classes = data_classes
        self.name = name
        self.parent = parent
        # if children is None:
        #     self.children = c.defaultdict(lambda: None)
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

    def make_rule(self) -> str:
        if not self.is_root():
            parent_rule = self.parent.rule
            node_rule = f"({self.parent.feature}=='{self.parent_category}')"
            if ("False" in node_rule) or ("True" in node_rule):
                node_rule = node_rule.replace("'", "")
            if self.parent.is_root():
                self.rule = node_rule
            else:
                self.rule = f"{parent_rule} and {node_rule}"

        return self.rule


def breadth_first_print(node):
    """
    Print the tree beginning at node in breadth-first, left-to-right order.
    param node: Node for which tree will be printed
    """
    space = "\t" * node.depth
    antecedent = "" if node.is_root() else f"{space}{node.parent_category}==>"
    print(f"{antecedent}{node.name}")
    for child_node in node.children:
        breadth_first_print(child_node)
