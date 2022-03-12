#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Standard library imports
import collections as c
from copy import deepcopy
import typing as t

# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p3.nodes.classification_decision_node import ClassificationDecisionNode
from p3.algorithms.classification_entropy import select_best_feature, split_attribute
from p3.trees import Tree


class RegressionDecisionTree(Tree):
    def __init__(self, theta: float = 0.0, verbose: bool = False):
        super().__init__()
        self.theta = theta
        self.verbose = verbose
        self.rules = None
        self.subtrees = []
        self.label = None
        self.features = None
        self.data_classes = None
        self.id_counter = 0

    def __repr__(self):
        return f"Tree rooted at {str(self.root)}."

    def set_id_node(self, node):
        node.id = self.id_counter
        self.id_counter += 1
        return node.id

    def make_tree(self):
        """
        Build a decision tree from the root node down.
        Wrapper function for make_tree_ method.
        """
        self.make_tree_(self.root)

    def make_tree_(self, node: ClassificationDecisionNode):
        """
        Recursively build a decision subtree from the given node down.
        :param node: Root of subtree
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute node entropy & get majority label for tuning / if node is leaf
        node.compute_entropy()
        node.get_majority_label()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return if node entropy is less than threshold
        if node.entropy <= self.theta:
            # Create leaf labeled by majority class
            node.name = f"{node.depth}_{node.majority_label}"
            node.feature = node.majority_label
            self.root.leaves.append(node)
            if self.verbose:
                space = node.depth * "\t"
                print(f"{space}{node.parent_category}==>{node.name}")
            return

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find the best feature
        summary_stats = split_attribute(node.data, self.label, self.features)
        best_feature = select_best_feature(summary_stats)
        node.name = f"{node.depth}_{best_feature}"
        node.feature = best_feature
        parent_categories = sorted(node.data[best_feature].unique())
        if self.verbose:
            space = node.depth * "\t"
            print(f"{space}{node.parent_category}==>{node.name}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterate over each category of best feature
        for category in parent_categories:
            mask = node.data[best_feature] == category
            data = node.data.copy()[mask]
            child_node = ClassificationDecisionNode(data, self.label, self.features, self.data_classes)

            self.add_node(child_node, node)
            child_node.parent_category = category
            child_node.assign_root(self.root)
            child_node.make_rule()
            self.make_tree_(child_node)

    def populate_tree_metadata(self):
        """
        Populate label and feature attributes using corresponding root node attributes.
        """
        if self.is_not_empty():
            self.label = self.root.label
            self.features = self.root.features
            self.data_classes = self.root.data_classes

    def prune_node(self, node: ClassificationDecisionNode) -> ClassificationDecisionNode:
        """
        Prune node if its leaf score >= its subtree score.
        :param node: Decision node to prune
        :return: Pruned node
        """
        node.prune_children()
        node.name = f"{node.depth}_{node.majority_label}"
        self.root.get_children()
        self.set_height()
        return node

    def prune_nodes(self, data: pd.DataFrame):
        """
        Prune interior nodes of tree.
        :param data: Pruning dataset
        """
        for _, node in reversed(self.nodes.items()):
            if node.is_interior() and not node.is_root():
                subtree_score, leaf_score = self.score_node(node, data)
                if self.test_node(subtree_score, leaf_score):
                    self.prune_node(node)

    def score_node(self, node: ClassificationDecisionNode, data: pd.DataFrame) -> tuple:
        """
        Test node subtree's predictive power against that when it's a leaf.
        :param node: Node to test
        :param data: Pruning data
        :param label: Label column
        :return: Subtree and leaf scores
        """
        # Subset data: Get data at node
        mask = data.eval(node.rule)
        node_data = data.copy()[mask]

        # Predict using subtree using node leaves
        rules = tree.make_rules(node)
        subtree_pred = pd.Series([None for x in range(len(node_data))], index=node_data.index, name="subtree_pred")
        for label_val, rule in rules.items():
            mask = node_data.eval(rule)
            subtree_pred.loc[mask] = label_val
        subtree_pred = (subtree_pred == node_data[self.label]).rename("subtree_pred")
        subtree_score: pd.Series = subtree_pred.sum()

        # Predict using majority class
        leaf_pred = (node_data[node.label] == node.majority_label).rename("leaf_pred")
        leaf_score = leaf_pred.sum()

        return subtree_score, leaf_score

    @staticmethod
    def make_rules(node: ClassificationDecisionNode) -> dict:
        """
        Make decision tree rules to be used for prediction.
        :return:  Rules provided as eval statements keyed on label
        """
        node.get_leaves()
        if node.is_leaf():
            return {node.majority_label: node.rule}
        rules_dict = c.defaultdict(lambda: [])
        for leaf in node.leaves:
            leaf_ = deepcopy(leaf)
            rules_dict[leaf_.majority_label].append(f"({leaf_.rule})")

        rules = {}
        for label_val, conjunctive_rules in rules_dict.items():
            rules[label_val] = " | ".join(conjunctive_rules)
        return rules

    @staticmethod
    def test_node(subtree_score: int, leaf_score: int) -> bool:
        """
        Test if node should be pruned.
        :param subtree_score: Score of node subtree
        :param leaf_score: Score if node were leaf
        :return: True if node should be pruned
        """
        return leaf_score >= subtree_score
