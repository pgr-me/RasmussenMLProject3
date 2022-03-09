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
from p3.nodes.decision_node import DecisionNode
from p3.algorithms.classification_entropy import select_best_feature, split_attribute


class ClassificationDecisionTree:
    def __init__(self, root_node: DecisionNode, theta: float = 0.1, verbose: bool = False):
        self.root_node = root_node
        self.theta = theta
        self.verbose = verbose
        self.rules = None

    def make_rules(self) -> dict:
        """
        Make decision rules from tree.
        :return: Rules defined as eval statements keyed on label
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterate over each leaf & work back up to root, building conjunctive eval statement
        rules_dict = c.defaultdict(lambda: [])
        for leaf in self.root_node.leaves:
            leaf_ = deepcopy(leaf)
            majority_label = leaf_.majority_label
            eval_strs = []

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Work up to root and then break the while loop
            while leaf_.parent is not None:
                eval_str = f"({leaf_.parent.feature}=='{leaf_.parent_category}')"
                if ("False" in eval_str) or ("True" in eval_str):
                    eval_str = eval_str.replace("'", "")
                eval_strs.append(eval_str)
                leaf_ = leaf_.parent

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle case when eval string only has one term
            if len(eval_strs) > 1:
                eval_str = f"({' & '.join(eval_strs)})"

            rules_dict[majority_label].append(eval_str)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build disjunctive rule sets from conjunctive eval strings
        rules = {}
        for label, rules_li in rules_dict.items():
            rules[label] = " | ".join(rules_li)
        self.rules = rules

        return self.rules

    def make_tree(self):
        self.make_tree_(self.root_node)

    def make_tree_(self, node: DecisionNode):
        """
        Make classification tree.
        :param node:
        :param theta:
        :return:
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute node entropy
        node.compute_entropy()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return if node entropy is less than threshold
        if node.entropy < self.theta:
            # Create leaf labeled by majority class
            node.get_majority_label()
            node.name = f"{node.depth}_{node.majority_label}"
            node.feature = node.majority_label
            self.root_node.leaves.append(node)
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
            print(f"{space}{node.name}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterate over each category of best feature
        for category in parent_categories:
            mask = node.data[best_feature] == category
            data = node.data.copy()[mask]
            child_node = DecisionNode(data, self.label, self.features, self.data_classes)
            child_node.parent = node
            child_node.depth = child_node.parent.depth + 1
            child_node.parent_category = category
            child_node.assign_root(self.root_node)
            node.children.append(child_node)
            self.make_tree_(child_node)
