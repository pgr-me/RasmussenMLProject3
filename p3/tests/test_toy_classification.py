#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

"""
# Standard library imports
import collections as c
import typing as t

# Third party imports
import pandas as pd

# Local imports
from p3.nodes import DecisionNode
from p3.trees import ClassificationDecisionTree

df = pd.DataFrame(
    [
        ["Sunny", "Hot", "High", False, "N"],
        ["Sunny", "Hot", "High", True, "N"],
        ["Overcast", "Hot", "High", False, "P"],
        ["Rainy", "Mild", "High", False, "P"],
        ["Rainy", "Cool", "Normal", False, "P"],
        ["Rainy", "Cool", "Normal", True, "N"],
        ["Overcast", "Cool", "Normal", True, "P"],
        ["Sunny", "Mild", "High", False, "N"],
        ["Sunny", "Cool", "Normal", False, "P"],
        ["Rainy", "Mild", "Normal", False, "P"],
        ["Sunny", "Mild", "Normal", True, "P"],
        ["Overcast", "Mild", "High", True, "P"],
        ["Overcast", "Hot", "Normal", False, "P"],
        ["Rainy", "Mild", "High", True, "N"],
    ],
    columns=["Outlook", "Temperature", "Humidity", "Wind", "Class"]
)
df.index = [1 + x for x in df.index]
df.index.names = ["Example"]
quantiles = [0.2, 0.4, 0.6, 0.8]
label = "Class"
features = ["Outlook", "Temperature", "Humidity", "Wind"]
data_classes = c.OrderedDict([(x, "categorical") for x in features])


def test_decision_node():
    assert isinstance(DecisionNode(df, label, features, data_classes), DecisionNode)


def test_classification_decision_tree():
    root = DecisionNode(df, label, features, data_classes)
    tree = ClassificationDecisionTree(verbose=True)
    tree.add_node(root)
    tree.populate_tree_metadata()
    tree.make_tree()
    tree.rules = tree.make_rules(root)


if __name__ == "__main__":
    test_classification_decision_tree()
