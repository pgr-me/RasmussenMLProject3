#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, test_nodes.py

This module tests the nodes module.

"""
# Standard library imports
import collections as c
import typing as t

# Third party imports
import pytest

# Local imports
from p3.nodes import DecisionNode


def test_node_is_interior():
    """
    Test node is interior.
    """
    children = c.defaultdict(lambda: t.Union[None, int])
    children["a"] = 1
    children["b"] = 2
    node = DecisionNode("zounds", children=children)
    assert node.is_interior()


def test_node_is_leaf():
    """
    Test node is a leaf.
    """
    node = DecisionNode("zounds")
    assert node.is_leaf()
