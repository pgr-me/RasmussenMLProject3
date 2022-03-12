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
from p3.nodes import Node
from p3.trees import Tree, TreeError


node_1 = Node("zounds")
node_2 = Node("bounds")
node_3 = Node("xounds")
node_4 = Node("lounds")
node_5 = Node("leaf")
tree = Tree()
tree.add_node(node_1)
tree.add_node(node_2, node_1)
tree.add_node(node_3, node_1)
tree.add_node(node_5, node_2)

# def test_root():
#     """
#     Test root of tree.
#     """
#     assert tree.root == node_1
#
#
# def test_node_is_leaf():
#     """
#     Test node is a leaf.
#     """
#     assert tree.nodes["leaf"].is_leaf()
#
#
# def test_add_parentless_node_to_nonempty_tree():
#     with pytest.raises(TreeError) as exc_info:
#         tree.add_node(node_4)


# def test_remove_too_far_interior():
#     """
#     Test exception raised if node with non-leaf children is removed.
#     """
#     with pytest.raises(NotImplementedError) as exc_info:
#         tree.remove_node(tree.root)
#
#
# def test_remove_node():
#     """
#     Test exception raised if node with non-leaf children is removed.
#     """
#
#     tree.remove_node("bounds")
#
#
# def test_height():
#     """
#     Test height computation.
#     """
#     assert tree.height == 1
