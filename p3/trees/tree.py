#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, tree.py

This module provides the base tree class and the tree error exception class.

"""

# Standard library imports
import collections as c
import typing as t

# Local imports
from p3.nodes import DecisionNode


class TreeError(Exception):
    pass


class Tree:
    """
    Base tree class.
    """

    def __init__(self):
        self.root: t.Union[DecisionNode, None] = None
        self.nodes: c.defaultdict[str: DecisionNode, None] = c.defaultdict(lambda: None)
        self.height: t.Union[int, None] = None

    def __repr__(self):
        return f"Tree rooted at {self.root}."

    def add_node(self, node: DecisionNode, parent_node: DecisionNode = None):
        """
        Add node to tree.
        :param node: Node to add
        :param parent_node: Parent node
        """
        if self.is_not_empty() and parent_node is None:
            raise TreeError("Parent node must be specified when tree is not empty.")
        if node.name in self.nodes:
            raise TreeError(f"Node {node.name} already exists in tree.")

        # Case when tree is empty
        if self.is_empty():
            self.root = node
            node.height = 0
            self.height = 0

        # Case when tree is not empty
        else:
            parent_node.children[node.name] = node
            node.parent = parent_node
            node.height = node.parent.height + 1

        self.nodes[node.name] = node

        # Update tree height
        self.set_height()

    def get_height(self) -> int:
        """
        Get height of tree.
        :return:Height of tree
        """
        if self.is_empty():
            return self.height
        height = 0
        for node_name, node in self.nodes.items():
            if node.height > height:
                height = node.height
        self.height = height
        return self.height

    def get_node(self, node_name: str) -> DecisionNode:
        """
        Get node by its name.
        :param node_name:
        :return: Node
        """
        if node_name not in self.nodes:
            raise TreeError(f"Node {node_name} not in tree.")
        return self.nodes[node_name]

    def is_empty(self) -> bool:
        """
        Return true if tree has no nodes.
        :return: True if tree is empty
        """
        return self.root is None

    def is_not_empty(self) -> bool:
        """
        Return true if tree has nodes.
        :return: True if tree is not empty
        """
        return self.root is not None

    def remove_node(self, node: t.Union[str, DecisionNode]) -> DecisionNode:
        """
        Remove node from tree.
        :param node: Node to remove
        :return: Removed node
        """
        if self.root is None:
            raise TreeError("Cannot remove a node from an empty tree.")
        if isinstance(node, str):
            node = self.get_node(node)
        if node.height < self.height - 1:
            msg = "Can only remove 1) leaves or 2) nodes whose children are only leaves."
            raise NotImplementedError(msg)

        # Case when node is root and childless
        if node.is_root() and node.is_leaf():
            self.root = None
            self.height = None

        # Case when node is a leaf
        elif node.is_leaf():
            node.parent.children.pop(node.name)
            node.parent = None

        # Case when node is interior
        elif node.is_interior():
            # Get node to promote
            _, promoted_node = self.select_promotion_node(node)
            # Decrement promoted node's height
            promoted_node.height -= 1
            if promoted_node.is_interior():
                raise TreeError(f"Promoted node {promoted_node} is not a leaf but must be.")
            # Set node's children to empty dict
            node.children = c.defaultdict(lambda: None)
            # If node is root, make promoted node new root and then set promoted node's parent to None
            if node.is_root():
                self.root = promoted_node
                promoted_node.parent = None
            # Otherwise, add promoted node to parent's children, remove node from parent's children, and update promoted node's parent
            else:
                node.parent.children[promoted_node.name] = promoted_node
                node.parent.children.pop(node.name)
                promoted_node.parent = node.parent
            # Un-wire node's parent
            node.parent = None

        # Raise an error if above cases have been incorrectly handled
        else:
            raise TreeError("Node must be either interior or a leaf.")

        # Remove node from tree's dictionary of nodes
        self.nodes.pop(node.name)
        # Update tree height
        self.set_height()

        # Return removed node
        return node

    def set_height(self) -> int:
        """
        Set height of tree as height of node which has max height.
        :return: Updated height of tree
        """
        self.height = self.get_height()
        return self.height

    @staticmethod
    def select_promotion_node(parent_node: DecisionNode) -> tuple:
        """
        Select child node to promote as parent.
        :param parent_node: Parent node of promotion candidates
        :return: Promoted child node name and associated node
        This is a base function that is over-ridden in classes that inherit from this class.
        """
        if not parent_node.children:
            raise TreeError("Cannot promote non-existent child from a parent node.")
        # For this base class, just promote the first node encountered in iteration over dict
        for child_node_name, child_node in parent_node.children.items():
            return child_node_name, child_node
