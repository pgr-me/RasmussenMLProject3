#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, decision_node.py

This module provides base node class.

"""
# Standard library imports
import collections as c
import typing as t


class Node:
    """
    Base node that DecisionNode inherits from.
    """

    def __init__(self, name: str, children=None):
        """
        Set name and optionally set data attributes.
        :param name: Node name
        :param children: Node children
        """
        if children is None:
            self.children = []
        else:
            self.children = children
        self.name = name
        self.parent = None
        self.depth = 0
        self.depth = 0
        self.root: t.Union[Node, None] = self
        self.leaves = []
        self.id = None

    def __repr__(self):
        return str(self.name)

    def assign_root(self, root):
        """

        :param root:
        :return:
        """
        self.root = root

    def get_depth(self):
        for node in self.leaves:
            if node.depth > self.depth:
                self.depth = node.depth

    def get_interior_nodes(self)->list:
        """
        Get interior nodes of node.
        :return: List of interior nodes
        """
        interior_nodes = []
        def _get_interior_nodes(node):
            for child in node.children:
                if child.is_interior():
                    interior_nodes.append(child)
                _get_interior_nodes(child)
        _get_interior_nodes(self)
        self.interior_nodes = interior_nodes
        return self.interior_nodes

    def get_leaves(self):
        leaves = []
        if not self.children:
            return leaves

        def _get_leaves(node):
            for child in node.children:
                if child.is_leaf():
                    leaves.append(child)
                _get_leaves(child)

        _get_leaves(self)
        self.leaves = leaves
        return self.leaves

    def is_interior(self):
        return len(self.children) > 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def prune_children(self):
        children = self.get_children()
        children.sort(key=lambda x: x.name, reverse=True)
        for child in children:
            child.root = None
            child.parent = None
        self.children = []
        return self

    def get_children(self):
        """
        Get all children - not just immediate - of node.
        :param node: Node to find children for
        :return: Children - not just immediate - of node
        """
        children = []
        if self.is_leaf():
            return children

        def _get_children(node):
            """
            Inner recursive helper function.
            """
            for child in node.children:
                children.append(child)
                _get_children(child)

        _get_children(self)

        return children

    def set_id(self, node_id: int)->int:
        self.id = node_id
        return node_id