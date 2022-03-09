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
        self.name = name
        self.parent = None
        self.height = 0
        self.depth = 0
        self.root: t.Union[Node, None] = self
        self.leaves = []

    def __repr__(self):
        return str(self.name)

    def assign_root(self, root):
        """

        :param root:
        :return:
        """
        self.root = root

    def is_interior(self):
        return len(self.children) > 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
