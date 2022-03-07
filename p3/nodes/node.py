#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, decision_node.py

This module provides base node class.

"""
# Standard library imports
import collections as c
import typing as t


class Node:
    """
    Base node for singly-linked list.
    """

    def __init__(self, name: str, children=None):
        """
        Set name and optionally set data attributes.
        :param name: Node name
        :param children: Node children
        """
        # function arguments
        if children is None:
            children = c.defaultdict(lambda: None)
        self.name = name
        self.parent = None
        self.children = children
        self.height = 0
        self.root: t.Union[Node, None] = self

    def __repr__(self):
        return self.name

    def assign_root(self, root):
        self.root = root

    def is_interior(self):
        return len(self.children) > 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
