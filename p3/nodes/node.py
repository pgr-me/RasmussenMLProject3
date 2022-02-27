#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, node.py

This module provides base node class.

"""


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
            children = {}
        self.name = name
        self.parent = None
        self.children = children
        self.height = 0

    def __repr__(self):
        return self.name

    def is_interior(self):
        return len(self.children) > 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
