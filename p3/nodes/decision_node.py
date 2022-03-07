#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, node.py

This module provides base node class.

"""
# Standard library imports
import typing as t

# Local imports
from p3.nodes import BaseNode

class Node(BaseNode):
    """
    Base node for singly-linked list.
    """

    def __init__(self, name: str, children=None):
        """
        Set name and optionally set data attributes.
        :param name: BaseNode name
        :param children: BaseNode children
        """
        # function arguments
        super().__init__(name, children)
        self.mask: t.Union[]

    def __repr__(self):
        return self.name

    def is_interior(self):
        return len(self.children) > 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None
