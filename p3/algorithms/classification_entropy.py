#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Standard library imports
import collections as c

# Third party libraries
import numpy as np
import pandas as pd


def compute_entropy(feature_counts: pd.Series)->float:
    """
    Compute entropy of a node.
    :param feature_counts: Node data
    :return: Node entropy
    Based on course module 5, lecture 2.
    """
    _, label = feature_counts.index.names
    tots = feature_counts.groupby(label).sum().to_frame()
    tots["frac"] = tots["ct"] / tots["ct"].sum()
    return tots[["frac"]].multiply(np.log2(tots["frac"]), axis=0).sum().abs().iloc[0]

def compute_expected_entropy(unweighted_entropies, feature_counts, feature: str):
    """
    Compute expected entropy for a feature.
    :param unweighted_entropies:
    :param feature_counts: Table of counts by feature by label
    :param feature: Feature to compute expected entropy for
    :return:
    """
    feature_cts = (feature_counts.groupby(feature).sum()).rename("feature_ct")
    feature_wts = (feature_cts / feature_counts.sum()).rename("feature_wt")
    return feature_wts.to_frame().multiply(unweighted_entropies, axis=0).sum().iloc[0]

def compute_gain(node_entropy, expected_entropy):
    return node_entropy - expected_entropy


def compute_gain_ratio(gain: float, split_info: float):
    return gain / split_info


def compute_split_info(feature_counts: pd.Series, feature):
    totals = feature_counts.groupby(feature).sum()
    frac = totals / totals.sum()
    lg_frac = np.log2(frac)
    return abs(frac.multiply(lg_frac, axis=0).sum())


def compute_unweighted_entropies(feature_counts):
    """
    Compute entropies of node feature splits.
    :param data:
    :param label:
    :param feature:
    :return: Node feature split entropies.
    """
    pivot_counts = pivot_feature_counts(feature_counts)
    pivot_counts["ct"] = pivot_counts.sum(axis=1)
    fracs = pivot_counts.iloc[:, :-1].divide(pivot_counts["ct"], axis=0)
    return fracs.multiply(np.log2(fracs)).fillna(0).abs().sum(axis=1)


def get_feature_counts(data: pd.DataFrame, label, feature, ct_col="ct"):
    feature_counts = data.copy()
    feature_counts[ct_col] = 1
    return feature_counts.groupby([feature, label])[ct_col].sum().sort_index()


def pivot_feature_counts(feature_counts):
    feature, label = feature_counts.index.names
    pivot = pd.pivot(feature_counts.reset_index(), index=feature, columns=label, values="ct")
    return pivot.fillna(0).sort_index().astype(int)


def split_attribute(data, label, features)->dict:
    entropy_stats = c.OrderedDict()
    for feature in features:
        print(feature)
        feature_counts = get_feature_counts(data, label, feature)
        entropy = compute_entropy(feature_counts)
        unweighted_entropies = compute_unweighted_entropies(feature_counts)
        expected_entropy = compute_expected_entropy(unweighted_entropies, feature_counts, feature)
        gain = compute_gain(entropy, expected_entropy)
        split_info = compute_split_info(feature_counts, feature)
        gain_ratio = compute_gain_ratio(gain, split_info)
        entropy_stats[feature] = dict(entropy=entropy, gain=gain, split_info=split_info, gain_ratio=gain_ratio)
    entropy_stats = pd.DataFrame(entropy_stats).transpose().sort_values(by="gain_ratio", ascending=False)
    return dict(best_feature=entropy_stats.index.values[0], entropy_stats=entropy_stats)




