#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Standard library imports
import collections as c

# Third party libraries
import numpy as np
import pandas as pd


def compute_entropy(feature_counts: pd.Series) -> float:
    """
    Compute entropy of a node.
    :param feature_counts: Node data
    :return: Node entropy
    """
    label = feature_counts.index.names
    tots = feature_counts.groupby(label).sum().to_frame()
    tots["frac"] = tots["ct"] / tots["ct"].sum()
    return tots[["frac"]].multiply(np.log2(tots["frac"]), axis=0).sum().abs().iloc[0]


def compute_expected_entropy(unweighted_entropies, feature_counts, feature: str) -> float:
    """
    Compute expected entropy for a feature.
    :param unweighted_entropies: Unweighted entropy for each feature category
    :param feature_counts: Table of label counts for a feature's categories
    :param feature: Feature to compute expected entropy for
    :return: Expected entropy
    """
    feature_cts = (feature_counts.groupby(feature).sum()).rename("feature_ct")
    feature_wts = (feature_cts / feature_counts.sum()).rename("feature_wt")
    return feature_wts.to_frame().multiply(unweighted_entropies, axis=0).sum().iloc[0]


def compute_gain(node_entropy, expected_entropy) -> float:
    """
    Compute gain for feature split.
    :param node_entropy: Entropy of node
    :param expected_entropy: Expected entropy of split
    :return: Information gain of split
    """
    return node_entropy - expected_entropy


def compute_gain_ratio(gain: float, split_info: float) -> float:
    """
    Compute gain ratio for feature split.
    :param gain: Information gain of feature split
    :param split_info: Split information of feature (i.e., penalty term)
    :return: Gain ratio
    """
    return 0 if gain == 0 else gain / split_info


def compute_split_info(feature_counts: pd.Series, feature: str) -> float:
    """
    Compute split information for feature.
    :param feature_counts: Table of label counts for a feature's categories
    :param feature: Feature to compute split information for
    :return: Split information for a feature
    """
    totals = feature_counts.groupby(feature).sum()
    frac = totals / totals.sum()
    lg_frac = np.log2(frac)
    return abs(frac.multiply(lg_frac, axis=0).sum())


def compute_unweighted_entropies(feature_counts: pd.Series):
    """
    Compute unweighted entropy of each feature category.
    :param feature_counts: Table of label counts for a feature's categories
    :return: Unweighted entropy for each feature category
    Example output:
        Outlook
        Overcast    0.000000
        Rainy       0.970951
        Sunny       0.970951
        dtype: float64
    """
    pivot_counts = pivot_feature_counts(feature_counts)
    pivot_counts["ct"] = pivot_counts.sum(axis=1)
    fracs = pivot_counts.iloc[:, :-1].divide(pivot_counts["ct"], axis=0)
    return fracs.multiply(np.log2(fracs)).fillna(0).abs().sum(axis=1)


def get_feature_counts(data: pd.DataFrame, label, feature=None, ct_col="ct"):
    """
    Aggregate counts for a feature by label.
    :param data: Dataset
    :param label: Label / class of data
    :param feature: Feature to compute category label counts for
    :param ct_col: Name of count column
    :return: Table of label counts for a feature's categories
    Example output:
        Outlook   Class
        Overcast  P        4
        Rainy     N        2
                  P        3
        Sunny     N        3
                  P        2
        Name: ct, dtype: int64
    """
    feature_counts = data.copy()
    feature_counts[ct_col] = 1
    if feature is None:
        index = [label]
    else:
        index = [feature, label]
    return feature_counts.groupby(index)[ct_col].sum().sort_index()


def get_feature_importances(tree, dataset_name: str, fold: int, pruned: bool) -> pd.DataFrame:
    """
    Obtain list and associated metadata of nodes by feature importance.
    :param tree: Decision tree
    :return: Feature importances
    """
    feat_imp_li = [[k, v.__str__(), v.entropy] for k, v in tree.nodes.items()]
    feat_imp = pd.DataFrame(feat_imp_li, columns=["id", "name", "entropy"])
    interior_mask = feat_imp["name"].str.contains("__")
    feat_imp = feat_imp[interior_mask]
    feat_imp["depth"] = feat_imp["name"].str.split("_").str[0].astype(int)
    feat_imp["feature"] = feat_imp["name"].str.split("_").str[1]
    feat_imp["dataset_name"] = dataset_name
    feat_imp["fold"] = fold
    feat_imp["pruned"] = pruned
    return feat_imp.set_index(["dataset_name", "fold", "pruned"]).reset_index()


def pivot_feature_counts(feature_counts):
    """
    Make a pivot table of feature counts.
    :param feature_counts: Table of label counts for that feature's categories
    :return: Pivot table of
    Example output:
        Class     N  P
        Outlook
        Overcast  0  4
        Rainy     2  3
        Sunny     3  2
    """
    feature, label = feature_counts.index.names
    pivot = pd.pivot(feature_counts.reset_index(), index=feature, columns=label, values="ct")
    return pivot.fillna(0).sort_index().astype(int)


def split_attribute(data, label, features) -> pd.DataFrame:
    """
    Find the best attribute to split.
    :param data: Dataset
    :param label: Label / class of dataset
    :param features: Features under consideration
    :return: Summary stats for each feature considered sorted by gain ratio
    Example output:
                      entropy      gain  split_info  gain_ratio
        Outlook      0.940286  0.246750    1.577406    0.156428
        Humidity     0.940286  0.151836    1.000000    0.151836
        Wind         0.940286  0.048127    0.985228    0.048849
        Temperature  0.940286  0.029223    1.556657    0.018773
    """
    entropy = compute_entropy(get_feature_counts(data, label))
    entropy_stats = c.OrderedDict()
    for feature in features:
        feature_counts = get_feature_counts(data, label, feature)
        unweighted_entropies = compute_unweighted_entropies(feature_counts)
        expected_entropy = compute_expected_entropy(unweighted_entropies, feature_counts, feature)
        gain = compute_gain(entropy, expected_entropy)
        split_info = compute_split_info(feature_counts, feature)
        gain_ratio = compute_gain_ratio(gain, split_info)
        entropy_stats[feature] = dict(entropy=entropy, gain=gain, split_info=split_info, gain_ratio=gain_ratio)
    return pd.DataFrame(entropy_stats).transpose().sort_values(by="gain_ratio", ascending=False)


def select_best_feature(entropy_stats: pd.DataFrame) -> str:
    """
    Select best feature among candidate features.
    :param entropy_stats: Table of entropy statistics sorted by gain ratio
    :return: Best feature on the basis of gain ratio
    """
    return entropy_stats.index.values[0]
