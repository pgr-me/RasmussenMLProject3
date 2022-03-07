#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Standard library imports
import collections as c
import typing as t

# Third party libraries
import numpy as np
import pandas as pd


def compute_entropy(feature_counts: pd.Series) -> float:
    """
    Compute entropy of a node.
    :param feature_counts: Node data
    :return: Node entropy
    """
    _, label = feature_counts.index.names
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
    return gain / split_info


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


def discretize_numeric(data: pd.DataFrame, feature: str, quantiles: list) -> tuple:
    """
    Discretize numeric column so that it can be treated as a categorical column.
    :param data: Data providing the numeric feature to be discretized
    :param feature: Numeric feature to discretize
    :param quantiles: List of quantiles to segment numeric data
    :return: Two-class categorical column and dict mapping of quantile-to-cutoff value
    Note: Two classes created at a time but same numeric feature could be reused elsewhere.
    Note: The "left" class includes values less than or equal to cutoff value.
    Example discretized_features output:
                Numeric_0.1 Numeric_0.2 Numeric_0.4 Numeric_0.5 Numeric_0.6  \
        Example
        1              left        left        left        left        left
        2             right        left        left        left        left
        3             right       right       right       right       right
        4             right       right       right       right       right
        5             right       right       right       right        left
        6             right       right        left        left        left
        7             right       right        left        left        left
        8             right       right        left        left        left
        9             right       right       right        left        left
        10            right       right       right       right       right
    Example cut_dict output:
        OrderedDict([(0.1, 0.065),
                     (0.2, 0.16000000000000003),
                     (0.4, 0.72),
                     (0.5, 0.825),
                     (0.6, 0.858),
                     (0.8, 0.92),
                     (0.9, 0.957)])
    """
    quantile_bins = data["Numeric"].quantile(quantiles)
    discretized_features: t.Union[list, pd.DataFrame] = []
    cut_dict = c.OrderedDict()
    for quantile in quantiles:
        col_name = f"{feature}_{quantile}"
        bins = [-float("inf"), quantile_bins.loc[quantile], float("inf")]
        labels = ["left", "right"]
        cut, cut_val = pd.cut(data[feature], bins=bins, labels=labels, retbins=True)
        cut.rename(col_name, inplace=True)
        discretized_features.append(cut)
        cut_dict[quantile] = cut_val[1]
    discretized_features = pd.concat(discretized_features, axis=1)
    return discretized_features, cut_dict


def get_feature_counts(data: pd.DataFrame, label, feature, ct_col="ct"):
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
    return feature_counts.groupby([feature, label])[ct_col].sum().sort_index()


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
    entropy_stats = c.OrderedDict()
    for feature in features:
        feature_counts = get_feature_counts(data, label, feature)
        entropy = compute_entropy(feature_counts)
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
