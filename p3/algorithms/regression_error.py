#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Standard library imports
import collections as c

# Third party libraries
import numpy as np
import pandas as pd


def compute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute mean squared error between y-true and y-prediction.
    :param y_true: Series of true values
    :param y_pred: Series of predicted values
    :return: Mean squared error
    """
    return np.square(y_true - y_pred).sum() / len(y_true)


def make_thetas(data: pd.DataFrame, label: str, rel_err_li: list) -> dict:
    thetas = {}
    for rel_err in rel_err_li:
        thetas[rel_err] = np.square(data[label].abs() - data[label].abs() * (1 + rel_err)).sum() / len(data)
    return thetas


def select_best_feature(summary_stats: pd.DataFrame) -> str:
    return summary_stats.groupby("feature")["mse"].sum().sort_values().index[0]


def split_attribute(data, label, features) -> pd.DataFrame:
    """
    Find the best attribute to split.
    :param data: Dataset
    :param label: Y / target of dataset
    :param features: Features under consideration
    :return: Summary stats for each feature considered sorted by gain ratio
    """
    error_stats = []
    for feature in features:
        branch_stats = compute_branch_stats(data, label, feature)
        if len(branch_stats) == len(branch_stats.dropna()):
            branch_stats.index.names = ["branch_cat"]
            branch_stats.reset_index(inplace=True)
            branch_stats["feature"] = feature
            error_stats.append(branch_stats)
    # If there are no error stats then the node is a leaf
    # Just return an empty dataframe
    if not error_stats:
        return pd.DataFrame()
    return pd.concat(error_stats).set_index("feature")


def compute_branch_stats(data, label, feature):
    frame = data.copy()[[feature, label]]
    branch_means = data.groupby(feature)[label].mean().rename("mean")
    branch_counts = data.groupby(feature)[label].count().rename("ct")
    branch_stats = branch_means.to_frame().join(branch_counts)
    frame = frame.merge(branch_stats, left_on=feature, right_index=True)
    frame["err"] = np.square(frame[label].subtract(frame["mean"]))
    sse = frame.groupby(feature)["err"].sum().rename("sse")
    branch_stats = branch_stats.join(sse)
    branch_stats["mse"] = branch_stats["sse"].divide(branch_stats["ct"])

    # Handle cases when there is only one branch to choose from
    # In this case, we will let the node be a leaf

    # branch_stats = branch_stats.reset_index().sort_values(by="mse", ascending=True)
    # mask = branch_stats["mean"].isnull()
    # nan_li = branch_stats[mask]["feature"].tolist()
    # mask = branch_stats["feature"].isin(nan_li)
    # branch_stats = branch_stats[~mask]
    return branch_stats
