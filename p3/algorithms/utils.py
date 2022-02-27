#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Third party libraries
import numpy as np
import pandas as pd


def compute_classification_scores(results: pd.DataFrame, scores_name: str = None) -> pd.DataFrame:
    """
    Compute classification scores from results set.
    :param results: Results dataframe produced by compute_tp_tn_fp_fn function
    :param scores_name: Optional name of series
    :return: Classification scores
    Example output:
        n       10.000000
        pos      2.000000
        neg      8.000000
        tp       2.000000
        tn       7.000000
        fp       1.000000
        prec     0.666667
        rec      0.875000
        f1       0.756757
        acc      0.900000
    """
    n = len(results)
    pos = results["truth"].sum()
    neg = n - pos
    tp = results["tp"].sum()
    fp = results["fp"].sum()
    tn = results["tn"].sum()
    fn = results["fn"].sum()
    prec = tp / (tp + fp)
    rec = tp / pos
    f1 = 2 * prec * rec / (prec + rec)
    acc = (results["truth"] == results["pred"]).sum()
    scores_dict = dict(n=n, pos=pos, neg=neg, tp=tp, tn=tn, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1, acc=acc)
    return pd.Series(scores_dict, name=scores_name).to_frame()


def compute_tp_tn_fp_fn(results) -> pd.DataFrame:
    """
    Compute truth positives, truth negatives, false positives, and false negatives.
    :param results: Dataframe with columns "truth" and "pred" created by make_results function
    :return: Results dataframe with tp, tn, fp, and fn columns
    Example output:
               truth  pred     tp     tn     fp     fn
        index
        29       1.0   1.0   True  False  False  False
        42       0.0   0.0  False   True  False  False
        88       1.0   1.0   True  False  False  False
    """
    try:
        results["tp"] = (results["truth"] == 1) & (results["pred"] == 1)
        results["tn"] = (results["truth"] == 0) & (results["pred"] == 0)
        results["fp"] = (results["truth"] == 0) & (results["pred"] == 1)
        results["fn"] = (results["truth"] == 1) & (results["pred"] == 0)
    except:
        results["tp"] = 0
        results["tn"] = 0
        results["fp"] = 0
        results["fn"] = 0
    return results


def gaussian_smoother(distances: np.array, sigma: float):
    """
    Apply a Gaussian smoother to create weights for an array of distances.
    :param distances: 1D array of distances
    :param sigma: Spread parameter
    Regarding sigma:
        Lower values weight nearer points more heavily
        Higher values weight points more evenly
    Example output:
        obs_2_ix
        316    0.586337
        219    0.327372
        159    0.086291
    """
    unnormalized_weight = np.exp(-1 / (2 * sigma) * distances)
    return unnormalized_weight / unnormalized_weight.sum()


def minkowski_distance(x, y, p=2):
    """
    Compute the Minkowski distance between a vector and a vector or matrix.
    :param x: x vector
    :param y: y vector or matrix
    :param p: p-norm
    p-norm = 1 for Manhattan distance, 2 for Euclidean distance, etc.
    """
    diff = np.abs(x - y).T
    power = np.power(diff.T, p * np.ones(len(diff))).T
    power_sum = np.sum(power, axis=0)
    return np.power(power_sum.T, 1 / p)


def minkowski_distances(X: np.array, Y: np.array, p: int = 2) -> np.array:
    """
    Compute Minkowski distances between two matrices.
    :param X: Array of points where each row is a point and each column is a dimension
    :param Y: Array of points where each row is a point and each column is a dimension
    :param p: p-norm
    :return: Array of distances between each observation in X and Y
    Output includes same-point distances (e.g., point 1 distance from point 1 distance is computed).
    Output ordered s.t. point 1 (P1) repeated across points 1 to m, P2 repeated across points 1 to m, etc.
    p-norm = 1 for Manhattan distance, 2 for Euclidean distance, etc.
    Example output:
        array([0, 6.38634802, 8.75831689, ..., 7.43018813, 6.41678194])
    """
    row_repeat = np.repeat(X, len(Y), axis=0)
    stacked = np.vstack([Y for y in range(len(X))])
    diff = np.abs(np.subtract(row_repeat, stacked))
    power = np.power(diff, p * np.ones(diff.shape))
    power_sum = np.sum(power, axis=1)
    return np.power(power_sum, np.ones(power_sum.shape) / p)


def relative_error(truth: float, pred: float) -> float:
    """
    Compute the relative error between true and predicted values.
    :param truth: True value
    :param pred: Predicted value
    :return: Relative error
    """
    return abs((truth - pred) / truth)
