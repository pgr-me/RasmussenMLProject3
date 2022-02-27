#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, knn_classifer.py

This module provides the KNNClassifer class, which inherits from the KNN base class. KNNClassifier supports three
methods: default, edited, and condensed.

"""
# Standard library imports
import typing as t

# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p3.algorithms.knn import KNN
from p3.algorithms.utils import compute_classification_scores, compute_tp_tn_fp_fn, minkowski_distances


class KNNClassifier(KNN):
    """
    k nearest neighbors classifier.
    """

    def __init__(self, data: pd.DataFrame, k: int, label: str, index: str, method: str = None):
        super().__init__(data, k, label, index, method)

    def make_results(self, test: pd.DataFrame, pred: pd.Series):
        """
        Join test and prediction data and compute true / false positives / negatives.
        :param test: Test (or validation) dataset
        :param pred: Prediction values
        """
        results = test.copy()[[self.label]].rename(columns={self.label: "truth"}).join(pred)
        return compute_tp_tn_fp_fn(results)

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        """
        Predict the majority class given an observation's k nearest neighbor classes.
        :param test_data: Test data used for prediction
        :return: Majority labels for each observation
        """
        test_indices = np.repeat(test_data.index.values, len(self.neigh_data))
        train_indices = np.vstack([self.neigh_data.index.values for x in range(len(test_data))]).ravel()
        distances = minkowski_distances(test_data.drop(axis=1, labels=self.label).values,
                                        self.neigh_data.drop(axis=1, labels=self.label).values)
        distances = pd.DataFrame(np.stack([test_indices, train_indices, distances], axis=1),
                                 columns=["test_ix", "neigh_ix", "dist"])
        distances.loc[:, "test_ix": "neigh_ix"] = distances.loc[:, "test_ix": "neigh_ix"].astype("Int32")
        neigh_y = self.neigh_data.copy()[[self.label]].rename(columns={self.label: "pred"})

        distances = distances.merge(neigh_y, left_on="neigh_ix", right_index=True, how="left")
        try:
            majority_labels = distances.sort_values(by="dist").groupby("test_ix").head(self.k).groupby("test_ix")[
                "pred"].agg(
                pd.Series.mode)
        except:
            majority_labels = pd.Series([1 for x in range(len(test_data))], index=test_data.index, name="pred")

        return majority_labels

    def modify_lookups(self) -> pd.Series:
        """
        Modify lookups using either the condensed or edited method.
        :return: Condensed or edited training mask
        """
        for observation in self.observations:
            lookups = self.all_lookups.copy().loc[observation].set_index("obs_2_ix")
            lookups = lookups.join(self.training_mask[self.training_mask]).dropna()
            k_nearest_points = self.get_k_nearest_points(lookups)
            pred_label = self.determine_majority_label(k_nearest_points)
            true_label = self.data.copy().loc[observation, self.label]
            if pred_label != true_label:
                boolean = False if self.method == "edited" else True
                self.training_mask.loc[observation] = boolean
        return self.training_mask

    def train(self) -> pd.DataFrame:
        """
        Train the k nearest neighbor classifier.
        Note: We don't need to "do" anything to the training mask if we're not editing or condensing.
        :return: Subsetted training data to be used for prediction
        """
        if self.method not in [None, "None"]:
            while True:
                prior_training_mask = self.training_mask.copy()
                self.modify_lookups()
                if self.method == "edited" or prior_training_mask.equals(self.training_mask):
                    break
        self.neigh_data = self.data.copy()[self.training_mask]
        return self.neigh_data

    @staticmethod
    def determine_majority_label(k_nearest_points: pd.DataFrame,
                                 empty_set_label: t.Union[int, bool, float] = -1) -> t.Union[int, bool, float]:
        """
        Determine the majority label among k nearest neighbors to query point.
        :param k_nearest_points: KNN, distances from query point, indices, & labels
        :param empty_set_label: Empty set label used when there are zero votes for the majority label
        :return: Majority label
        """
        votes = k_nearest_points["pred"].value_counts()
        if len(votes) == 0:
            return empty_set_label
        return votes.index[0]

    @staticmethod
    def score(results: pd.DataFrame, scores_name: str = None) -> pd.DataFrame:
        """
        Compute classification scores from results set.
        :param results: Results dataframe produced by compute_tp_tn_fp_fn function
        :param scores_name: Name of series
        :return: Scores
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
        return compute_classification_scores(results, scores_name=scores_name)
