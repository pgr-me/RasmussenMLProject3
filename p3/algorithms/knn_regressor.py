#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, knn_regressor.py

This module provides the KNNRegressor= class, which inherits from the KNN base class. KNNRegressor supports three
methods: default, edited, and condensed.

"""
# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p3.algorithms.knn import KNN
from p3.algorithms.utils import gaussian_smoother, minkowski_distances, relative_error


class KNNRegressor(KNN):
    """
    k nearest neighbors regressor.
    """

    def __init__(self, data: pd.DataFrame, k: int, label: str, index: str, method: str = None):
        super().__init__(data, k, label, index, method)

    def make_results(self, test: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
        """
        Join test and prediction data and compute true / false positives / negatives.
        :param test: Test (or validation) dataset
        :param pred: Prediction values
        :return: Results
        """
        results = test.copy()[[self.label]].rename(columns={self.label: "truth"}).join(pred)
        return results

    def predict(self, test_data, sigma: int) -> pd.Series:
        """
        Predict the label value given an observation's k nearest neighbors.
        :test_data: Test / validation data used for prediction
        :return: Predicted values for each test / validation observation
        """
        test_indices = np.repeat(test_data.index.values, len(self.neigh_data))
        train_indices = np.vstack([self.neigh_data.index.values for x in range(len(test_data))]).ravel()

        # Compute distances
        distances = minkowski_distances(test_data.drop(axis=1, labels=self.label).values,
                                        self.neigh_data.drop(axis=1, labels=self.label).values)
        columns = ["test_ix", "neigh_ix", "dist"]
        distances = pd.DataFrame(np.stack([test_indices, train_indices, distances], axis=1), columns=columns)
        distances.loc[:, "test_ix": "neigh_ix"] = distances.loc[:, "test_ix": "neigh_ix"].astype("Int32")

        # Merge distances to neighbor labels
        neigh_labels = self.neigh_data.copy()[[self.label]].rename(columns={self.label: "pred"})
        distances = distances.merge(neigh_labels, left_on="neigh_ix", right_index=True, how="left").sort_values(
            by="dist")

        # Apply the Gaussian kernel smoother
        k_neigh = distances.groupby("test_ix").head(self.k)
        k_neigh.loc[:, "weight"] = k_neigh.groupby("test_ix")["dist"].apply(lambda x: gaussian_smoother(x, sigma))
        k_neigh.loc[:, "weighted_pred"] = k_neigh["pred"].multiply(k_neigh["weight"])

        # Compute the prediction values as the weighted sums of the nearest neighbor label values
        pred = k_neigh.groupby("test_ix")["weighted_pred"].sum().rename("pred")

        return pred

    def modify_lookups(self, sigma: float, epsilon: float) -> pd.Series:
        """
        Modify lookups using either the condensed or edited method.
        :return: Condensed or edited training mask
        """
        for observation in self.observations:
            lookups = self.all_lookups.copy().loc[observation].set_index("obs_2_ix")
            lookups = lookups.join(self.training_mask[self.training_mask]).dropna()
            k_nearest_points = self.get_k_nearest_points(lookups)
            k_nearest_points["weight"] = gaussian_smoother(k_nearest_points.dist, sigma)
            truth = self.data.loc[observation, self.label]
            pred = (k_nearest_points["weight"] * k_nearest_points["pred"]).sum()
            rel_err = relative_error(truth, pred)
            if rel_err > epsilon:
                boolean = False if self.method == "edited" else True
                self.training_mask.loc[observation] = boolean

        return self.training_mask

    def train(self, sigma: float, epsilon: float) -> pd.DataFrame:
        """
        Train the k nearest neighbor classifier.
        Note: We don't need to "do" anything to the training mask if we're not editing or condensing.
        :return: Subsetted training data to be used for prediction
        """
        counter = 0
        if self.method not in [None, "None"]:
            while True:
                prior_training_mask = self.training_mask.copy()
                self.modify_lookups(sigma, epsilon)
                if self.method == "edited" or prior_training_mask.equals(self.training_mask):
                    break
                counter += 1
                if counter > 4:
                    break
        self.neigh_data = self.data.copy()[self.training_mask]
        return self.neigh_data

    @staticmethod
    def score(results: pd.DataFrame) -> dict:
        """
        Return the SSE, RMSE, normalized RMSE, and RMSE interquartile scores.
        :param results: Dataframe of truth and pred label values
        :return: Dictionary of scores keyed by metric
        Example output:
            {'sse': 709.5868721236423,
             'rmse': 5.95645394561077,
             'nrmse': 0.1118995642743971,
             'rmseiqr': 0.24054532857940075}
        """
        diff = results["truth"] - results["pred"]
        sse = (diff ** 2).sum()
        rmse = (sse / len(results)) ** 0.5
        quantile = results["truth"].quantile([0.25, 0.75])
        q3, q1 = quantile[0.75], quantile[0.25]
        iqr = q3 - q1
        nrmse = rmse / (results["truth"].max() - results["truth"].min())
        rmseiqr = rmse / iqr
        return dict(sse=sse, rmse=rmse, nrmse=nrmse, rmseiqr=rmseiqr)
