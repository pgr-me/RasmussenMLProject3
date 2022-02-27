#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, knn.py

This module provides the KNN class, the base class of KNNClassifier and KNNRegressor classes.

"""
# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p3.algorithms.utils import minkowski_distances


class KNN:
    """
    Base class for k nearest neighbors classification and regression models.
    We use a "training mask" as a way to subset the data for edited and condensed methods.
    """

    def __init__(self, data: pd.DataFrame, k: int, label: str, index: str, method: str = None, p: int = 2,
                 random_state: int = 777):
        self.data = data
        self.k = k
        self.label = label
        self.index = index
        methods = [None, "None", "edited", "condensed"]
        if method not in methods:
            raise ValueError(f"Method {method} is not one of {methods}.")
        self.method = method
        self.p = p
        self.random_state = random_state

        self.observations = self.data.index.tolist()

        # Initialize the training mask
        bools = [True for x in range(len(self.data))]
        self.training_mask = pd.Series(bools, index=self.data.index, name="training_mask")

        # If method is condensed, we reverse booleans of training mask
        if method == "condensed":
            self.training_mask.loc[:] = False

        self.training_mask.index.name = index

        # Other variables to initialize
        self.lookup_table: pd.DataFrame = None
        self.all_lookups: pd.DataFrame = None
        self.neigh_data: pd.DataFrame = None

    def get_k_nearest_points(self, lookups: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve k nearest points for selected indices.
        :param lookups: Lookup table
        :return: Dataframe of k nearest points per observation
        Example output:
                      obs_2_ix      dist
            obs_1_ix
            0               15  4.093547
            0               35  4.117682
            ...            ...       ...
            99              92  3.736637
            99              26  3.745070
        """
        return lookups.sort_values(by="dist").head(self.k)

    def get_kth_nearest_point(self, obs_1_ixs: list = None) -> pd.DataFrame:
        """
        Retrieve the kth nearest points from the distance lookup table for selected indices.
        :obs_1_ixs: Select provided indices if not None
        :return: Dataframe of kth nearest point to observation point for all observations
        Example output:
                      obs_2_ix      dist
            obs_1_ix
            1               23  3.807596
            2                4  4.421822
            3               41  4.884305
        """
        frame = self.all_lookups.copy()
        if obs_1_ixs is not None:
            frame = frame.set_index("obs_1_ix").loc[sorted(obs_1_ixs)]
        return frame.groupby("obs_1_ix").nth(self.k - 1)

    def make_lookup_table(self) -> pd.DataFrame:
        """
        Make a lookup table of distances - computed by Minkowski p-norm - among points.
        :return: Distance lookup table sorted by observation and shortest distance to next observation
        Same-point distances are excluded from the output
        Example output:
                      obs_2_ix      dist
            obs_1_ix
            0               15  4.093547
            0               35  4.117682
            ...            ...       ...
            99              78  8.021557
            99              11  9.205464
        """

        frame = self.data.copy().drop(axis=1, labels=self.label, errors="ignore")

        # Create index mappings
        obs_1_indices = np.repeat(frame.index.values, len(frame))
        obs_2_indices = np.vstack([frame.index.values for x in range(len(frame))]).ravel()

        # Compute Minkowski distances for selected p
        distances = minkowski_distances(frame.values, frame.values, p=self.p)

        # Organize into a lookup table
        all_lookups = np.stack([obs_1_indices, obs_2_indices, distances], axis=1)

        all_lookups = pd.DataFrame(all_lookups, columns=["obs_1_ix", "obs_2_ix", "dist"])
        all_lookups.loc[:, "obs_1_ix": "obs_2_ix"] = all_lookups.loc[:, "obs_1_ix": "obs_2_ix"].astype("Int32")

        # Eliminate same observation-same observation distances (which are zero)
        mask = all_lookups["obs_1_ix"] != all_lookups["obs_2_ix"]
        all_lookups = all_lookups[mask].sort_values(by=["obs_1_ix", "dist"])

        # Join obs_2 y's and obs_1 y's and call them pred and truth, respectively
        all_lookups = all_lookups.set_index("obs_2_ix").join(self.data[[self.label]]).reset_index()
        all_lookups.rename(columns={"index": "obs_2_ix", self.label: "pred"}, inplace=True)

        all_lookups.set_index("obs_1_ix", inplace=True)
        all_lookups = all_lookups.join(self.data[[self.label]], how="left").rename(columns={self.label: "truth"})
        all_lookups.index.names = ["obs_1_ix"]

        self.all_lookups = all_lookups.copy().sort_values(by="dist")
        return self.all_lookups
