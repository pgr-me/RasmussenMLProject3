#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 3, preprocessing.py

This module provides the Preprocessor class.

"""
# Standard library imports
from pathlib import Path
import collections as c

import typing as t

# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p3.preprocessing.preprocessor import Preprocessor

QUANTILES = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class TreePreprocessor(Preprocessor):
    def __init__(self, dataset_name: str, dataset_meta: dict, data_dir: Path, quantiles=QUANTILES):
        super().__init__(dataset_name, dataset_meta, data_dir)
        self.data: t.Union[pd.DataFrame, None] = None
        self.quantiles = quantiles
        self.cut_dicts = {}

    def discretize(self, quantiles=None) -> pd.DataFrame:
        """
        Discretize each numeric column into a set of binary categories.
        :return: Discretized numeric data
        """
        if quantiles is None:
            quantiles = self.quantiles
        self.set_data_classes()
        data = self.data.copy()
        cut_dicts = {}
        for feature, data_class in self.data_classes.items():
            if data_class in ["numeric", "ordinal"]:
                if data_class == "ordinal":
                    disc_feats = self.discretize_ordinal(data, feature)
                else:
                    disc_feats, cut_dict = self.discretize_numeric(data, feature, quantiles)
                    cut_dicts[feature] = cut_dict
                data = data.drop(axis=1, labels=feature).join(disc_feats)
        self.data = data
        self.cut_dicts = cut_dicts
        self.features = [x for x in self.data if x != self.label]
        self.data_classes = {k: "categorical" for k in self.features}
        return self.data, self.cut_dicts


    def discretize_nontrain(self, nontrain_data: pd.DataFrame)->pd.DataFrame:
        """
        Apply cuts derived from training set to non-training data.
        :param nontrain_data: Tuning / pruning / validation data
        :return: Discretized nontrain data
        """
        # Apply training splits to numeric data
        for numeric_feat, cuts in self.cut_dicts.items():
            for cut in cuts:
                cut_val = float(cut.replace("_", "."))
                col = f"{numeric_feat}__{cut}"
                nontrain_data[col] = "left"
                mask = nontrain_data[numeric_feat] > cut_val
                nontrain_data.loc[mask, col] = "right"
            nontrain_data.drop(axis=1, labels=numeric_feat, inplace=True)

        # Apply training splits to ordinal data
        mask = self.names_meta["data_class"] == "ordinal"
        ordinal_feats = self.names_meta[mask].index.values.tolist()
        for ordinal_feat in ordinal_feats:
            train_cols = self.data.filter(regex=f"^{ordinal_feat}_.+")
            for train_col in train_cols:
                cutoff_val = int(train_col.split("__")[-1])
                nontrain_data[train_col] = "left"
                mask = nontrain_data[ordinal_feat] > cutoff_val
                nontrain_data.loc[mask, train_col] = "right"
            nontrain_data = nontrain_data.drop(axis=1, labels=ordinal_feat)

        return nontrain_data

    @staticmethod
    def discretize_ordinal(data, feature) -> pd.DataFrame:
        """
        Create binary splits at each ordinal value.
        :param data:
        :param feature:
        :return: Discrtized ordinal feature
        Example output:
                    Ordinal_1 Ordinal_2 Ordinal_3 Ordinal_4
            Example
            0           right      left      left      left
            1           right      left      left      left
            2           right     right     right     right
            3           right     right     right     right
            4           right     right     right      left
        """
        discretized_features = []
        feat_vals = sorted([int(x) for x in data[feature].unique()])

        for feat_val in feat_vals[:-1]:  # We don't need the last, max ordinal value
            col_name = f"{feature}__{feat_val}"
            bins = [-float("inf"), feat_val, float("inf")]
            labels = ["left", "right"]
            cut, cut_val = pd.cut(data[feature], bins=bins, labels=labels, retbins=True)
            cut.rename(col_name, inplace=True)
            discretized_features.append(cut)
        discretized_features = pd.concat(discretized_features, axis=1)
        return discretized_features

    @staticmethod
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
            ...             ...         ...         ...         ...         ...
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
        quantile_bins = data[feature].quantile(quantiles)
        discretized_features: t.Union[list, pd.DataFrame] = []
        cut_dict = c.OrderedDict()
        for quantile in quantiles:
            col_name = f"{feature}__{quantile}".replace(".", "_")
            bins = [-float("inf"), quantile_bins.loc[quantile], float("inf")]
            labels = ["left", "right"]
            cut, cut_val = pd.cut(data[feature], bins=bins, labels=labels, retbins=True)
            cut.rename(col_name, inplace=True)
            discretized_features.append(cut)
            cut_dict[str(quantile).replace(".", "_")] = cut_val[1]
        discretized_features = pd.concat(discretized_features, axis=1)
        return discretized_features, cut_dict
