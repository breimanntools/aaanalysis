"""
This is a script for the frontend of the NumericalFeature class, a supportive class for the CPP feature engineering,
including scale and feature filtering methods.
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, Literal, Dict, Union, List, Tuple, Type

import aaanalysis.utils as ut

from ._backend.check_feature import check_df_scales
from ._backend.num_feat.filter_correlation import filter_correlation_
from ._backend.num_feat.extend_alphabet import extend_alphabet_


# I Helper Functions
def check_match_df_scales_letter_new(df_scales=None, letter_new=None):
    """Check if new letter not already in df_scales"""
    alphabet = df_scales.index.to_list()
    if letter_new in alphabet:
        raise ValueError(f"Letter '{letter_new}' already exists in alphabet of 'df_scales': {alphabet}")


# II Main Functions
class NumericalFeature:
    """
    Utility feature engineering class to process and filter numerical data structures,
    such as amino acid scales or a feature matrix.
    """

    @staticmethod
    def filter_correlation(X: ut.ArrayLike2D,
                           max_cor: float = 0.7
                           ) -> np.array:
        """
        Filter features based on Pearson correlation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix. `Rows` typically correspond to proteins and `columns` to features.
        max_cor : float, default=0.5
            Maximum Pearson correlation [0-1] of feature scales used as threshold for filtering.

        Returns
        -------
        is_selected
            1D boolean array with shape (n_features) indicating which features are selected (True) or not (False).

        Notes
        -----
        * Features in ``X`` should be provided in decreasing order of importance. The first occurring features
          will be kept, while subsequent features that correlate with them will be removed.
        Examples
        --------
        .. include:: examples/nf_filter_correlation.rst
        """
        # Check input
        X = ut.check_X(X=X, min_n_unique_features=2)
        ut.check_number_range(name="min_cor", val=max_cor, min_val=0, max_val=1, just_int=False, accept_none=False)
        # Filter features
        is_selected = filter_correlation_(X, max_cor=max_cor)
        return is_selected

    @staticmethod
    def extend_alphabet(df_scales: pd.DataFrame = None,
                        new_letter: str = None,
                        value_type: Literal["min", "mean", "median", "max"] = "mean",
                        ) -> pd.DataFrame:
        """
        Extend amino acid alphabet of ``df_scales`` by new letter.

        This function adds a new row to the DataFrame, representing the new amino acid letter.
        For each scale (column), it computes a specific statistic (min, mean, median, max) based on the
        values of existing amino acids (rows) and assigns this computed value to the new amino acid.

        Parameters
        ----------
        df_scales : pd.DataFrame, shape (n_letters, n_scales)
            DataFrame of scales with letters typically representing amino acids.
        new_letter : str
            The new letter to be added to the alphabet.
        value_type : {'min', 'mean', 'median', 'max'}, default='mean'
            The type of statistic to compute for the new letter.

        Returns
        -------
        df_scales : pd.DataFrame, shape (n_letters + 1, n_scales)
            DataFrame with the extended alphabet including the new amino acid letter.

        Examples
        --------
        .. include:: examples/nf_extend_alphabet.rst
        """
        # Check input
        df_scales = df_scales.copy()
        check_df_scales(df_scales=df_scales)
        ut.check_str(name="letter_new", val=new_letter)
        ut.check_str_options(name="value_type", val=value_type, accept_none=False,
                             list_str_options=["min", "mean", "median", "max"])
        check_match_df_scales_letter_new(df_scales=df_scales, letter_new=new_letter)
        # Compute the statistic for each scale
        df_scales = extend_alphabet_(df_scales=df_scales, new_letter=new_letter, value_type=value_type)
        return df_scales
