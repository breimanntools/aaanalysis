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

# I Helper Functions
def check_value_type(value_type=None):
    """Check if value type is valid"""
    list_value_type = ["min", "mean", "median", "max"]
    if value_type not in list_value_type:
        raise ValueError(f"'value_type' ('{value_type}') should be on of following: {list_value_type}")

def check_match_df_scales_letter_new(df_scales=None, letter_new=None):
    """Check if new letter not already in df_scales"""
    alphabet = df_scales.index.to_list()
    if letter_new in alphabet:
        raise ValueError(f"Letter '{letter_new}' already exists in alphabet of 'df_scales': {alphabet}")


# TODO add corr filtering


# II Main Functions
class NumericalFeature:
    """
    Utility feature engineering class to process and filter numerical data structures,
    such as amino acid scales or a feature matrix.
    """

    """
    TODO
    @staticmethod
    def comp_correlation():

    @staticmethod
    def filter_correlation():
    @staticmethod
    def scale_coverage():
    """

    @staticmethod
    def extend_alphabet(df_scales : pd.DataFrame = None,
                        letter_new : str = None,
                        value_type : Literal["min", "mean", "median", "max"] = "mean" ,
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
        letter_new : str
            The new letter to be added to the alphabet.
        value_type : {'min', 'mean', 'median', 'max'}, default='mean'
            The type of statistic to compute for the new letter (one of 'min', 'mean', 'median', 'max').

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
        ut.check_str(name="letter_new", val=letter_new)
        check_value_type(value_type=value_type)
        check_match_df_scales_letter_new(df_scales=df_scales, letter_new=letter_new)
        # Compute the statistic for each scale
        if value_type == "min":
            new_values = df_scales.min()
        elif value_type == "mean":
            new_values = df_scales.mean()
        elif value_type == "median":
            new_values = df_scales.median()
        else:
            new_values = df_scales.max()
        # Add the new letter to the DataFrame
        df_scales.loc[letter_new] = new_values
        return df_scales

    """
    @staticmethod
    def merge_alphabet(df_scales : pd.DataFrame = None,
                       letters_to_merge : List[str] = None,
                       letter_new : str = None,
                       value_type: Literal["min", "mean", "median", "max"] = "mean"
                       ) -> pd.DataFrame:
    """
