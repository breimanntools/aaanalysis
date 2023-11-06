"""
Script for SequenceFeature() object that combines scales, splits, and parts to create
    feature names, feature values, or a feature matrix for ML or CPP pipelines.
"""
import math
import warnings
import pandas as pd

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.cpp.sequence_feature import (get_df_parts, get_split_kws,
                                            get_features, feat_matrix, get_feat_names)


# I Helper Functions
# Check load functions
def check_clustered(complete=False, clust_th=0.7):
    """Check input for loading functions"""
    if not complete and clust_th not in [0.9, 0.7, 0.5, 0.3]:
        raise ValueError("'clust_th' should be 0.3, 0.5, 0.7, or 0.9")


# Check functions get_split_kws
def check_split_types(split_types=None):
    """Check split_type"""
    if type(split_types) is str:
        split_types = [split_types]
    list_split_types = [ut.STR_SEGMENT, ut.STR_PATTERN, ut.STR_PERIODIC_PATTERN]
    if split_types is None:
        split_types = list_split_types
    if not set(list_split_types).issuperset(set(split_types)):
        raise ValueError(f"'split_types'({split_types}) must be in {list_split_types}")
    return split_types


def check_split_int_args(kwargs_int=None):
    """Check type of given arguments"""
    for arg in kwargs_int:
        arg_val = kwargs_int[arg]
        ut.check_number_range(name=arg, val=arg_val, just_int=False)


def check_split_list_args(kwargs_list=None, accept_none=True):
    """Check type of given arguments"""
    for arg in kwargs_list:
        arg_val = kwargs_list[arg]
        if not (accept_none and arg_val is None):
            if type(arg_val) != list:
                raise ValueError(f"'{arg}' ({arg_val}) should be list with non-negative integers")
            else:
                for i in arg_val:
                    if type(i) != int or i < 0:
                        raise ValueError(f"Elements in '{arg}' ({arg_val}) should be non-negative integer")


# Check functions feature values
def _get_missing_elements(df_parts=None, scale_elements=None, accept_gaps=False):
    """Get missing elements"""
    seq_elements = set("".join(df_parts.values.flatten()))
    if accept_gaps:
        missing_elements = [x for x in seq_elements if x not in scale_elements and x != ut.STR_AA_GAP]
    else:
        missing_elements = [x for x in seq_elements if x not in scale_elements]
    return missing_elements


def check_dict_scale(dict_scale=None, df_parts=None, accept_gaps=False):
    """Check if dict_scale is dictionary with numerical values"""
    if not isinstance(dict_scale, dict):
        raise ValueError("'dict_scale' must be a dictionary with values of type float or int")
    if accept_gaps:
        f = lambda key: type(dict_scale[key]) not in [float, int]
    else:
        f = lambda key: type(dict_scale[key]) not in [float, int] or math.isnan(dict_scale[key])
    wrong_type = [(key, dict_scale[key]) for key in dict_scale if f(key)]
    if len(wrong_type) > 0:
        error = "'dict_scale' must be a dictionary with values of type float or int." \
                "\n Following key-value pairs are not accepted: {}".format(wrong_type)
        raise ValueError(error)
    # Check matching of scale to sequences of df_parts
    args = dict(df_parts=df_parts, scale_elements=list(dict_scale.keys()), accept_gaps=accept_gaps)
    missing_elements = _get_missing_elements(**args)
    if len(missing_elements) > 0:
        raise ValueError(f"Scale does not match for following sequence element: {missing_elements}")


# Check functions feature matrix
def check_df_scales_matches_df_parts(df_scales=None, df_parts=None, accept_gaps=False):
    """Check if df_scales has values for all Letters in sequences from df_parts"""
    args = dict(df_parts=df_parts, scale_elements=list(df_scales.index), accept_gaps=accept_gaps)
    missing_elements = _get_missing_elements(**args)
    if len(missing_elements) > 0:
        raise ValueError(f"Scale does not match for following sequence element: {missing_elements}")


def check_parts_in_df_parts(df_parts=None, part=None):
    """Check if part in df_parts"""
    if part.lower() not in list(df_parts):
        raise ValueError("'part' ({}) must be in columns of 'df_parts': {}".format(part, list(df_parts)))


# II Main Functions
class SequenceFeature:
    """Retrieve and create sequence feature components (Part, Split, and Scale).

    Notes
    -----
    Feature Components:
    * Part: A continuous subset of sequence, such as a protein domain (e.g, transmembrane domain of membrane proteins).
    * Split: Continuous or discontinuous subset of a sequence part, such as a segment or a pattern.
    * Scale: A physicochemical scale assigning each amino acid a numerical value (typically min-max-normalized [0-1]).

    Feature: Part + Split + Scale
        Physicochemical property (expressed as numerical scale) present at distinct amino acid
        positions within a protein sequence. The positions are obtained by splitting sequence parts
        into segments or patterns.

    Feature value: Realization of a Feature
        For a given sequence, a feature value is the average of a physicochemical scale over
        all amino acids obtained by splitting a sequence part.

    List of valid sequence parts:
        ['tmd', 'tmd_e', 'tmd_n', 'tmd_c', 'jmd_n', 'jmd_c', 'ext_c', 'ext_n',
        'tmd_jmd', 'jmd_n_tmd_n', 'tmd_c_jmd_c', 'ext_n_tmd_n', 'tmd_c_ext_c']
    """

    # Part and Split methods
    @staticmethod
    def get_df_parts(df_seq=None,
                     list_parts=None,
                     jmd_n_len=None,
                     jmd_c_len=None,
                     ext_len=None,
                     all_parts=False
                     ) -> pd.DataFrame:
        """Create DataFrane with sequence parts.

        Parameters
        ----------
        df_seq: :class:`pandas.DataFrame`
            DataFrame with sequence information comprising either sequence ('sequence', 'tmd_start', 'tmd_stop')
            or sequence part ('jmd_n', 'tmd', 'jmd_c') columns.
        list_parts: list of string, len>=1
            Names of sequence parts which should be created (e.g., 'tmd').
        jmd_n_len: int, default = None, optional
            Length of JMD-N in number of amino acids. If None, 'jmd_n' column must be given in df_seq.
        jmd_c_len: int, default = None, optional
            Length of JMD-N in number of amino acids. If None, 'jmd_c' column must be given in df_seq.
        ext_len: int, default = 4
            Lenght of N- resp. C-terminal extra part of TMD.
        all_parts: bool, default = False
            Whether to create DataFrame with all possible sequence parts (if True) or parts given by list_parts.

        Returns
        -------
        df_parts: :class:`pandas.DataFrame`
            DataFrame with sequence parts.

        Notes
        -----
        List of valid sequence parts can be found in :class: ´aaanalysis.SequenceFeature´.

        Examples
        --------
        Get sequence parts from df_seq with 'tmd_e', and 'tmd_jmd' as parts and jmd length of 10:

        >>> import aaanalysis as aa
        >>> sf = aa.SequenceFeature()
        >>> df_seq = aa.load_dataset(name='DOM_GSEC')
        >>> df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd_e", "tmd_jmd"], jmd_n_len=10, jmd_c_len=10)
        """
        # Check input
        ut.check_args_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len, accept_tmd_none=True)
        df_seq = ut.check_df_seq(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        # Create df parts
        df_parts = get_df_parts(df_seq=df_seq, list_parts=list_parts,
                                jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
        return df_parts

    @staticmethod
    def get_split_kws(n_split_min=1,
                      n_split_max=15,
                      steps_pattern=None,
                      n_min=2,
                      n_max=4,
                      len_max=15,
                      steps_periodicpattern=None,
                      split_types=None):
        """Create dictionary with kwargs for three split types: Segment, Pattern, PeriodicPattern

        Parameters
        ----------
        n_split_min: int, default = 1
            Number greater 0 to specify the greatest Segment (e.g., 1/1 TMD alias whole TMD sequence).
        n_split_max: int, default = 15,
            Number greater n_split_min to specfiy the smallest Segment (e.g., 1/15 TMD).
        steps_pattern: list of integers, default = [3, 4, 6, 7, 8]
            Possible steps sizes for Pattern.
        n_min: int, default = 2
            Minimum number of steps for Pattern.
        n_max: int, default = 4
            Maximum number of steps for Pattern.
        len_max: int, default = 10
            Maximum length in amino acid position for Pattern by varying start position.
        steps_periodicpattern: list of integers, default = [3, 4]
            Step sizes for PeriodicPattern.
        split_types: list of strings, default = ["Segment", "Pattern" "PeriodicPattern"]
            Split types for which paramter dictionary should be generated.

        Returns
        -------
        split_kws: dict
            Nested dictionary with parameters for chosen split_types:

            a) Segment: {n_split_min:1, n_split_max=15}
            b) Pattern: {steps=[3, 4], n_min=2, n_max=4, len_max=15}
            c) PeriodicPattern: {steps=[3, 4]}

        Examples
        --------
        Get default arguments for all splits types (Segment, Pattern, PeriodicPattern):

        >>> import aaanalysis as aa
        >>> sf = aa.SequenceFeature()
        >>> split_kws = sf.get_split_kws()

        Get default argumetns for Segment split:

        >>> import aaanalysis as aa
        >>> sf = aa.SequenceFeature()
        >>> split_kws = sf.get_split_kws(split_types="Segment")
        """
        # Check input
        split_types = check_split_types(split_types=split_types)
        args_int = dict(n_split_min=n_split_min, n_split_max=n_split_max, n_min=n_min, n_max=n_max, len_max=len_max)
        check_split_int_args(kwargs_int=args_int)
        args_list = dict(steps_pattern=steps_pattern, steps_periodicpattern=steps_periodicpattern)
        check_split_list_args(kwargs_list=args_list)
        # Create kws for splits
        split_kws = get_split_kws(n_split_min=n_split_min,
                                  n_split_max=n_split_max,
                                  steps_pattern=steps_pattern,
                                  n_min=n_min,
                                  n_max=n_max,
                                  len_max=len_max,
                                  steps_periodicpattern=steps_periodicpattern,
                                  split_types=split_types)
        # Post check
        ut.check_split_kws(split_kws=split_kws)
        return split_kws

    # Feature methods
    @staticmethod
    def feat_matrix(features=None,
                    df_parts=None,
                    df_scales=None,
                    accept_gaps=False,
                    n_jobs=None,
                    verbose=False,
                    return_labels=False
                    ):
        """Create feature matrix for given feature ids and sequence parts.

        Parameters
        ----------
        features: str, list of strings, pd.Series
            Ids of features for which matrix of feature values should be created.
        df_parts: :class:`pandas.DataFrame`
            DataFrame with sequence parts.
        df_scales: :class:`pandas.DataFrame`, optional
            DataFrame with default amino acid scales.
        accept_gaps: bool, default = False
            Whether to accept missing values by enabling omitting for computations (if True).
        n_jobs: int, default = None,
            The number of jobs to run in parallel. If None, it will be set to the maximum.
        verbose: bool, default = True
            Whether to print size of to be created feature matrix (if True) or not otherwise.
        return_labels: bool, default = False
            Whether to return sample labels in addition to feature matrix.

        Returns
        -------
        feat_matrix: array-like or sparse matrix, shape (n_samples, n_features)
            Feature values of samples.
        """
        # Check input
        ut.check_number_range(name="j_jobs", val=n_jobs, accept_none=True, min_val=1, just_int=True)
        if df_scales is None:
            df_scales = aa.load_scales()
        ut.check_df_scales(df_scales=df_scales)
        ut.check_df_parts(df_parts=df_parts)
        features = ut.check_features(features=features, parts=df_parts, df_scales=df_scales)
        check_df_scales_matches_df_parts(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        if verbose:
            n_feat = len(features)
            n_samples = len(df_parts)
            n_vals = n_feat * n_samples
            print(f"Feature matrix for {n_feat} features and {n_samples} samples will be created")
            if n_vals > 1000 * 1000:
                warning = f"Feature matrix with n={n_vals}>=10^6 values will be created, which will take some time.\n" \
                          "It is recommended to create a feature matrix for a pre-selected number features " \
                          "so that 10^6 values are not exceeded."
                warnings.warn(warning)
        # Create feature matrix using parallel processing
        _feat_matrix, labels = feat_matrix(features=features,
                                           df_parts=df_parts,
                                           df_scales=df_scales,
                                           accept_gaps=accept_gaps,
                                           n_jobs=n_jobs)
        if return_labels:
            if verbose:
                print("Tuple of (feat_matrix, labels) will be returned")
            return _feat_matrix, labels  # X, y
        else:
            if verbose:
                print("Only feat_matrix (without labels) will be returned")
            return _feat_matrix  # X

    def get_features(self,
                     list_parts=None,
                     split_kws=None,
                     df_scales=None,
                     all_parts=False
                     ):
        """Create list of all feature ids for given Parts, Splits, and Scales

        Parameters
        ----------
        list_parts: list of strings (n>=1 parts), default = ["tmd_e", "jmd_n_tmd_n", "tmd_c_jmd_c"]
            Names of sequence parts which should be created (e.g., 'tmd').
        split_kws: dict, default = SequenceFeature.get_split_kws
            Nested dictionary with parameter dictionary for each chosen split_type.
        df_scales: :class:`pandas.DataFrame`, default = SequenceFeature.load_scales
            DataFrame with default amino acid scales.
        all_parts: bool, default = False
            Whether to create DataFrame with all possible sequence parts (if True) or parts given by list_parts.

        Returns
        -------
        features: list of strings
            Ids of all possible features for combination of Parts, Splits, and Scales with form: PART-SPLIT-SCALE

        """
        # Check input
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        ut.check_split_kws(split_kws=split_kws)
        ut.check_df_scales(df_scales=df_scales, accept_none=True)
        # Load defaults
        if df_scales is None:
            df_scales = aa.load_scales()
        if split_kws is None:
            split_kws = self.get_split_kws()
        # Get features
        features = get_features(list_parts=list_parts, split_kws=split_kws, df_scales=df_scales)
        return features

    @staticmethod
    def get_feat_names(features=None,
                       df_cat=None,
                       tmd_len=20,
                       jmd_c_len=10,
                       jmd_n_len=10,
                       ext_len=0,
                       start=1
                       ):
        """Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions]).

        Parameters
        ----------
        features: str, list of strings, pd.Series
            Ids of features for which feature names should be created.
        df_cat: :class:`pandas.DataFrame`, default = SequenceFeature.load_categories
            DataFrame with default categories for physicochemical amino acid scales
        tmd_len: int, >0
            Length of TMD.
        jmd_n_len: int, >0
            Length of JMD-N.
        jmd_c_len: int, >0
            Length of JMD-C.
        ext_len:int, >0
            Length of TMD-extending part (starting from C and N terminal part of TMD).
            Conditions: ext_len<jmd_m_len and ext_len<jmd_c_len
        start: int, >=0
            Position label of first amino acid position (starting at N-terminus).

        Returns
        -------
        feat_names: list of strings
            Names of features.

        Notes
        -----
        Positions are given depending on the three split types:
            - Segment: [first...last]
            - Pattern: [all positions]
            - PeriodicPattern: [first..step1/step2..last]
        """
        # Check input
        features = ut.check_features(features=features)
        ut.check_df_cat(df_cat=df_cat)
        if df_cat is None:
            df_cat = aa.load_scales(name=ut.STR_SCALE_CAT)
        # Get feature names
        feat_names = get_feat_names(features=features,
                                    df_cat=df_cat,
                                    tmd_len=tmd_len,
                                    jmd_c_len=jmd_c_len,
                                    jmd_n_len=jmd_n_len,
                                    ext_len=ext_len,
                                    start=start)
        return feat_names
