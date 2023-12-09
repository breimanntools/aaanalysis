"""
This is a script for the frontend of the SequenceFeature class, a supportive class for the CPP feature engineering.
"""
from typing import Optional, Union, List
import warnings
import pandas as pd

import aaanalysis as aa
import aaanalysis.utils as ut

from ._backend.cpp._utils_feature import (get_df_parts_, get_positions_, get_amino_acids_,
                                          get_feature_matrix_, get_df_pos_)
from ._backend.cpp.sequence_feature import (get_split_kws_, get_features_, get_feature_names_, get_df_feat_)
from ._backend.check_feature import check_split_kws


# I Helper Functions
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


# Check functions features
def _get_missing_elements(df_parts=None, scale_elements=None, accept_gaps=False):
    """Get missing elements"""
    seq_elements = set("".join(df_parts.values.flatten()))
    if accept_gaps:
        missing_elements = [x for x in seq_elements if x not in scale_elements and x != ut.STR_AA_GAP]
    else:
        missing_elements = [x for x in seq_elements if x not in scale_elements]
    return missing_elements


# Check matching functions
def check_match_df_scales_df_parts(df_scales=None, df_parts=None, accept_gaps=False):
    """Check if df_scales has values for all Letters in sequences from df_parts"""
    args = dict(df_parts=df_parts, scale_elements=list(df_scales.index), accept_gaps=accept_gaps)
    missing_elements = _get_missing_elements(**args)
    if len(missing_elements) > 0:
        raise ValueError(f"Scale does not match for following sequence element: {missing_elements}")


def check_match_df_parts_parts(df_parts=None, part=None):
    """Check if part in df_parts"""
    if part.lower() not in list(df_parts):
        raise ValueError("'part' ({}) must be in columns of 'df_parts': {}".format(part, list(df_parts)))

# User warnings
def warn_creation_of_feature_matrix(features=None, df_parts=None, name="Feature matrix"):
    """"""
    n_feat = len(features)
    n_samples = len(df_parts)
    n_vals = n_feat * n_samples
    ut.print_out(f"'{name}' for {n_feat} features and {n_samples} samples will be created")
    if n_vals > 1000 * 1000:
        warning = f"Feature matrix with n={n_vals}>=10^6 values will be created, which will take some time.\n" \
                  "It is recommended to create a feature matrix for a pre-selected number features " \
                  "so that 10^6 values are not exceeded."
        warnings.warn(warning)

# TODO finsih, check input, docstring, testing
# TODO update docstring, e.g., give default parts in docstring
# TODO all check functions in frontend
# II Main Functions
class SequenceFeature:
    """
    Retrieve and create sequence feature components (Part, Split, and Scale).

    Notes
    -----
    Feature Components:
        - Part: A continuous subset of sequence, such as a protein domain (e.g, transmembrane domain of membrane proteins).
        - Split: Continuous or discontinuous subset of a sequence part, such as a segment or a pattern.
        - Scale: A physicochemical scale assigning each amino acid a numerical value (typically min-max-normalized [0-1]).

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

    def __init__(self,
                 verbose: Optional[bool] = None
                 ):
        """
        Parameters
        ----------
        verbose
            If ``True``, verbose outputs are enabled. Global 'verbose' setting is used if 'None'.
        """
        self.verbose = ut.check_verbose(verbose)

    # Part and Split methods
    @staticmethod
    def get_df_parts(df_seq: pd.DataFrame = None,
                     list_parts: Optional[Union[str, List[str]]] = None,
                     all_parts = True,
                     jmd_n_len: Optional[int] = None,
                     jmd_c_len: Optional[int] = None,
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
        .. include:: examples/sf_get_df_parts.rst

        """
        # Check input
        ut.check_parts_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, accept_none_len=True)
        df_seq = ut.check_df_seq(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        # Create df parts
        df_parts = get_df_parts_(df_seq=df_seq, list_parts=list_parts,
                                 jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        return df_parts

    @staticmethod
    def get_split_kws(split_types: Union[str, List[str]] = None,
                      n_split_min: int = 1,
                      n_split_max: int = 15,
                      steps_pattern: int = None,
                      n_min: int = 2,
                      n_max: int = 4,
                      len_max: int = 15,
                      steps_periodicpattern: int = None,
                      ) -> dict:
        """Create dictionary with kwargs for three split types: Segment, Pattern, PeriodicPattern

        Parameters
        ----------
        split_types: list of strings, default = [``Segment``, ``Pattern``, ``PeriodicPattern``]
            Split types for which parameter dictionary should be generated.
        n_split_min: int, default = 1
            Number to specify the greatest ``Segment``. Should be > 0.
        n_split_max: int, default = 15,
            Number to specify the smallest ``Segment``. Should be > ``n_split_min``.
        steps_pattern: list of integers, default = [3, 4, 6, 7, 8]
            Possible steps sizes for ``Pattern``.
        n_min: int, default = 2
            Minimum number of steps for Pattern.
        n_max: int, default = 4
            Maximum number of steps for Pattern.
        len_max: int, default = 10
            Maximum length in amino acid position for Pattern by varying start position.
        steps_periodicpattern: list of integers, default = [3, 4]
            Step sizes for PeriodicPattern.


        Returns
        -------
        split_kws: dict
            Nested dictionary with parameters for chosen split_types:

            - Segment: {n_split_min:1, n_split_max=15}
            - Pattern: {steps=[3, 4], n_min=2, n_max=4, len_max=15}
            - PeriodicPattern: {steps=[3, 4]}

        Examples
        --------
        .. include:: examples/sf_get_split_kws.rst
        """
        # Check input
        # TODO Simplify by using check_split_kws HERE
        split_types = check_split_types(split_types=split_types)
        args_int = dict(n_split_min=n_split_min, n_split_max=n_split_max, n_min=n_min, n_max=n_max, len_max=len_max)
        check_split_int_args(kwargs_int=args_int)
        args_list = dict(steps_pattern=steps_pattern, steps_periodicpattern=steps_periodicpattern)
        check_split_list_args(kwargs_list=args_list)
        # Create kws for splits
        split_kws = get_split_kws_(n_split_min=n_split_min,
                                   n_split_max=n_split_max,
                                   steps_pattern=steps_pattern,
                                   n_min=n_min,
                                   n_max=n_max,
                                   len_max=len_max,
                                   steps_periodicpattern=steps_periodicpattern,
                                   split_types=split_types)
        # Post check
        check_split_kws(split_kws=split_kws)
        return split_kws

    # Feature methods
    def get_df_feat(self,
                    features: ut.ArrayLike1D = None,
                    df_parts: pd.DataFrame = None,
                    labels: ut.ArrayLike1D = None,
                    df_scales: Optional[pd.DataFrame] = None,
                    df_cat: Optional[pd.DataFrame] = None,
                    accept_gaps: bool = False,
                    parametric: bool = False,
                    start: int = 1,
                    tmd_len: int = 20,
                    jmd_c_len: int = 10,
                    jmd_n_len: int = 10,
                    ) -> pd.DataFrame:
        """
        Create feature DataFrame for given features.

        Depending on the provided labels, the DataFrame is created for one of the three following cases:

            1. Group vs group comparison
            2. Sample vs group comparison
            3. Sample vs sample comparison

        For the group vs group comparison, the general feature position will be provided, while the amino acid segments
        and patterns for the respective sample from the test dataset (label = 1) will be given.

        Parameters
        ----------
        features : `array-like, shape (n_features,)`
            Ids of features for which ``df_feat`` should be created.
        df_parts
            DataFrame with sequence parts.
        labels: `array-like, shape (n_samples,)`
            Class labels for samples in ``df_parts``. Should be 1 (test set) and 0 (reference set).
        df_scales
            DataFrame with amino acid scales. Default from :meth:`aaanalysis.load_scales`
            with 'name'='scales_cat'.
        df_cat
            DataFrame with default categories for physicochemical amino acid scales.
            Default from :meth:`aaanalysis.load_categories`
        accept_gaps
            Whether to accept missing values by enabling omitting for computations (if True).
        start
            Position label of first amino acid position (starting at N-terminus). Should contain two unique label values.
        tmd_len
            Length of TMD (>0).
        jmd_n_len
            Length of JMD-N (>=0).
        jmd_c_len
            Length of JMD-C (>=0).

        Returns
        -------
        df_feat
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        See Also
        --------
        * The :meth:`aaanalysis.CPP.run` method for creating and filtering CPP features for discriminating between
          two groups of sequences.
        """
        # Load defaults
        if df_scales is None:
            df_scales = aa.load_scales()
        if df_cat is None:
            df_cat = aa.load_scales(name="scales_cat")
        # Check input
        ut.check_df_parts(df_parts=df_parts)
        ut.check_df_scales(df_scales=df_scales)
        features = ut.check_features(features=features, list_parts=list(df_parts), list_scales=list(df_scales))
        labels = ut.check_labels(labels=labels, vals_requiered=[0, 1])
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        ut.check_bool(name="parametric", val=parametric)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        args_len, _ = ut.check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_match_features_df_parts(features=features, df_parts=df_parts)
        check_match_df_scales_df_parts(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        # User warning
        if self.verbose:
            warn_creation_of_feature_matrix(features=features, df_parts=df_parts, name="df_feat")
        # Get sample difference to reference group
        df_feat = get_df_feat_(features=features, df_parts=df_parts, labels=labels, df_scales=df_scales, df_cat=df_cat,
                               accept_gaps=accept_gaps, parametric=parametric,
                               start=start, jmd_n_len=jmd_n_len, tmd_len=tmd_len, jmd_c_len=jmd_c_len)
        return df_feat

    def feature_matrix(self,
                       features: ut.ArrayLike1D = None,
                       df_parts: pd.DataFrame = None,
                       df_scales: Optional[pd.DataFrame] = None,
                       accept_gaps: bool = False,
                       n_jobs: Optional[int] = None,
                       ) -> ut.ArrayLike2D:
        """Create feature matrix for given feature ids and sequence parts.

        Parameters
        ----------
        features : `array-like, shape (n_features,)`
            Ids of features for which matrix of feature values should be created.
        df_parts
            DataFrame with sequence parts.
        df_scales
            DataFrame with default amino acid scales.
        accept_gaps: bool, default ``False``
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        n_jobs
            The number of jobs to run in parallel. If None, it will be set to the maximum.

        Returns
        -------
        feat_matrix: array-like or sparse matrix, shape (n_samples, n_features)
            Feature values of samples.
        """
        # Load defaults
        if df_scales is None:
            df_scales = aa.load_scales()
        # Check input
        ut.check_df_scales(df_scales=df_scales)
        ut.check_df_parts(df_parts=df_parts)
        features = ut.check_features(features=features, list_parts=list(df_parts), list_scales=list(df_scales))
        ut.check_match_features_df_parts(features=features, df_parts=df_parts)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        check_match_df_scales_df_parts(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        ut.check_number_range(name="j_jobs", val=n_jobs, accept_none=True, min_val=1, just_int=True)
        # User warning
        if self.verbose:
            warn_creation_of_feature_matrix(features=features, df_parts=df_parts)
        # Create feature matrix using parallel processing
        feat_matrix = get_feature_matrix_(features=features,
                                          df_parts=df_parts,
                                          df_scales=df_scales,
                                          accept_gaps=accept_gaps,
                                          n_jobs=n_jobs)
        return feat_matrix

    def get_features(self,
                     list_parts: List[str] = None,
                     all_parts: bool = False,
                     split_kws: dict = None,
                     df_scales: Optional[pd.DataFrame] = None,
                     ) -> List[str]:
        """Create list of all feature ids for given Parts, Splits, and Scales.

        Parameters
        ----------
        list_parts: list of strings (n>=1 parts), default = ["tmd_e", "jmd_n_tmd_n", "tmd_c_jmd_c"]
            Names of sequence parts which should be created (e.g., 'tmd').
        split_kws: dict, default = SequenceFeature.get_split_kws
            Nested dictionary with parameter dictionary for each chosen split_type.
        df_scales
            DataFrame with default amino acid scales.
        all_parts: bool, default = False
            Whether to create DataFrame with all possible sequence parts (if True) or parts given by list_parts.

        Returns
        -------
        features: list of strings
            Ids of all possible features for combination of Parts, Splits, and Scales with form: PART-SPLIT-SCALE

        """
        # Load defaults
        if df_scales is None:
            df_scales = aa.load_scales()
        if split_kws is None:
            split_kws = self.get_split_kws()
        # Check input
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        check_split_kws(split_kws=split_kws)
        ut.check_df_scales(df_scales=df_scales, accept_none=True)
        # Get features
        features = get_features_(list_parts=list_parts, split_kws=split_kws, df_scales=df_scales)
        return features

    @staticmethod
    def get_feature_names(features: ut.ArrayLike1D = None,
                          df_cat: Optional[pd.DataFrame] = None,
                          start: int = 1,
                          tmd_len: int = 20,
                          jmd_c_len: int = 10,
                          jmd_n_len: int = 10,
                          ):
        """Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions]).

        Parameters
        ----------
        features : `array-like, shape (n_features,)`
            List of feature ids.
        df_cat
            DataFrame with default categories for physicochemical amino acid scales
        start
            Position label of first amino acid position (starting at N-terminus).
        tmd_len
            Length of TMD (>0).
        jmd_n_len
            Length of JMD-N (>=0).
        jmd_c_len
            Length of JMD-C (>=0).

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
        # Load defaults
        if df_cat is None:
            df_cat = aa.load_scales(name=ut.STR_SCALE_CAT)
        # Check input
        features = ut.check_features(features=features)
        ut.check_df_cat(df_cat=df_cat)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        args_len, _ = ut.check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_match_features_seq_parts(features=features, **args_len)
        # Get feature names
        feat_names = get_feature_names_(features=features,
                                        df_cat=df_cat,
                                        start=start,
                                        tmd_len=tmd_len,
                                        jmd_c_len=jmd_c_len,
                                        jmd_n_len=jmd_n_len)
        return feat_names

    @staticmethod
    def get_feature_positions(features: ut.ArrayLike1D = None,
                              start: int = 1,
                              tmd_len: int = 20,
                              jmd_n_len: int = 10,
                              jmd_c_len: int = 10,
                              tmd_seq: Optional[str] = None,
                              jmd_n_seq: Optional[str] = None,
                              jmd_c_seq: Optional[str] = None,
                              ) -> ut.ArrayLike1D:
        """Create for features a list of corresponding positions.

        Parameters
        ----------
        features : `array-like, shape (n_features,)`
            List of feature ids.
        start
            Position label of first amino acid position (starting at N-terminus, >=0).
        tmd_len
            Length of TMD (>0).
        jmd_n_len
            Length of JMD-N (>=0).
        jmd_c_len
            Length of JMD-C (>=0).
        tmd_seq
            Sequence of TMD. If given, respective amino acid segments/patterns will be returned instead of positions.
        jmd_n_seq
            Sequence of JMD-N. If given, respective amino acid segments/patterns will be returned instead of positions.
        jmd_c_seq
            Sequence of JMD-C. If given, respective amino acid segments/patterns will be returned instead of positions.

        Returns
        -------
        list_pos | list_aa
            List of positions or amino acids for each feature.

        """
        # Check input
        features = ut.check_features(features=features)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        args_len, args_seq = ut.check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                                tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        ut.check_match_features_seq_parts(features=features, **args_seq, **args_len)
        # Get feature position
        if args_seq["tmd_seq"] is not None:
            list_aa = get_amino_acids_(features=features, **args_seq)
            return list_aa
        else:
            list_pos = get_positions_(features=features, start=start, **args_len)
            return list_pos

    # TODO change against concrete function
    @staticmethod
    def get_df_pos(df_feat: pd.DataFrame = None,
                   col_value: str = "mean_dif",
                   col_scale: str = "category",
                   start: int = 1,
                   tmd_len: int = 20,
                   jmd_n_len: int = 10,
                   jmd_c_len: int = 10,
                   list_parts: Optional[Union[str, List[str]]] = None,
                   normalize=False):
        """
        Create DataFrame of aggregated (mean or sum) feature values per residue position and scale.
        """
        # Check input
        ut.check_list_parts(list_parts=list_parts)
        ut.check_df_feat(df_feat=df_feat, list_parts=list_parts)
        if col_scale not in ut.COLS_FEAT_SCALES:
            raise ValueError(f"'col_scale' {col_scale} should be one of the following: {ut.COLS_FEAT_SCALES}")
        cols_feat = ut.COLS_FEAT_STAT + ut.COLS_FEAT_WEIGHT
        if col_value not in cols_feat:
            raise ValueError(f"'col_value' {col_value} should be one of the following: {cols_feat}")
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        args_len, _ = ut.check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE], **args_len)
        # Get df pos
        stop = start + jmd_n_len + tmd_len + jmd_c_len - 1
        value_type = ut.DICT_VALUE_TYPE[col_value]
        df_pos = get_df_pos_(df_feat=df_feat, y=col_scale, value_type=value_type, col_value=col_value,
                             start=start, stop=stop)
        if normalize:
            df_pos = df_pos / abs(df_pos).sum().sum() * 100
        if list_parts is not None:
            pass
            # TODO translate positions onto parts
        return df_pos