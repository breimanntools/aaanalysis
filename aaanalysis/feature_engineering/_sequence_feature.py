"""
This is a script for the frontend of the SequenceFeature class, a supportive class for the CPP feature engineering.
"""
from typing import Optional, Union, List
import warnings
import pandas as pd

import aaanalysis.utils as ut
from ._backend.check_feature import (check_split_kws,
                                     check_parts_len,
                                     check_match_features_seq_parts,
                                     check_match_df_seq_jmd_len,
                                     check_match_df_parts_features,
                                     check_match_df_parts_df_scales,
                                     check_df_scales,
                                     check_match_df_scales_features,
                                     check_match_df_scales_df_cat,
                                     check_df_cat,
                                     check_match_df_cat_features)
from ._backend.cpp.utils_feature import (get_df_parts_,
                                         remove_entries_with_gaps_,
                                         replace_non_canonical_aa_,
                                         get_positions_, get_amino_acids_,
                                         get_feature_matrix_, get_df_pos_, get_df_pos_parts_)
from ._backend.cpp.sequence_feature import (get_split_kws_, get_features_, get_feature_names_, get_df_feat_)


# I Helper Functions
def check_split_types(split_types=None):
    """Check if split types valid (Segment, Pattern, or PeriodicPattern)"""
    split_types = ut.check_list_like(name="split_types", val=split_types, accept_str=True, accept_none=True)
    if split_types is None:
        split_types = ut.LIST_SPLIT_TYPES
    wrong_split_type = [x for x in split_types if x not in ut.LIST_SPLIT_TYPES]
    if len(wrong_split_type) > 0:
        raise ValueError(f"Wrong 'split_types' ({wrong_split_type}). Chose from {ut.LIST_SPLIT_TYPES}")
    return split_types


def check_steps(steps=None, steps_name="steps_pattern", len_min=2, fixed_len=False):
    """Sort steps and warn if empty list"""
    if steps is None:
        return steps # Skip tests
    steps = list(sorted(steps))
    if len(steps) < len_min:
        if fixed_len:
            raise ValueError(f"'{steps_name}' ({steps}) should contain exactly {len_min} non-negative integers.")
        else:
            raise ValueError(f"'{steps_name}' ({steps}) should contain >= {len_min} non-negative integers.")
    return steps


def warn_creation_of_feature_matrix(features=None, df_parts=None, name="Feature matrix"):
    """Warn if feature matrix gets too large"""
    n_feat = len(features)
    n_samples = len(df_parts)
    n_vals = n_feat * n_samples
    ut.print_out(f"'{name}' for {n_feat} features and {n_samples} samples will be created.")
    if n_vals > 1000 * 1000:
        warning = f"Feature matrix with n={n_vals}>=10^6 values will be created, which will take some time.\n" \
                  "It is recommended to create a feature matrix for a pre-selected number features " \
                  "so that 10^6 values are not exceeded."
        warnings.warn(warning)


def check_match_labels_label_test_label_ref(labels=None, label_test=1, label_ref=0):
    """Check if labels only contains label_test and label_ref"""
    wrong_labels = [x for x in labels if x not in [label_ref, label_test]]
    unique_wrong_labels = list(set(wrong_labels))
    n_wrong_labels = len(unique_wrong_labels)
    if n_wrong_labels > 0:
        raise ValueError(f"'labels' contains {n_wrong_labels} wrong labels: {unique_wrong_labels}")


def check_match_df_parts_label_test_label_ref(df_parts=None, labels=None, label_test=1, label_ref=0):
    """Check if 'jmd_n', 'tmd', and 'jmd_c' in df_parts if amino acid for label_test or label_ref should be retrieved"""
    list_parts = list(df_parts)
    requiered_parts = ["jmd_n", "tmd", "jmd_c"]
    mask_test = [x == label_test for x in labels]
    mask_ref = [x == label_ref for x in labels]
    if sum(mask_test) == 1:
        missing_parts = [x for x in requiered_parts if x not in list_parts]
        if len(missing_parts) > 0:
            raise ValueError(f"'df_parts' misses '{missing_parts}' parts necessary to retrieve amino acid positions"
                             f" for 'label_test' ({label_test}) if only one sample of it occurs in 'labels'."
                             f"\n Add them to the current parts of 'df_parts': {list_parts}")
    if sum(mask_ref) == 1:
        missing_parts = [x for x in requiered_parts if x not in list_parts]
        if len(missing_parts) > 0:
            raise ValueError(f"'df_parts' misses '{missing_parts}' parts necessary to retrieve amino acid positions"
                             f" for 'label_ref' ({label_ref}) if only one sample of it occurs in 'labels'."
                             f"\n Add them to the current parts of 'df_parts': {list_parts}")


def check_col_cat(col_cat=None):
    """Check if col_cat valid column from df_feat"""
    if col_cat not in ut.COLS_FEAT_SCALES:
        raise ValueError(f"'col_cat' {col_cat} should be one of: {ut.COLS_FEAT_SCALES}")


def check_col_val(col_val=None):
    """Check if col_val valid column from df_feat"""
    cols_feat = ut.COLS_FEAT_STAT + ut.COLS_FEAT_WEIGHT
    if col_val not in cols_feat:
        raise ValueError(f"'col_val' {col_val} should be one of: {cols_feat}")


# II Main Functions
class SequenceFeature:
    """
    Utility feature engineering class using sequences to create :class:`CPP` feature components (**Parts**, **Splits**,
    and  **Scales**) and data structures [Breimann25a]_.

    The three feature components are the primary input for the :class:`aaanalysis.CPP` class and define
    Comparative Physicochemical Profiling (CPP) features.

    Notes
    -----
    Feature Components:
        - **Part**: A continuous subset of a sequence, such as a protein domain.
        - **Split**: Continuous or discontinuous subset of a **Part**, either segment or pattern.
        - **Scale**: A physicochemical scale, i.e., a set of numerical values (typically [0-1]) assigned to amino acids.

    Main Parts:
        We define three main parts from which each other part can be derived from:

        - **TMD (target middle domain)**: Protein domain of interest with varying length.
        - **JMD-N (juxta middle domain N-terminal)**: Protein domain or sequence region directly
          N-terminally next to the TMD, typically set to a fixed length (10 by default).
        - **JMD-C (juxta middle domain C-terminal)**: Protein domain or sequence region directly
          C-terminally next to the TMD, typically set to a fixed length (10 by default).

    Feature: Part + Split + Scale
        Physicochemical property (expressed as numerical scale) present at distinct amino acid
        positions within a protein sequence. The positions are obtained by splitting sequence parts
        into segments or patterns.

    Feature value: Realization of a Feature
        For a given sequence, a feature value is the average of a physicochemical scale over
        all amino acids obtained by splitting a sequence part.

    Valid sequence parts:
        - ``tmd``: Target Middle Domain (TMD).
        - ``tmd_e``: TMD extended N- and C-terminally by a number of residues, defined by the ``ext_len`` configuration option.
        - ``tmd_n``: N-terminal half of the TMD.
        - ``tmd_c``: C-terminal half of the TMD.
        - ``jmd_n``: N-terminal Juxt Middle Domain (JMD).
        - ``jmd_c``: C-terminal JMD.
        - ``ext_c``: Extended C-terminal region.
        - ``ext_n``: Extended N-terminal region.
        - ``tmd_jmd``: Combination of JMD-N, TMD, and JMD-C.
        - ``jmd_n_tmd_n``: Combination of JMD-N and N-terminal half of TMD.
        - ``tmd_c_jmd_c``: Combination of C-terminal half of TMD and JMD-C.
        - ``ext_n_tmd_n``: Extended N-terminal region and N-terminal half of TMD.
        - ``tmd_c_ext_c``: C-terminal half of TMD and extended C-terminal region.

    Default parts:
        The following three parts are provided by default: ``tmd``, ``jmd_n_tmd_n``, ``tmd_c_jmd_c``.

    """

    def __init__(self,
                 verbose: bool = True
                 ):
        """
        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        """
        self.verbose = ut.check_verbose(verbose)

    # Part and Split methods
    def get_df_parts(self,
                     df_seq: pd.DataFrame = None,
                     list_parts: Optional[Union[str, List[str]]] = None,
                     all_parts: bool = False,
                     jmd_n_len: Union[int, None] = 10,
                     jmd_c_len: Union[int, None] = 10,
                     remove_entries_with_gaps: bool = False,
                     replace_non_canonical_aa: bool = False,
                     ) -> pd.DataFrame:
        """
        Create DataFrane with selected sequence parts.

        Parameters
        ----------
        df_seq : pd.DataFrame, shape (n_samples, n_seq_info)
            DataFrame containing an ``entry`` column with unique protein identifiers and sequence information
            in a distinct format: **Position-based**, **Part-based**, **Sequence-based**, or **Sequence-TMD-based**.
        list_parts: list of str, default={``tmd``, ``jmd_n_tmd_n``, ``tmd_c_jmd_c``}
            Names of sequence parts that should be obtained for sequences from ``df_seq``.
        jmd_n_len: int, default=10
            Length of JMD-N in number of amino acids. If ``None``, ``jmd_n`` and ``jmd_c`` should be given.
        jmd_c_len: int, default=10
            Length of JMD-N in number of amino acids. If ``None``, ``jmd_n`` and ``jmd_c`` should be given.
        all_parts: bool, default=False
            Whether to create DataFrame with all possible sequence parts (if ``True``) or parts given by ``list_parts``.
        remove_entries_with_gaps: bool, default=False
            Whether to exclude entries containing missing residues in their sequence parts (if ``True``),
            usually resulting from sequences being too short.
        replace_non_canonical_aa: bool, default=False
            Whether to replace non-canonical amino acids (e.g., 'X') by gap ('-') symbol.

        Returns
        -------
        df_parts: pd.DataFrame, shape (n_samples, n_parts)
            Sequence parts DataFrame.

        See Also
        --------
        * :class:`aaanalysis.SequenceFeature` for definition of parts, and lists of all existing and default parts.

        Notes
        -----
        * If ``ext_len`` in aaanalysis.options is not set to > 0, following parts containing extended tmd are not
          considered for ``all_parts=True``: ['tmd_e', 'ext_c', 'ext_n', 'ext_n_tmd_n', 'tmd_c_ext_c'].
        * ``jmd_n_len`` and ``jmd_c_len`` must be both given, except for the part-based format.

        Formats for ``df_seq`` are differentiated by their respective columns:

        **Position-based format**
            - 'sequence': The complete amino acid sequence.
            - 'tmd_start': Starting positions of the TMD in the sequence.
            - 'tmd_stop': Ending positions of the TMD in the sequence.

        **Part-based format**
            - 'jmd_n': Amino acid sequence for JMD-N.
            - 'tmd': Amino acid sequence for TMD.
            - 'jmd_c': Amino acid sequence for JMD-C.

        **Sequence-TMD-based format**
            - 'sequence' and 'tmd' columns.

        **Sequence-based format**
            - Only the 'sequence' column.

        Examples
        --------
        .. include:: examples/sf_get_df_parts.rst
        """
        # Check input
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        check_parts_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, accept_none_tmd_len=True)
        ut.check_df_seq(df_seq=df_seq)
        ut.check_bool(name="all_parts", val=all_parts)
        ut.check_bool(name="replace_non_canonical_aa", val=replace_non_canonical_aa)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts, accept_none=True)
        df_seq = check_match_df_seq_jmd_len(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        # Create df parts
        df_parts = get_df_parts_(df_seq=df_seq, list_parts=list_parts, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        if remove_entries_with_gaps:
            n_before = len(df_parts)
            df_parts = remove_entries_with_gaps_(df_parts=df_parts)
            n_removed = n_before - len(df_parts)
            if n_removed > 0 and self.verbose:
                warnings.warn(f"{n_removed} entries have been removed from 'df_seq' due to introduced gaps.")
        if replace_non_canonical_aa:
            df_parts = replace_non_canonical_aa_(df_parts=df_parts)
        if len(df_parts) == 0:
            raise ValueError(f"All entries have been removed from 'df_seq'. "
                             f"Reduce 'jmd_n_len' ({jmd_n_len}) and 'jmd_c_len' ({jmd_c_len}) settings.")
        return df_parts

    @staticmethod
    def get_split_kws(split_types: Union[str, List[str]] = None,
                      n_split_min: int = 1,
                      n_split_max: int = 15,
                      steps_pattern: Optional[List[int]] = None,
                      n_min: int = 2,
                      n_max: int = 4,
                      len_max: int = 15,
                      steps_periodicpattern: Optional[List[int]] = None,
                      ) -> dict:
        """
        Create dictionary with kwargs for three split types:

            - **Segment**: continuous sub-sequence.
            - **Pattern**: non-periodic discontinuous sub-sequence
            - **PeriodicPattern**: periodic discontinuous sub-sequence.

        Parameters
        ----------
        split_types: list of str, default=[``Segment``, ``Pattern``, ``PeriodicPattern``]
            Split types for which parameter dictionary should be generated.
        n_split_min: int, default=1
            Number to specify the greatest ``Segment``. Should be > 0.
        n_split_max: int, default=15,
            Number to specify the smallest ``Segment``. Should be > ``n_split_min``.
        steps_pattern: list of int, default=[3, 4], optional
            Possible steps sizes for ``Pattern``. Should contain at least 1 non-negative integers
            if ``Pattern`` split_type is used. If ``None``, default is used.
        n_min: int, default=2
            Minimum number of steps for ``Pattern``. Should be <= ``n_max``.
        n_max: int, default=4
            Maximum number of steps for ``Pattern``. Should be >= ``n_min``.
        len_max: int, default=10
            Maximum length in amino acid position for ``Pattern`` by varying start position.
            Should be > min(``steps_pattern``).
        steps_periodicpattern: list of int, default=[3, 4], optional
            Size of odd and even steps for ``PeriodicPattern``. Should contain two non-negative integers if
            ``PeriodicPattern`` split_type is used. If ``None``, default is used.

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
        split_types = check_split_types(split_types=split_types)
        args_int = dict(n_split_min=n_split_min, n_split_max=n_split_max, n_min=n_min, n_max=n_max, len_max=len_max)
        for name in args_int:
            ut.check_number_range(name=name, val=args_int[name], just_int=False, min_val=1)
        steps_pattern = ut.check_list_like(name="steps_pattern", val=steps_pattern,
                                           accept_none=True, check_all_non_neg_int=True)
        steps_periodicpattern = ut.check_list_like(name="steps_periodicpattern", val=steps_periodicpattern,
                                                   accept_none=True, check_all_non_neg_int=True)
        steps_pattern = check_steps(steps=steps_pattern, steps_name="steps_pattern", len_min=1, fixed_len=False)
        steps_periodicpattern = check_steps(steps=steps_periodicpattern, steps_name="steps_periodicpattern",
                                            len_min=2, fixed_len=True)
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
                    label_test: int = 1,
                    label_ref: int = 0,
                    df_scales: Optional[pd.DataFrame] = None,
                    df_cat: Optional[pd.DataFrame] = None,
                    start: int = 1,
                    tmd_len: int = 20,
                    jmd_c_len: int = 10,
                    jmd_n_len: int = 10,
                    accept_gaps: bool = False,
                    parametric: bool = False,
                    n_jobs: Union[int, None] = 1,
                    ) -> pd.DataFrame:
        """
        Create feature DataFrame for given features.

        Depending on the provided labels, the DataFrame is created for one of the three following cases:

            1. Group vs group comparison
            2. Sample vs group comparison
            3. Sample vs sample comparison

        * For the group vs group comparison, the general feature position will be provided.
        * For sample vs group or sample vs sample comparison, the amino acid segments
          and patterns for the respective sample from the test dataset (label = 1) will be given.

        Parameters
        ----------
        features : array-like, shape (n_features,)
            Ids of features for which ``df_feat`` should be created.
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            DataFrame with sequence parts. Must cover all parts in ``features``.
        labels: array-like, shape (n_samples,)
            Class labels for samples in ``df_parts``. Should contain only two different integer label values,
            representing test and reference group (typically, 1 and 0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        parametric : bool, default=False
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney U test) test for p-value computation.
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if True).
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        * Use parallel processing only for high number of features (>~1000 features per core)
        * For sample vs group or sample vs sample comparison, ``df_parts`` must comprise ``jmd_n``, ``tmd``, and
          ``jmd_c`` sequence parts as well as all parts in features.

        See Also
        --------
        * The :meth:`CPP.run` method for creating and filtering CPP features for discriminating between
          two groups of sequences.

        Examples
        --------
        .. include:: examples/sf_get_df_feat.rst
        """
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        ut.check_df_parts(df_parts=df_parts)
        check_df_scales(df_scales=df_scales)
        features = ut.check_features(features=features, list_parts=list(df_parts), list_scales=list(df_scales))
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref],
                                 len_requiered=len(df_parts), allow_other_vals=False)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_bool(name="parametric", val=parametric)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        check_match_df_parts_features(df_parts=df_parts, features=features)
        check_match_df_scales_features(df_scales=df_scales, features=features)
        check_match_features_seq_parts(features=features, **args_len)
        check_match_df_scales_df_cat(df_scales=df_scales, df_cat=df_cat, verbose=self.verbose)
        df_parts = check_match_df_parts_df_scales(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        check_match_labels_label_test_label_ref(labels=labels, label_test=label_test, label_ref=label_ref)
        check_match_df_parts_label_test_label_ref(df_parts=df_parts, labels=labels,
                                                  label_test=label_test, label_ref=label_ref)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        # User warning
        if self.verbose:
            warn_creation_of_feature_matrix(features=features, df_parts=df_parts, name="df_feat")
        # Get sample difference to reference group
        df_feat = get_df_feat_(features=features, df_parts=df_parts, labels=labels,
                               label_test=label_test, label_ref=label_ref,
                               df_scales=df_scales, df_cat=df_cat,
                               accept_gaps=accept_gaps, parametric=parametric,
                               start=start, jmd_n_len=jmd_n_len, tmd_len=tmd_len, jmd_c_len=jmd_c_len,
                               n_jobs=n_jobs)
        return df_feat

    def feature_matrix(self,
                       features: ut.ArrayLike1D = None,
                       df_parts: pd.DataFrame = None,
                       df_scales: Optional[pd.DataFrame] = None,
                       accept_gaps: bool = False,
                       n_jobs: Union[int, None] = 1,
                       ) -> ut.ArrayLike2D:
        """
        Create feature matrix for given feature ids and sequence parts.

        Parameters
        ----------
        features : array-like, shape (n_features,)
            Ids of features for which matrix of feature values should be created.
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            DataFrame with sequence parts.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores.

        Returns
        -------
        X: array-like , shape (n_samples, n_features)
            Feature matrix containing feature values for samples.

        Notes
        -----
        * Use parallel processing only for high number of features (>~1000 features per core)

        Examples
        --------
        .. include:: examples/sf_feature_matrix.rst
        """
        # Load defaults
        if df_scales is None:
            df_scales = ut.load_default_scales()
        # Check input
        check_df_scales(df_scales=df_scales)
        ut.check_df_parts(df_parts=df_parts)
        features = ut.check_features(features=features, list_parts=list(df_parts), list_scales=list(df_scales))
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        check_match_df_parts_features(df_parts=df_parts, features=features)
        check_match_df_scales_features(df_scales=df_scales, features=features)
        df_parts = check_match_df_parts_df_scales(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        # User warning
        if self.verbose:
            warn_creation_of_feature_matrix(features=features, df_parts=df_parts)
        # Create feature matrix using parallel processing
        X = get_feature_matrix_(features=features,
                                df_parts=df_parts,
                                df_scales=df_scales,
                                accept_gaps=accept_gaps,
                                n_jobs=n_jobs)
        return X

    def get_features(self,
                     list_parts: Optional[List[str]] = None,
                     all_parts: bool = False,
                     split_kws: Optional[dict] = None,
                     list_scales: Optional[List[str]] = None,
                     ) -> List[str]:
        """
        Create list of all feature ids for given Parts, Splits, and Scales.

        Parameters
        ----------
        list_parts: list of str, default=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"]
            Names of sequence parts which should be created (e.g., 'tmd'). Length should be >= 1.
        all_parts: bool, default=False
            Whether to create DataFrame with all possible sequence parts (if ``True``) or parts given by ``list_parts``.
        split_kws : dict, optional
            Dictionary with parameter dictionary for each chosen split_type. Default from :meth:`SequenceFeature.get_split_kws`.
        list_scales : list of str, optional
            Names of scales. Default scales from :meth:`load_scales` with ``name='scales'``.

        Returns
        -------
        features: list of str
            Ids of all possible features for combination of Parts, Splits, and Scales with form: PART-SPLIT-SCALE

        Notes
        -----
        * If ``ext_len`` in aaanalysis.options is not set to > 0, following parts containing extended tmd are not
          considered for ``all_parts=True``: ['tmd_e', 'ext_c', 'ext_n', 'ext_n_tmd_n', 'tmd_c_ext_c'].

        Examples
        --------
        .. include:: examples/sf_get_features.rst
        """
        # Load defaults
        if list_scales is None:
            list_scales = list(ut.load_default_scales())
        if split_kws is None:
            split_kws = self.get_split_kws()
        # Check input
        ut.check_bool(name="all_parts", val=all_parts)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        check_split_kws(split_kws=split_kws)
        list_scales = ut.check_list_like(name="list_scales", val=list_scales, accept_none=False)
        # Get features
        features = get_features_(list_parts=list_parts, split_kws=split_kws, list_scales=list_scales)
        return features

    @staticmethod
    def get_feature_names(features: ut.ArrayLike1D = None,
                          df_cat: Optional[pd.DataFrame] = None,
                          start: int = 1,
                          tmd_len: int = 20,
                          jmd_n_len: int = 10,
                          jmd_c_len: int = 10,
                          ) -> List[str]:
        """
        Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions]).

        Parameters
        ----------
        features : array-like, shape (n_features,)
            List of feature ids (>0).
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).

        Returns
        -------
        feat_names: list of str
            Names of features.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with ids in ``features``.
        * Positions are given depending on the three split types:

            - Segment: [first...last]
            - Pattern: [all positions]
            - PeriodicPattern: [first..step1/step2..last]

        Examples
        --------
        .. include:: examples/sf_get_feature_names.rst
        """
        # Load defaults
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        features = ut.check_features(features=features)
        check_df_cat(df_cat=df_cat)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        check_match_df_cat_features(df_cat=df_cat, features=features)
        check_match_features_seq_parts(features=features, **args_len)
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
        """
        Create for features a list of corresponding positions or amino acids.

        Parameters
        ----------
        features : array-like, shape (n_features,)
            List of feature ids.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        tmd_seq : str, optional
            Sequence of TMD. If given, respective amino acid segments/patterns will be returned instead of positions.
        jmd_n_seq : str, optional
            Sequence of JMD-N. If given, respective amino acid segments/patterns will be returned instead of positions.
        jmd_c_seq : str, optional
            Sequence of JMD-C. If given, respective amino acid segments/patterns will be returned instead of positions.

        Returns
        -------
        list_pos or list_aa
            List of positions or amino acids for each feature.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with ids in ``features``.
        * Length of sequence (``tmd_seq``, ``jmd_n_seq``, ``jmd_c_seq``) must match with ids in ``features``.

        Examples
        --------
        .. include:: examples/sf_get_feature_positions.rst
        """
        # Check input
        features = ut.check_features(features=features)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, args_seq = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                             tmd_seq=tmd_seq, jmd_n_seq=jmd_n_seq, jmd_c_seq=jmd_c_seq)
        check_match_features_seq_parts(features=features, **args_seq, **args_len)
        # Get feature position
        if args_seq["tmd_seq"] is not None:
            list_aa = get_amino_acids_(features=features, **args_seq)
            return list_aa
        else:
            list_pos = get_positions_(features=features, start=start, **args_len)
            return list_pos

    @staticmethod
    def get_df_pos(df_feat: pd.DataFrame = None,
                   col_val: str = "mean_dif",
                   col_cat: str = "category",
                   start: int = 1,
                   tmd_len: int = 20,
                   jmd_n_len: int = 10,
                   jmd_c_len: int = 10,
                   list_parts: Optional[Union[str, List[str]]] = None,
                   normalize : bool = False,
                   ) -> pd.DataFrame:
        """
        Create DataFrame of aggregated (mean or sum) feature values per residue position and scale.

        Parameters
        ----------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.
        col_val : {'abs_auc', 'abs_mean_dif', 'mean_dif', 'std_test', 'std_ref'}, default='mean_dif'
            Column name in ``df_feat`` containing numerical values to ``average``. If feature importance and impact
            are provided as {'feat_importance', 'feat_impact'} columns, their ``sum`` of values is computed.
        col_cat : {'category', 'subcategory', 'scale_name'}, default='category'
            Column name in ``df_feat`` for categorizing the numerical values during aggregation.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        list_parts: str or list of str, optional
            Specific sequence parts to consider for numerical value aggregation.
        normalize : bool, default=False
            If ``True``, normalizes aggregated numerical values to a total of 100%.

        Returns
        -------
        df_pos : pd.DataFrame, shape (n_categories, n_positions)
            DataFrame with aggregated numerical values per position.

        Notes
        -----
        * Length parameters (``tmd_len``, ``jmd_n_len``, ``jmd_c_len``) must match with feature ids in ``df_feat``.

        Examples
        --------
        .. include:: examples/sf_get_df_pos.rst
        """
        # Check input
        list_parts = ut.check_list_parts(list_parts=list_parts, return_default=False, accept_none=True)
        # Do not check for list_parts since df_pos can be obtained for any part
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_col_cat(col_cat=col_cat)
        check_col_val(col_val=col_val)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_bool(name="normalize", val=normalize)
        check_match_features_seq_parts(features=df_feat[ut.COL_FEATURE], **args_len)
        # Get df pos
        stop = start + jmd_n_len + tmd_len + jmd_c_len - 1
        value_type = ut.DICT_VALUE_TYPE[col_val]
        df_pos = get_df_pos_(df_feat=df_feat, col_cat=col_cat, col_val=col_val, value_type=value_type,
                             start=start, stop=stop)
        if normalize:
            df_pos = df_pos / abs(df_pos).sum().sum() * 100
        if list_parts is not None:
            df_pos = get_df_pos_parts_(df_pos=df_pos, value_type=value_type,
                                       start=start, **args_len, list_parts=list_parts)
        return df_pos
