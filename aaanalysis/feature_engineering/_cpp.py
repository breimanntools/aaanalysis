"""
This is a script for the frontend of the CPP class, a sequence-based feature engineering object.
"""
import numpy as np
import pandas as pd
from typing import Optional, List

import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool

# Import supportive class (exception for importing from same sub-package)
from ._backend.cpp.sequence_feature import get_split_kws_
from ._backend.check_feature import (check_split_kws,
                                     check_parts_len, check_match_df_parts_split_kws,
                                     check_df_scales, check_df_cat, check_match_df_parts_df_scales,
                                     check_match_df_scales_df_cat)
from ._backend.cpp_run import cpp_run_single, cpp_run_batch
from ._backend.cpp.cpp_eval import evaluate_features


# I Helper Functions
def check_sample_in_df_seq(sample_name=None, df_seq=None):
    """Check if sample name in df_seq"""
    list_names = list(df_seq[ut.COL_NAME])
    if sample_name not in list_names:
        error = f"'sample_name' ('{sample_name}') not in '{ut.COL_NAME}' of 'df_seq'." \
                f"\nValid names are: {list_names}"
        raise ValueError(error)


def check_match_list_df_feat_list_df_parts(list_df_feat=None, list_df_parts=None):
    """Check if all elements in list are valid feature DataFrames"""
    for df_feat, df_parts in zip(list_df_feat, list_df_parts):
        ut.check_df_feat(df_feat=df_feat, list_parts=list(df_parts))


def check_match_list_df_feat_names_feature_sets(list_df_feat=None, names_feature_sets=None):
    """Check if length of list_df_feat and names match"""
    if names_feature_sets is None:
        return None # Skip check
    if len(list_df_feat) != len(names_feature_sets):
        raise ValueError(f"Length of 'list_df_feat' ({len(list_df_feat)}) and 'names_feature_sets'"
                         f" ({len(names_feature_sets)} does not match) ")


# II Main Functions
class CPP(Tool):
    """
    Comparative Physicochemical Profiling (**CPP**) class to create and filter features that are most discriminant
    between two sets of sequences [Breimann25a]_.

    CPP aims at identifying a set of non-redundant features that are most discriminant between the
    test and reference group of sequences.

    Attributes
    ----------
    df_parts
        DataFrame with sequence **Parts**.
    split_kws
        Nested dictionary defining **Splits** with parameter dictionary for each chosen split_type.
    df_scales
        DataFrame with amino acid **Scales**.
    df_cat
        DataFrame with categories for physicochemical amino acid **Scales**.
    """
    def __init__(self,
                 df_parts: pd.DataFrame = None,
                 split_kws: Optional[dict] = None,
                 df_scales: Optional[pd.DataFrame] = None,
                 df_cat: Optional[pd.DataFrame] = None,
                 accept_gaps: bool = False,
                 verbose: bool = True,
                 random_state: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            DataFrame with sequence parts.
        split_kws : dict, optional
            Dictionary with parameter dictionary for each chosen split_type. Default from :meth:`SequenceFeature.get_split_kws`.
        df_scales : pd.DataFrame, shape (n_letters, n_scales), optional
            DataFrame of scales with letters typically representing amino acids. Default from :meth:`load_scales`
            unless specified in ``options['df_scales']``.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info), optional
            DataFrame of categories for physicochemical scales. Must contain all scales from ``df_scales``.
            Default from :meth:`load_scales` with ``name='scales_cat'``, unless specified in ``options['df_cat']``.
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.
        random_state : int, optional
            The seed used by the random number generator. If a positive integer, results of stochastic processes are
            consistent, enabling reproducibility. If ``None``, stochastic processes will be truly random.

        Notes
        -----
        * All scales from ``df_scales`` must be contained in ``df_cat``

        See Also
        --------
        * :class:`CPPPlot`: the respective plotting class.
        * :class:`SequenceFeature` for definition of sequence **Parts**.
        * :meth:`SequenceFeature.split_kws` for definition of **Splits** key word arguments.
        * :func:`load_scales` for definition of amino acid **Scales** and their categories.

        Examples
        --------
        .. include:: examples/cpp.rst
        """
        # Load defaults
        if split_kws is None:
            split_kws = get_split_kws_()
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        verbose = ut.check_verbose(verbose)
        random_state = ut.check_random_state(random_state=random_state)
        ut.check_df_parts(df_parts=df_parts)
        check_split_kws(split_kws=split_kws)
        check_df_scales(df_scales=df_scales)
        check_df_cat(df_cat=df_cat)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        df_parts = check_match_df_parts_df_scales(df_parts=df_parts, df_scales=df_scales, accept_gaps=accept_gaps)
        check_match_df_parts_split_kws(df_parts=df_parts, split_kws=split_kws)
        df_scales, df_cat = check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        # Internal attributes
        self._accept_gaps = accept_gaps
        self._verbose = verbose
        self._random_state = random_state
        # Feature components: Scales + Part + Split
        self.df_cat = df_cat.copy()
        self.df_scales = df_scales.copy()
        self.df_parts = df_parts.copy()
        self.split_kws = split_kws

    # Main method
    def run(self,
            labels: ut.ArrayLike1D = None,
            label_test: int = 1,
            label_ref: int = 0,
            n_filter: int = 100,
            n_pre_filter: Optional[int] = None,
            pct_pre_filter: int = 5,
            max_std_test: float = 0.2,
            max_overlap: float = 0.5,
            max_cor: float = 0.5,
            check_cat: bool = True,
            parametric: bool = False,
            start: int = 1,
            tmd_len: int = 20,
            jmd_n_len: int = 10,
            jmd_c_len: int = 10,
            n_jobs: Optional[int] = None,
            vectorized: bool = True,
            n_batches: Optional[int] = None,
            ) -> pd.DataFrame:
        """
        Perform Comparative Physicochemical Profiling (CPP) algorithm: creation and two-step filtering of
        interpretable sequence-based features.

        The aim of the CPP algorithm is to identify a set of unique, non-redundant features that are most
        discriminant between the test and reference group of sequences. See [Breimann25a]_ for details on the algorithm.

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        n_filter : int, default=100
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter : int, optional
            Number of feature to be pre-filtered by CPP algorithm. If ``None``, a percentage of all features is used.
        pct_pre_filter : int, default=5
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test : float, default=0.2
            Maximum standard deviation [>0-<1] within the test group used as threshold for pre-filtering.
        max_overlap : float, default=0.5
            Maximum positional overlap [0-1] of features used as threshold for filtering.
        max_cor : float, default=0.5
            Maximum Pearson correlation [0-1] of feature scales used as threshold for filtering.
        check_cat : bool, default=True
            Whether to check for redundancy within scale categories during filtering.
        parametric : bool, default=False
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney U test) test for p-value computation.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_len : int, default=20
            Length of TMD (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        n_jobs : int, None, or -1, default=None
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores.
        vectorized : bool, default=True
            Whether to apply sequence splitting and the Mann-Whitney U test in 'vectorized' mode (``True``),
            improving speed but increasing memory consumption.
        n_batches : int, None, default=None
            Number of batches (>=2) used for batch processing. If ``None``, single-processing is used, which is faster
            but more memory-intensive. Increasing ``n_batches`` (up to the maximum number of scales in ``df_scales``)
            reduces memory consumption but slows down processing.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_feature_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        * Pre-filtering can be adjusted by the following parameters: {'n_pre_filter', 'pct_pre_filter', 'max_std_test'}.
        * Filtering can be adjusted by the following parameters: {'n_filter', 'max_overlap', 'max_cor', 'check_cat'}.
        * For large datasets (due to long sequences or a high number of samples) or memory-limited systems,
          memory consumption can be reduced by:

          - Disabling vectorized mode (``vectorized=False``)
          - Reducing ``n_jobs`` (down to ``n_jobs=1``)
          - Using batch processing (``n_batches>=2``, with higher values reducing memory usage)

          While this helps to prevent crashes, it may slow down processing.

        * ``df_feat`` contains the following 13 columns, including the unique feature id (1), scale information (2-5),
          statistical results for filtering and ranking (6-12), and feature positions (13):

            1. 'features': Feature ID (PART-SPLIT-SCALE)
            2. 'category': Scale category
            3. 'subcategory': Sub category of scales
            4. 'scale_name': Name of scales
            5. 'scale_description': Description of the scale
            6. 'abs_auc': Absolute adjusted AUC (area under the curve) [-0.5 to 0.5]
            7. 'abs_mean_dif': Absolute mean differences between test and reference group [0 to 1]
            8. 'mean_dif': Mean differences between test and reference group [-1 to 1]
            9. 'std_test': Standard deviation in test group
            10. 'std_ref': Standard deviation in reference group
            11. 'p_val': Non-parametric (mann_whitney) or parametric (ttest_indep) statistic
            12. 'p_val_fdr_bh': Benjamini-Hochberg FDR corrected p-values
            13. 'positions': Feature positions for default settings

        See Also
        --------
        * :func:`comp_auc_adjusted` for details on ``abs_auc``.

        Examples
        --------
        .. include:: examples/cpp_run.rst
        """
        # Check input
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref],
                                 len_requiered=len(self.df_parts), allow_other_vals=False)
        ut.check_number_range(name="n_filter", val=n_filter, min_val=1, just_int=True)
        ut.check_number_range(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True, just_int=True)
        ut.check_number_range(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100, just_int=True)
        ut.check_number_range(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0,
                              just_int=False, exclusive_limits=True)
        ut.check_number_range(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_bool(name="check_cat", val=check_cat)
        ut.check_bool(name="parametric", val=parametric)
        ut.check_number_val(name="start", val=start, just_int=True, accept_none=False)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        ut.check_bool(name="vectorized", val=vectorized)
        n_scales = len(list(self.df_scales))
        ut.check_number_range(name="n_batches", val=n_batches, just_int=True,
                              accept_none=True, min_val=2, max_val=n_scales)
        # Run the CPP algorithm on complete dataset
        args = dict(df_parts=self.df_parts, split_kws=self.split_kws,
                    df_scales=self.df_scales, df_cat=self.df_cat,
                    verbose=self._verbose, accept_gaps=self._accept_gaps,
                    labels=labels, label_test=label_test, label_ref=label_ref,
                    n_filter=n_filter, n_pre_filter=n_pre_filter, pct_pre_filter=pct_pre_filter,
                    max_std_test=max_std_test, max_overlap=max_overlap, max_cor=max_cor,
                    check_cat=check_cat, parametric=parametric,
                    start=start, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                    n_jobs=n_jobs, vectorized=vectorized)
        if n_batches is None:
            df_feat = cpp_run_single(**args)
        else:
            df_feat = cpp_run_batch(**args, n_batches=n_batches)
        # Cconvert all np.str_ to str in object columns
        for col in df_feat.select_dtypes(include="object").columns:
            df_feat[col] = df_feat[col].apply(lambda x: str(x) if isinstance(x, (np.str_, np.generic)) else x)
        return df_feat

    def eval(self,
             list_df_feat: List[pd.DataFrame] = None,
             labels: ut.ArrayLike1D = None,
             label_test: int = 1,
             label_ref: int = 0,
             min_th: float = 0.0,
             names_feature_sets: Optional[List[str]] = None,
             list_cat: Optional[List[str]] = None,
             list_df_parts: Optional[List[pd.DataFrame]] = None,
             n_jobs: Optional[int] = 1,
             ) -> pd.DataFrame:
        """
        Evaluate the quality of different sets of identified CPP features.

        Feature sets are evaluated regarding two quality groups:

        - **Discriminative Power**: The capability of features to distinguish between test and reference datasets.
        - **Redundancy**: Assessed by the optimized number of clusters, based on Pearson correlation among features.

        Parameters
        ----------
        list_df_feat : list of pd.DataFrames
            List of feature DataFrames each of shape (n_features, n_feature_info)
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (typically, test=1, reference=0).
        label_test : int, default=1,
            Class label of test group in ``labels``.
        label_ref : int, default=0,
            Class label of reference group in ``labels``.
        min_th : float, default=0.0
            Pearson correlation threshold for clustering optimization (between -1 and 1).
        names_feature_sets : list of str, optional
            List of names for feature sets corresponding to ``list_df_feat``.
        list_cat : list of str, optional
            List of scale categories to retrieve number of features from. Default:
            ['ASA/Volume', 'Composition', 'Conformation', 'Energy', 'Others', 'Polarity', 'Shape', 'Structure-Activity']
        list_df_parts : list of pd.DataFrames, optional
            List of part DataFrames each of shape (n_samples, n_parts). Must match with ``list_df_feat``.
        n_jobs : int, None, or -1, default=1
            Number of CPU cores (>=1) used for multiprocessing. If ``None``, the number is optimized automatically.
            If ``-1``, the number is set to all available cores.

        Returns
        -------
        df_eval : pd.DataFrame
            Evaluation results for each set of identified features. For each set, statistical
            measures were averaged across all features.

        Notes
        -----
        * ``df_eval`` includes the following columns (upper-case indicates direct reference to ``df_feat`` columns):

            - 'name': Name of the feature set, typically based on CPP run settings, if ``names`` is provided.
            - 'n_features': Tuple with total number of features and list of number of features per scale category from ``list_cat``.
            - 'avg_ABS_AUC': Absolute AUC averaged across all features.
            - 'range_ABS_AUC': Quintile range of absolute AUC among all features (min, 25%, median, 75%, max).
            - 'avg_MEAN_DIF': Tuple of mean differences averaged across all features separately
              for features with positive and negative 'mean_dif'.
            - 'n_clusters': Optimal number of clusters [2,100].
            - 'avg_n_feat_per_clust': Average number of features per cluster.
            - 'std_n_feat_per_clust': Standard deviation of feature number per cluster.

        * 'n_clusters' is optimized for a KMeans clustering model based on the minimum Pearson correlation between
          the cluster center and all cluster members across all clusters (``min_cor_center`` in :class:`AAclust`),
          which has to exceed the minimum correlation threshold ``min_th``.

        See Also
        --------
        * :func:`CPPPlot.eval`: the respective plotting method.
        * :ref:`usage_principles_aaontology` for details on scale categories.
        * :meth:`CPP.run` for details on CPP statistical measures.
        * :func:`comp_auc_adjusted` for details on 'abs_auc'.
        * :class:`sklearn.cluster.KMeans` for employed clustering model.
        * :class:`AAclust` ([Breimann24a]_) for details on cluster optimization using Pearson correlation.

        Examples
        --------
        .. include:: examples/cpp_eval.rst
        """
        # Check input
        list_df_feat = ut.check_list_like(name="list_df_feat", val=list_df_feat, min_len=2)
        ut.check_number_val(name="label_test", val=label_test, just_int=True)
        ut.check_number_val(name="label_ref", val=label_ref, just_int=True)
        labels = ut.check_labels(labels=labels, vals_requiered=[label_test, label_ref],
                                 len_requiered=len(self.df_parts), allow_other_vals=False)
        ut.check_number_range(name="min_th", val=min_th, min_val=-1, max_val=1, just_int=False)
        names_feature_sets = ut.check_list_like(name="names_feature_sets", val=names_feature_sets, accept_none=True,
                                                accept_str=True, check_all_str_or_convertible=True)
        list_cat = ut.check_list_like(name="list_cat", val=list_cat, accept_none=True, accept_str=True,
                                      check_all_str_or_convertible=True)
        n_jobs = ut.check_n_jobs(n_jobs=n_jobs)
        check_match_list_df_feat_names_feature_sets(list_df_feat=list_df_feat,
                                                    names_feature_sets=names_feature_sets)
        list_df_parts = ut.check_list_like(name="list_df_parts", val=list_df_parts, accept_none=True)
        mask_test = [x == label_test for x in labels]
        if list_df_parts is None:
            list_df_parts = [self.df_parts[mask_test]] * len(list_df_feat)
        if list_cat is None:
            list_cat = ut.LIST_CAT
        check_match_list_df_feat_list_df_parts(list_df_feat=list_df_feat, list_df_parts=list_df_parts)
        df_eval = evaluate_features(list_df_feat=list_df_feat,
                                    names_feature_sets=names_feature_sets,
                                    list_cat=list_cat,
                                    list_df_parts=list_df_parts,
                                    df_scales=self.df_scales,
                                    accept_gaps=self._accept_gaps,
                                    min_th=min_th,
                                    n_jobs=n_jobs,
                                    random_state=self._random_state)
        return df_eval
