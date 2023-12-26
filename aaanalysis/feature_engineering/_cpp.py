"""
This is a script for the frontend of the CPP class, a sequence-based feature engineering object.
"""
import numpy as np
import pandas as pd
from typing import Optional

import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool

# Import supportive class (exception for importing from same sub-package)
from ._backend.cpp.sequence_feature import get_features_, get_split_kws_
from ._backend.check_feature import (check_split_kws,
                                     check_parts_len, check_match_features_seq_parts,
                                     check_df_parts, check_match_df_parts_features,
                                     check_match_df_parts_list_parts, check_match_df_parts_split_kws,
                                     check_df_scales, check_match_df_scales_features,
                                     check_df_cat, check_match_df_cat_features,
                                     check_match_df_parts_df_scales, check_match_df_scales_df_cat)
from ._backend.cpp.utils_feature import get_positions_, add_scale_info_
from ._backend.cpp.cpp_run import pre_filtering_info, pre_filtering, filtering, add_stat


# I Helper Functions
# Check for add methods
def check_sample_in_df_seq(sample_name=None, df_seq=None):
    """Check if sample name in df_seq"""
    list_names = list(df_seq[ut.COL_NAME])
    if sample_name not in list_names:
        error = f"'sample_name' ('{sample_name}') not in '{ut.COL_NAME}' of 'df_seq'." \
                f"\nValid names are: {list_names}"
        raise ValueError(error)

# TODO all check functions in frontend (check_steps)
# TODO simplify checks & interface (end-to-end check with tests & docu)
# TODO  TODO add link to explanation for TMD, JMDs
# II Main Functions
class CPP(Tool):
    """
    Create and filter features that are most discriminant between two sets of sequences.

    Attributes
    ----------
    df_parts
        DataFrame with sequence parts.
    split_kws
        Nested dictionary with parameter dictionary for each chosen split_type.
    df_scales
        DataFrame with amino acid scales.
    df_cat
        DataFrame with categories for physicochemical amino acid scales.
    """
    def __init__(self,
                 df_parts: pd.DataFrame = None,
                 split_kws: Optional[dict] = None,
                 df_scales: Optional[pd.DataFrame] = None,
                 df_cat: Optional[pd.DataFrame] = None,
                 accept_gaps: bool = False,
                 verbose: Optional[bool] = None):
        """
        Parameters
        ----------
        df_parts : pd.DataFrame, shape (n_samples, n_parts)
            DataFrame with sequence parts.
        split_kws : dict, optional
            Dictionary with parameter dictionary for each chosen split_type. Default from :meth:`SequenceFeature.get_split_kws`.
        df_scales : pd.DataFrame, shape (n_amino_acids, n_scales)
            DataFrame with amino acid scales. Default from :meth:`load_scales` with ``name='scales'``.
        df_cat : pd.DataFrame, shape (n_scales, n_scales_info)
            DataFrame with categories for physicochemical amino acid scales.
            Default from :meth:`load_scales` with ``name='scales_cat'``.
        accept_gaps : bool, default=False
            Whether to accept missing values by enabling omitting for computations (if ``True``).
        verbose : bool, optional
            If ``True``, verbose outputs are enabled. Global 'verbose' setting is used if ``None``.
        """
        # Load defaults
        if split_kws is None:
            split_kws = get_split_kws_()
        if df_scales is None:
            df_scales = ut.load_default_scales()
        if df_cat is None:
            df_cat = ut.load_default_scales(scale_cat=True)
        # Check input
        check_df_parts(df_parts=df_parts)
        check_split_kws(split_kws=split_kws)
        check_df_scales(df_scales=df_scales)
        check_df_cat(df_cat=df_cat)
        ut.check_bool(name="accept_gaps", val=accept_gaps)
        df_parts = check_match_df_parts_df_scales(df_parts=df_parts, df_scales=df_scales, accept_gaps=accept_gaps)
        df_scales, df_cat = check_match_df_scales_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        check_match_df_parts_split_kws(df_parts=df_parts, split_kws=split_kws)
        # Internal attributes
        self._verbose = ut.check_verbose(verbose)
        self._accept_gaps = accept_gaps
        # Feature components: Scales + Part + Split
        self.df_cat = df_cat.copy()
        self.df_scales = df_scales.copy()
        self.df_parts = df_parts.copy()
        self.split_kws = split_kws

    # Main method
    def run(self,
            labels: ut.ArrayLike1D = None,
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
            n_processes: Optional[int] = None
            ) -> pd.DataFrame:
        """
        Perform CPP pipeline by creation and two-step filtering of features. CPP aims to
        identify a collection of non-redundant features that are most discriminant between
        a test and a reference group of sequences.

        Parameters
        ----------
        labels : array-like, shape (n_samples,)
            Class labels for samples in sequence DataFrame (test=1, reference=0).
        n_filter : int, default=100
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter : int, optional
            Number of feature to be pre-filtered by CPP algorithm. If ``None``, a percentage of all features is used.
        pct_pre_filter : int, default=5
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test : float, default=0.2
            Maximum standard deviation [0-1] within the test group used as threshold for pre-filtering.
        max_overlap : float, default=0.5
            Maximum positional overlap [0-1] of features used as threshold for filtering.
        max_cor : float, default=0.5
            Maximum Pearson correlation [0-1] of features used as threshold for filtering.
        check_cat : bool, default=True
            Whether to check for redundancy within scale categories.
        parametric : bool, default=False
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney-U-test) test for p-value computation.
        start : int, default=1
            Position label of first amino acid position (starting at N-terminus, >=0).
        tmd_len : int, default=20
            Length of TMD (>0).
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len : int, default=10
            Length of JMD-C (>=0).
        n_processes : int, optional
            Number of CPUs used for multiprocessing. If ``None``, number will be optimized automatically.

        Returns
        -------
        df_feat : pd.DataFrame, shape (n_features, n_features_info)
            Feature DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        The feature DataFrame contains the following 11 columns, including the unique feature id (1),
        scale information (2-4), statistical results for filtering and ranking (5-10), and feature positions (11):

        1. features: Feature ID (PART-SPLIT-SCALE)
        2. category: Scale category
        3. subcategory: Sub category of scales
        4. scale_name: Name of scales
        5. abs_auc: Absolute adjusted AUC [-0.5 to 0.5]
        6. abs_mean_dif: Absolute mean differences between test and reference group [0 to 1]
        7. std_test: Standard deviation in test group
        8. std_ref: Standard deviation in reference group
        9. p_val: Non-parametric (mann_whitney) or parametric (ttest_indep) statistic
        10. p_val_fdr_bh: Benjamini-Hochberg FDR corrected p-values
        11. positions: Feature positions for default settings

        """
        # Check input
        labels = ut.check_labels(labels=labels, vals_requiered=[0, 1], len_requiered=len(self.df_parts))
        ut.check_number_range(name="n_filter", val=n_filter, min_val=1, just_int=True)
        ut.check_number_range(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True, just_int=True)
        ut.check_number_range(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100, just_int=True)
        ut.check_number_range(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_cor", val=max_cor, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_bool(name="check_cat", val=check_cat)
        ut.check_bool(name="parametric", val=parametric)
        args_len, _ = check_parts_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.check_number_range(name="n_process", val=n_processes, min_val=1, accept_none=True, just_int=True)
        # Settings and creation of objects
        args = dict(split_kws=self.split_kws, df_scales=self.df_scales)
        n_feat = len(get_features_(**args, list_parts=list(self.df_parts)))
        n_filter = n_feat if n_feat < n_filter else n_filter
        if self._verbose:
            ut.print_out(f"1. CPP creates {n_feat} features for {len(self.df_parts)} samples")
            ut.print_start_progress()
        # Pre-filtering: Select best n % of feature (filter_pct) based std(test set) and mean_dif
        abs_mean_dif, std_test, features = pre_filtering_info(**args,
                                                              df_parts=self.df_parts,
                                                              y=labels,
                                                              accept_gaps=self._accept_gaps,
                                                              verbose=self._verbose,
                                                              n_processes=n_processes)
        n_feat = int(len(features))
        if n_pre_filter is None:
            n_pre_filter = int(n_feat * (pct_pre_filter / 100))
            n_pre_filter = n_filter if n_pre_filter < n_filter else n_pre_filter
        if self._verbose:
            ut.print_finished_progress()
            pct_pre_filter = np.round((n_pre_filter/n_feat*100), 2)
            ut.print_out(f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest '{ut.COL_ABS_MEAN_DIF}'"
                         f" and 'max_std_test' <= {max_std_test}")
        df = pre_filtering(features=features,
                           abs_mean_dif=abs_mean_dif,
                           std_test=std_test,
                           n=n_pre_filter,
                           max_std_test=max_std_test)
        features = df[ut.COL_FEATURE].to_list()
        # Add feature information
        df = add_stat(df_feat=df, df_scales=self.df_scales, df_parts=self.df_parts,
                      labels=labels, parametric=parametric, accept_gaps=self._accept_gaps)
        feat_positions = get_positions_(features=features, start=start, **args_len)
        df[ut.COL_POSITION] = feat_positions
        df = add_scale_info_(df_feat=df, df_cat=self.df_cat)
        # Filtering using CPP algorithm
        if self._verbose:
            ut.print_out(f"3. CPP filtering algorithm")
        df_feat = filtering(df=df, df_scales=self.df_scales,
                            n_filter=n_filter, check_cat=check_cat,
                            max_overlap=max_overlap, max_cor=max_cor)
        # Adjust df_feat
        df_feat.reset_index(drop=True, inplace=True)
        df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
        if self._verbose:
            ut.print_out(f"4. CPP returns df with {len(df_feat)} unique features including general information and statistics")
        return df_feat

    # TODO get evaluation for any dataset for complete
    def eval(self, list_df_feat=None):
        """"""
        pass
