"""
This is a script for ...
"""
import pandas as pd

from aaanalysis.cpp.feature import SequenceFeature
from aaanalysis.cpp._feature_stat import SequenceFeatureStatistics

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool

# I Helper Functions


# Filtering functions
def _filtering_info(df=None, df_scales=None, check_cat=True):
    """Get datasets structures for filtering, two dictionaries with feature to scales category resp.
    feature positions and one datasets frame with paired pearson correlations of all scales"""
    if check_cat:
        dict_c = dict(zip(df[ut.COL_FEATURE], df["category"]))
    else:
        dict_c = dict()
    dict_p = dict(zip(df[ut.COL_FEATURE], [set(x) for x in df["positions"]]))
    df_cor = df_scales.corr()
    return dict_c, dict_p, df_cor


# TODO simplify checks & interface (end-to-end check with tests & docu)

# II Main Functions
class CPP(Tool):
    """
    Create and filter features that are most discriminant between two sets of sequences.

    Parameters
    ----------
    df_scales : :class:`pandas.DataFrame`
        DataFrame with amino acid scales.
    df_cat : :class:`pandas.DataFrame`, default = aa.load_categories
        DataFrame with default categories for physicochemical amino acid scales.
    df_parts : :class:`pandas.DataFrame`
        DataFrame with sequence parts.
    split_kws : dict, default = SequenceFeature.get_split_kws
        Nested dictionary with parameter dictionary for each chosen split_type.
    accept_gaps : bool, default = False
        Whether to accept missing values by enabling omitting for computations (if True).

    verbose : bool, default = True
        Whether to print progress information about the algorithm (if True).

    Notes
    -----
    The CPP.run() method performs all steps of the CPP algorithm.
    """
    def __init__(self, df_scales=None, df_cat=None, df_parts=None, split_kws=None,
                 accept_gaps=False, verbose=True):
        # Load default scales if not specified
        sf = SequenceFeature()
        if df_cat is None:
            df_cat = aa.load_scales(name=ut.STR_SCALE_CAT)
        if df_scales is None:
            df_scales = aa.load_scales()
        if split_kws is None:
            split_kws = sf.get_split_kws()
        ut.check_bool(name="verbose", val=verbose)
        ut.check_df_parts(df_parts=df_parts, verbose=verbose)
        df_parts = ut.check_df_scales(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        df_cat, df_scales = ut.check_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        ut.check_split_kws(split_kws=split_kws)
        self._verbose = verbose
        self._accept_gaps = accept_gaps
        # Feature components: Scales + Part + Split
        self.df_cat = df_cat.copy()
        self.df_scales = df_scales.copy()
        self.df_parts = df_parts.copy()
        self.split_kws = split_kws

    # Adder methods for CPP analysis (used in run method)
    def _add_scale_info(self, df_feat=None):
        """
        Add scale information to DataFrame. Scale information are–from general to specific–scale categories,
        sub categories, and scale names.

        Parameters
        ----------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame to add scale categories.

        Returns
        -------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame including scale categories.
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)

        # Add scale categories
        df_cat = self.df_cat.copy()
        i = df_feat.columns.get_loc(ut.COL_FEATURE)
        for col in [ut.COL_SCALE_DES, ut.COL_SCALE_NAME, ut.COL_SUBCAT, ut.COL_CAT]:
            if col in list(df_feat):
                df_feat.drop(col, inplace=True, axis=1)
            dict_cat = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[col]))
            vals = [dict_cat[s.split("-")[2]] for s in df_feat[ut.COL_FEATURE]]
            df_feat.insert(i + 1, col, vals)
        return df_feat

    def _add_stat(self, df_feat=None, labels=None, parametric=False, accept_gaps=False):
        """
        Add summary statistics for each feature to DataFrame.

        Parameters
        ----------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame to add statistics.
        labels: array-like, shape (n_samples)
            Class labels for samples in df_parts attribute.
        parametric: bool, default = False
            Whether to use parametric (T-test) or non-parametric (U-test) test for p-value computation.
        accept_gaps: bool, default = False
            Whether to accept missing values by enabling omitting for computations (if True).

        Returns
        -------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame including statistics for comparing two given groups.

        Notes
        -----
        P-values are calculated Mann-Whitney U test (non-parametric) or T-test (parametric) as implemented in SciPy.

        For multiple hypothesis correction, the Benjamini-Hochberg FDR correction is applied on all given features
        as implemented in SciPy.
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_labels(labels=labels, df=self.df_parts, name_df="df_parts")
        ut.check_bool(name="parametric", val=parametric)

        # Add feature statistics
        features = list(df_feat[ut.COL_FEATURE])
        sf = SequenceFeature()
        sfs = SequenceFeatureStatistics()
        X = sf.feat_matrix(df_parts=self.df_parts,
                           features=features,
                           df_scales=self.df_scales,
                           accept_gaps=accept_gaps)
        df_feat = sfs.add_stat(df=df_feat, X=X, y=labels, parametric=parametric)
        return df_feat

    @staticmethod
    def _add_positions(df_feat=None, tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=4, start=1):
        """Add sequence positions to DataFrame."""
        # Check input (length checked by SequenceFeaturePositions)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        # Add positions of features
        features = df_feat[ut.COL_FEATURE].to_list()
        sf = SequenceFeature()
        feat_positions = sf.add_position(features=features, tmd_len=tmd_len, start=start,
                                         jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
        df_feat[ut.COL_POSITION] = feat_positions
        return df_feat

    # Filtering methods
    @staticmethod
    def _pre_filtering(features=None, abs_mean_dif=None, std_test=None, max_std_test=0.2, n=10000):
        """CPP pre-filtering based on thresholds."""
        df = pd.DataFrame(zip(features, abs_mean_dif, std_test),
                          columns=[ut.COL_FEATURE, ut.COL_ABS_MEAN_DIF, ut.COL_STD_TEST])
        df = df[df[ut.COL_STD_TEST] <= max_std_test]
        df = df.sort_values(by=ut.COL_ABS_MEAN_DIF, ascending=False).head(n)
        return df

    def _filtering(self, df=None, max_overlap=0.5, max_cor=0.5, n_filter=100, check_cat=True):
        """CPP filtering algorithm based on redundancy reduction in descending order of absolute AUC."""
        dict_c, dict_p, df_cor = _filtering_info(df=df, df_scales=self.df_scales, check_cat=check_cat)
        df = df.sort_values(by=[ut.COL_ABS_AUC, ut.COL_ABS_MEAN_DIF], ascending=False).copy().reset_index(drop=True)
        list_feat = list(df[ut.COL_FEATURE])
        list_top_feat = [list_feat.pop(0)]  # List with best feature
        for feat in list_feat:
            add_flag = True
            # Stop condition for limit
            if len(list_top_feat) == n_filter:
                break
            # Compare features with all top features (added if low overlap & weak correlation or different category)
            for top_feat in list_top_feat:
                if not check_cat or dict_c[feat] == dict_c[top_feat]:
                    # Remove if feat positions high overlap or subset
                    pos, top_pos = dict_p[feat], dict_p[top_feat]
                    overlap = len(top_pos.intersection(pos))/len(top_pos.union(pos))
                    if overlap >= max_overlap or pos.issubset(top_pos):
                        # Remove if high pearson correlation
                        scale, top_scale = feat.split("-")[2], top_feat.split("-")[2]
                        cor = df_cor[top_scale][scale]
                        if cor > max_cor:
                            add_flag = False
            if add_flag:
                list_top_feat.append(feat)
        df_top_feat = df[df[ut.COL_FEATURE].isin(list_top_feat)]
        return df_top_feat

    # Main method
    def run(self, labels=None, parametric=False, n_filter=100,
            tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=4, start=1,
            check_cat=True, n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5, max_cor=0.5,
            n_processes=None):
        """
        Perform CPP pipeline by creation and two-step filtering of features. CPP aims to
        identify a collection of non-redundant features that are most discriminant between
        a test and a reference group of sequences.

        Parameters
        ----------
        labels : array-like, shape (n_samples)
            Class labels for samples in sequence DataFrame (test=1, reference=0).
        parametric : bool, default = False
            Whether to use parametric (T-test) or non-parametric (U-test) test for p-value computation.
        n_filter : int, default = 100
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter : int, optional
            Number of feature to be pre-filtered by CPP algorithm. If None, a percentage of all features is used.
        tmd_len : int, >0
            Length of TMD used for positions. TODO add link to explanation
        start : int, >=0
            Position label of first amino acid position (starting at N-terminus).
        jmd_n_len : int, >=0, default = 10
            Length of JMD-N.
        jmd_c_len : int, >=0, default = 10
            Length of JMD-C.
        ext_len : int, >=0, default = 4
            Length of TMD-extending part (starting from C and N terminal part of TMD).
            Should be longer than jmd_n_len and jmd_c_len
        check_cat : bool, default = True
            Whether to check for redundancy within scale categories.
        pct_pre_filter : int, default = 5
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test : float [0-1], default = 0.2
            Maximum standard deviation within the test group used as threshold for pre-filtering.
        max_overlap : float [0-1], default = 0.5
            Maximum positional overlap of features used as threshold for filtering.
        max_cor : float [0-1], default = 0.5
            Maximum Pearson correlation of features used as threshold for filtering.
        n_processes : int, default = None
            Number of CPUs used for multiprocessing. If None, number will be optimized automatically.

        Returns
        -------
        df_feat : :class:`pandas.DataFrame`, shape (n_feature, n_feature_information)
            DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        The feature DataFrame contains the following eleven columns, including the unique
        feature id (1), scale information (2-4), statistical results used for filtering and
        ranking (5-10), and feature positions (11):

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
        ut.check_labels(labels=labels, df=self.df_parts, name_df="df_parts")
        ut.check_args_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
        ut.check_non_negative_number(name="n_filter", val=n_filter, min_val=1)
        ut.check_non_negative_number(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True)
        ut.check_non_negative_number(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100)
        ut.check_non_negative_number(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_non_negative_number(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_bool(name="verbose", val=self._verbose)
        # Settings and creation of objects
        args = dict(split_kws=self.split_kws, df_scales=self.df_scales)
        if self._verbose:
            sf = SequenceFeature()
            n_feat = len(sf.get_features(**args, list_parts=list(self.df_parts)))
            print(f"1. CPP creates {n_feat} features for {len(self.df_parts)} samples")
            ut.print_start_progress()
        # Pre-filtering: Select best n % of feature (filter_pct) based std(test set) and mean_dif
        sfs = SequenceFeatureStatistics()
        abs_mean_dif, std_test, features = sfs.pre_filtering_info(**args,
                                                                  df_parts=self.df_parts,
                                                                  y=labels,
                                                                  accept_gaps=self._accept_gaps,
                                                                  verbose=self._verbose,
                                                                  n_processes=n_processes)
        if n_pre_filter is None:
            n_pre_filter = int(len(features) * (pct_pre_filter / 100))
        if self._verbose:
            ut.print_finished_progress()
            print(f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest '{ut.COL_ABS_MEAN_DIF}'"
                  f" and 'max_std_test' <= {max_std_test}")
        df = self._pre_filtering(features=features,
                                 abs_mean_dif=abs_mean_dif,
                                 std_test=std_test,
                                 n=n_pre_filter,
                                 max_std_test=max_std_test)
        # Filtering using CPP algorithm
        df = self._add_stat(df_feat=df, labels=labels, parametric=parametric, accept_gaps=self._accept_gaps)
        if self._verbose:
            print(f"3. CPP filtering algorithm")
        df = self._add_positions(df_feat=df, tmd_len=tmd_len, start=start)
        df = self._add_scale_info(df_feat=df)
        df_feat = self._filtering(df=df, n_filter=n_filter, check_cat=check_cat, max_overlap=max_overlap, max_cor=max_cor)
        df_feat.reset_index(drop=True, inplace=True)
        if self._verbose:
            print(f"4. CPP returns df with {len(df_feat)} unique features including general information and statistics")
        return df_feat

    @staticmethod
    def eval(df_feat=None, features=None):
        """Get evaluation for provided dataset"""
        # TODO get evaluation for any dataset for compelete
