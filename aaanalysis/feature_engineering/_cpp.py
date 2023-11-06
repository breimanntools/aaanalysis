"""
This is a script for the frontend of the CPP class, a sequence-based feature engineering object.
"""
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple, Type

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.template_classes import Tool

# Import supportive class (exception for importing from same sub-package)
from ._sequence_feature import SequenceFeature
from ._backend.cpp._utils_cpp import get_feat_matrix
from ._backend.cpp.cpp_run import pre_filtering_info, pre_filtering, filtering, add_stat
from ._backend.cpp.cpp_methods import (get_positions,
                                       get_dif)


# I Helper Functions
# Check for add methods
def check_ref_group(ref_group=0, labels=None):
    """Check if ref group class lable"""
    if ref_group not in labels:
        raise ValueError(f"'ref_group' ({ref_group}) not class label: {set(labels)}.")


def check_sample_in_df_seq(sample_name=None, df_seq=None):
    """Check if sample name in df_seq"""
    list_names = list(df_seq[ut.COL_NAME])
    if sample_name not in list_names:
        error = f"'sample_name' ('{sample_name}') not in '{ut.COL_NAME}' of 'df_seq'." \
                f"\nValid names are: {list_names}"
        raise ValueError(error)

# Adder methods for CPP analysis (used in run method)
def _add_stat(df_feat=None, df_parts=None, df_scales=None, labels=None, parametric=False, accept_gaps=False):
        """
        Add summary statistics for each feature to DataFrame.

        Notes
        -----
        P-values are calculated Mann-Whitney U test (non-parametric) or T-test (parametric) as implemented in SciPy.
        For multiple hypothesis correction, the Benjamini-Hochberg FDR correction is applied on all given features
        as implemented in SciPy.
        """
        # Add feature statistics
        features = list(df_feat[ut.COL_FEATURE])
        X = get_feat_matrix(features=features,
                            df_parts=df_parts,
                            df_scales=df_scales,
                            accept_gaps=accept_gaps)
        df_feat = add_stat(df=df_feat, X=X, y=labels, parametric=parametric)
        return df_feat

def _add_scale_info(df_feat=None, df_cat=None):
    """Add scale information to DataFrame (scale categories, sub categories, and scale names)."""
    # Add scale categories
    df_cat = df_cat.copy()
    i = df_feat.columns.get_loc(ut.COL_FEATURE)
    for col in [ut.COL_SCALE_DES, ut.COL_SCALE_NAME, ut.COL_SUBCAT, ut.COL_CAT]:
        if col in list(df_feat):
            df_feat.drop(col, inplace=True, axis=1)
        dict_cat = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[col]))
        vals = [dict_cat[s.split("-")[2]] for s in df_feat[ut.COL_FEATURE]]
        df_feat.insert(i + 1, col, vals)
    return df_feat


# Common interface
doc_param_len_pos = \
"""\
start
    Position label of first amino acid position (starting at N-terminus, >=0).
tmd_len
    Length of TMD (>0).
jmd_n_len
    Length of JMD-N (>=0).
jmd_c_len
    Length of JMD-C (>=0).
ext_len
    Length of TMD-extending part (starting from C and N terminal part of TMD, >=0).\
"""


# TODO simplify checks & interface (end-to-end check with tests & docu)
# TODO  TODO add link to explanation for TMD, JMDs
# II Main Functions
class CPP(Tool):
    """
    Create and filter features that are most discriminant between two sets of sequences.

    Parameters
    ----------
    df_parts
        DataFrame with sequence parts.
    split_kws
        Nested dictionary with parameter dictionary for each chosen split_type.
        Default from :meth:`SequenceFeature.get_split_kws`
    df_scales
        DataFrame with amino acid scales. Default from :meth:`load_scales` with 'name'='scales_cat'.
    df_cat
        DataFrame with default categories for physicochemical amino acid scales.
        Default from :meth:`load_categories`
    accept_gaps
        Whether to accept missing values by enabling omitting for computations (if ``True``).
    verbose
        If ``True``, verbose outputs are enabled. Global 'verbose' setting is used if ``None``.

    Attributes
    ----------
    df_parts
        DataFrame with sequence parts.
    split_kws
        Nested dictionary with parameter dictionary for each chosen split_type.
    df_scales
        DataFrame with amino acid scales.
    df_cat
        DataFrame with default categories for physicochemical amino acid scales.
    """
    def __init__(self,
                 df_parts: pd.DataFrame = None,
                 split_kws : Optional[dict] = None,
                 df_scales : Optional[pd.DataFrame] = None,
                 df_cat : Optional[pd.DataFrame] = None,
                 accept_gaps : bool = False,
                 verbose: Optional[bool] = None):
        # Load default scales if not specified
        if split_kws is None:
            sf = SequenceFeature()
            split_kws = sf.get_split_kws()
        if df_scales is None:
            df_scales = aa.load_scales()
        if df_cat is None:
            df_cat = aa.load_scales(name=ut.STR_SCALE_CAT)
        # Check input
        verbose = ut.check_verbose(verbose)
        ut.check_df_parts(df_parts=df_parts, verbose=verbose)
        df_parts = ut.check_df_scales(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        df_cat, df_scales = ut.check_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        ut.check_split_kws(split_kws=split_kws)
        # Internal attributes
        self._verbose = verbose
        self._accept_gaps = accept_gaps
        # Feature components: Scales + Part + Split
        self.df_cat = df_cat.copy()
        self.df_scales = df_scales.copy()
        self.df_parts = df_parts.copy()
        self.split_kws = split_kws

    # Main method
    @ut.doc_params(doc_param_len_pos=doc_param_len_pos)
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
            ext_len: int = 4,
            n_processes : Optional[int] = None
            ) -> pd.DataFrame:
        """
        Perform CPP pipeline by creation and two-step filtering of features. CPP aims to
        identify a collection of non-redundant features that are most discriminant between
        a test and a reference group of sequences.

        Parameters
        ----------
        labels : `array-like, shape (n_samples, )`
            Class labels for samples in sequence DataFrame (test=1, reference=0).
        n_filter
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter
            Number of feature to be pre-filtered by CPP algorithm. If None, a percentage of all features is used.
        pct_pre_filter
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test
            Maximum standard deviation [0-1] within the test group used as threshold for pre-filtering.
        max_overlap
            Maximum positional overlap [0-1] of features used as threshold for filtering.
        max_cor
            Maximum Pearson correlation [0-1] of features used as threshold for filtering.
        check_cat
            Whether to check for redundancy within scale categories.
        parametric
            Whether to use parametric (T-test) or non-parametric (Mann-Whitney-U-test) test for p-value computation.
        {doc_param_len_pos}
        n_processes
            Number of CPUs used for multiprocessing. If None, number will be optimized automatically.

        Returns
        -------
        df_feat
            DataFrame with a unique identifier, scale information, statistics, and positions for each feature.

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
        ut.check_labels_(labels=labels, df=self.df_parts, name_df="df_parts")
        ut.check_args_len(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
        ut.check_number_range(name="n_filter", val=n_filter, min_val=1, just_int=True)
        ut.check_number_range(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True, just_int=True)
        ut.check_number_range(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100, just_int=True)
        ut.check_number_range(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_number_range(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        # Settings and creation of objects
        args = dict(split_kws=self.split_kws, df_scales=self.df_scales)
        if self._verbose:
            sf = SequenceFeature()
            n_feat = len(sf.get_features(**args, list_parts=list(self.df_parts)))
            ut.print_out(f"1. CPP creates {n_feat} features for {len(self.df_parts)} samples")
            ut.print_start_progress()
        # Pre-filtering: Select best n % of feature (filter_pct) based std(test set) and mean_dif
        abs_mean_dif, std_test, features = pre_filtering_info(**args,
                                                              df_parts=self.df_parts,
                                                              y=labels,
                                                              accept_gaps=self._accept_gaps,
                                                              verbose=self._verbose,
                                                              n_processes=n_processes)
        if n_pre_filter is None:
            n_pre_filter = int(len(features) * (pct_pre_filter / 100))
            n_pre_filter = n_filter if n_pre_filter < n_filter else n_pre_filter
        if self._verbose:
            ut.print_finished_progress()
            ut.print_out(f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest '{ut.COL_ABS_MEAN_DIF}'"
                         f" and 'max_std_test' <= {max_std_test}")
        df = pre_filtering(features=features,
                           abs_mean_dif=abs_mean_dif,
                           std_test=std_test,
                           n=n_pre_filter,
                           max_std_test=max_std_test)
        # Add feature information
        df = _add_stat(df_feat=df, df_scales=self.df_scales, df_parts=self.df_parts,
                       labels=labels, parametric=parametric, accept_gaps=self._accept_gaps)
        df = self.add_positions(df_feat=df, start=start,
                                tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
        df = _add_scale_info(df_feat=df, df_cat=self.df_cat)
        # Filtering using CPP algorithm
        if self._verbose:
            ut.print_out(f"3. CPP filtering algorithm")
        df_feat = filtering(df=df, df_scales=self.df_scales,
                            n_filter=n_filter, check_cat=check_cat,
                            max_overlap=max_overlap, max_cor=max_cor)
        df_feat.reset_index(drop=True, inplace=True)
        if self._verbose:
            ut.print_out(f"4. CPP returns df with {len(df_feat)} unique features including general information and statistics")
        return df_feat

    # Feature information methods (can be included to df_feat for individual sequences)
    # TODO add sequence positions
    @staticmethod
    @ut.doc_params(doc_param_len_pos=doc_param_len_pos)
    def add_positions(df_feat: pd.DataFrame = None,
                      start: int = 1,
                      tmd_len: int = 20,
                      jmd_n_len: int = 10,
                      jmd_c_len: int = 10,
                      ext_len: int = 4
                      ) -> pd.DataFrame:
        """Create list with positions for given feature names

        Parameters
        ----------
        df_feat
            Feature DataFrame, output of CPP.run(), to add sample difference.
        {doc_param_len_pos}

        Returns
        -------
        df_feat
            Feature DataFrame with positions for each feature in feat_names

        Notes
        -----
        - The sum of length parameters define the total number of positions (``jmd_n_len`` + ``tmd_len`` + ``jmd_c_len``).
        - ``ext_len`` < ``jmd_m_len`` and ``ext_len`` < ``jmd_c_len``
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        features = list(df_feat["feature"])
        features = ut.check_features(features=features)
        ut.check_number_range(name="tmd_len", val=tmd_len, just_int=True, min_val=1)
        args = dict(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len, start=start)
        for name in args:
            ut.check_number_range(name=name, val=args[name], just_int=True, min_val=0)
        # Get feature position
        feat_positions = get_positions(features=features, start=start,
                                        tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                        ext_len=ext_len)
        df_feat[ut.COL_POSITION] = feat_positions
        return df_feat

    def add_dif(self,
                df_feat: pd.DataFrame = None,
                labels: ut.ArrayLike1D = None,
                list_names: List[str] = None,
                sample_name: str = None,
                ref_group: int = 0
                ) -> pd.DataFrame:
        """
        Add feature value difference between sample and reference group to DataFrame.

        Parameters
        ----------
        df_feat
            Feature DataFrame (CPP output) to add sample difference.
        labels: `array-like, shape (n_samples, )`
            Class labels for samples in sequence DataFrame.
        ref_group
            Class label of reference group.
        list_names
            List of names matching to `df_parts`.
        sample_name
            Name of sample for which the feature value difference to a given reference group should be computed.

        Returns
        -------
        df_feat
            Feature DataFrame with feature value difference.
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        features = list(df_feat["feature"])
        features = ut.check_features(features=features)
        check_ref_group(ref_group=ref_group, labels=labels)
        # Add sample difference to reference group
        feat_dif = get_dif(features=features,
                           df_parts=self.df_parts,
                           df_scales=self.df_scales,
                           accept_gaps=self._accept_gaps,
                           list_names=list_names,
                           sample_name=sample_name,
                           labels=labels,
                           ref_group=ref_group)
        df_feat[f"dif_{sample_name}"] = feat_dif
        return df_feat


    # TODO get evaluation for any dataset for compelete
    def eval(self, df_feat=None, features=None):
        pass
