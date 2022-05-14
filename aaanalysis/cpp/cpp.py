"""
This is a script for ...
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

import aaanalysis.cpp._utils as ut
from aaanalysis.cpp.feature import SequenceFeature
from aaanalysis.cpp._feature_stat import SequenceFeatureStatistics
from aaanalysis.cpp._feature_pos import SequenceFeaturePositions
from aaanalysis.cpp._cpp import CPPPlots, cpp_statistics


# I Helper Functions
# TODO separate interface from backend
# TODO simplify interface (delete old profile)
# TODO delete SHAP
# TODO add importance plot for heatmap
# TODO add ranking

# Check CPP parameters
def check_len_ext_and_jmd(jmd_n_len=None, jmd_c_len=None, ext_len=None):
    """Check if lengths are matching"""
    ut.check_non_negative_number(name="jmd_n_len", val=jmd_n_len)
    ut.check_non_negative_number(name="jmd_c_len", val=jmd_c_len)
    ut.check_non_negative_number(name="ext_len", val=ext_len)
    if ext_len > jmd_n_len:
        raise ValueError(f"'ext_len' ({ext_len}) must be <= jmd_n_len ({jmd_n_len})")
    if ext_len > jmd_c_len:
        raise ValueError(f"'ext_len' ({ext_len}) must be <= jmd_c_len ({jmd_c_len})")


# Check for add methods
def check_shap_value_for_feat_impact(df_feat=None, col_shap=None):
    """Check if SHAP value column in df"""
    if col_shap not in df_feat:
        raise ValueError(f"'{col_shap}' must be column in 'df_feat' to compute feature impact")
    wrong_types = [x for x in list(df_feat[col_shap]) if type(x) not in [float, int]]
    if len(wrong_types) > 0:
        error = f"Values in '{col_shap}' should be type float or int\n" \
                f" but following values do not match: {wrong_types}"
        raise ValueError(error)


def check_feat_impact_in_df_feat(df_feat=None, name_feat_impact=None):
    """Check if name for feature impact column already"""
    if name_feat_impact in df_feat:
        error = f"'name_feat_impact' ('{name_feat_impact}') already in 'df_feat' columns: {list(df_feat)}"
        raise ValueError(error)


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


# Check get df positions
def check_value_type(val_type=None, count_in=True):
    """Check if value type is valid"""
    list_value_type = ["mean", "sum", "std"]
    if count_in:
        list_value_type.append("count")
    if val_type not in list_value_type:
        raise ValueError(f"'val_type' ('{val_type}') should be on of following: {list_value_type}")


def check_normalize(normalize=True):
    """Check normalize parameter"""
    if not (type(normalize) == bool or normalize == "positions"):
        raise ValueError(f"'normalize' ('{normalize}') should be bool or, if normalized for positions, 'positions'.")
    normalize_for_positions = False
    if normalize == "positions":
        normalize = True
        normalize_for_positions = True
    return normalize, normalize_for_positions


# Check for plotting methods
def check_args_len(tmd_seq=None, jmd_n_seq=None, jmd_c_seq=None, tmd_len=None, jmd_n_len=None, jmd_c_len=None):
    """Check if parameters for sequence size and sequences match"""
    count = 0
    for seq in [tmd_seq, jmd_c_seq, jmd_n_seq]:
        if type(seq) == str:
            count += 1
    if count == 3:
        if len(jmd_n_seq) != jmd_n_len:
            error = f"'jmd_n_seq' ('{jmd_n_seq}', len={len(jmd_n_seq)}) does not match CPP setting: ({jmd_n_len})."
            raise ValueError(error)
        if len(jmd_c_seq) != jmd_c_len:
            error = f"'jmd_c_seq' ('{jmd_c_seq}', len={len(jmd_c_seq)}) does not match CPP setting: ({jmd_c_len})."
            raise ValueError(error)
        tmd_len, jmd_n_len, jmd_c_len = len(tmd_seq), len(jmd_n_seq), len(jmd_c_seq)
    elif count != 0:
        raise ValueError("'jmd_n_seq' 'tmd_seq', and 'jmd_c_seq' must all be None or sequence (type string)")
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    return args_len


def check_args_size(seq_size=None, tmd_fontsize=None, jmd_fontsize=None):
    """Check if sequence size parameters match"""
    ut.check_non_negative_number(name="seq_size", val=seq_size, min_val=1, accept_none=True, just_int=False)
    ut.check_non_negative_number(name="tmd_fontsize", val=tmd_fontsize, min_val=1, accept_none=True, just_int=False)
    ut.check_non_negative_number(name="jmd_fontsize", val=jmd_fontsize, min_val=1, accept_none=True, just_int=False)
    args_size = dict(seq_size=seq_size, tmd_fontsize=tmd_fontsize, jmd_fontsize=jmd_fontsize)
    return args_size


def check_args_xtick(xtick_size=None, xtick_width=None, xtick_length=None):
    """Check if x tick parameters non-negative float"""
    args = dict(accept_none=True, just_int=False, min_val=0)
    ut.check_non_negative_number(name="xtick_size", val=xtick_size, **args)
    ut.check_non_negative_number(name="xtick_width", val=xtick_width, **args)
    ut.check_non_negative_number(name="xtick_length", val=xtick_length, **args)
    args_xtick = dict(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
    return args_xtick


def check_args_ytick(ytick_size=None, ytick_width=None, ytick_length=None):
    """Check if y tick parameters non-negative float"""
    args = dict(accept_none=True, just_int=False, min_val=1)
    ut.check_non_negative_number(name="ytick_size", val=ytick_size, **args)
    ut.check_non_negative_number(name="ytick_width", val=ytick_width, **args)
    ut.check_non_negative_number(name="ytick_length", val=ytick_length, **args)
    args_ytick = dict(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
    return args_ytick


def check_part_color(tmd_color=None, jmd_color=None):
    """Check if part colors valid"""
    ut.check_color(name="tmd_color", val=tmd_color)
    ut.check_color(name="jmd_color", val=jmd_color)
    args_part_color = dict(tmd_color=tmd_color, jmd_color=jmd_color)
    return args_part_color


def check_seq_color(tmd_seq_color=None, jmd_seq_color=None):
    """Check sequence colors"""
    ut.check_color(name="tmd_seq_color", val=tmd_seq_color)
    ut.check_color(name="jmd_seq_color", val=jmd_seq_color)
    args_seq_color = dict(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)
    return args_seq_color


def check_figsize(figsize=None):
    """"""
    ut.check_tuple(name="figsize", val=figsize, n=2)
    ut.check_non_negative_number(name="figsize:width", val=figsize[0], min_val=1, just_int=False)
    ut.check_non_negative_number(name="figsize:height", val=figsize[1], min_val=1, just_int=False)


def check_dict_color(dict_color=None, df_cat=None):
    """Check if color dictionary is matching to DataFrame with categories"""
    list_cats = list(sorted(set(df_cat[ut.COL_CAT])))
    if dict_color is None:
        dict_color = ut.DICT_COLOR
    if not isinstance(dict_color, dict):
        raise ValueError(f"'dict_color' should be a dictionary with colors for: {list_cats}")
    list_cat_not_in_dict_cat = [x for x in list_cats if x not in dict_color]
    if len(list_cat_not_in_dict_cat) > 0:
        error = f"'dict_color' not complete! Following categories are missing from 'df_cat': {list_cat_not_in_dict_cat}"
        raise ValueError(error)
    for key in dict_color:
        color = dict_color[key]
        ut.check_color(name=key, val=color)
    return dict_color


def check_parameters(func=None, name_called_func=None, e=None):
    """Check parameters string from error message of third party packages"""
    list_arg_str = ["property ", "attribute ", "argument ", "parameter "]
    str_error = ""
    for arg_str in list_arg_str:
        if arg_str in str(e):
            error_arg = str(e).split(arg_str)[1]
            str_error += "Error due to {} parameter. ".format(error_arg)
            break
    args = [x for x in inspect.getfullargspec(func).args if x != "self"]
    str_error += "Arguments are allowed from {} and as follows: {}".format(name_called_func, args)
    return str_error


# Check heatmap plotting
def check_vmin_vmax(vmin=None, vmax=None):
    """Check if number of cmap colors is valid with given value range"""
    ut.check_float(name="vmin", val=vmin, accept_none=True, just_float=False)
    ut.check_float(name="vmax", val=vmax, accept_none=True, just_float=False)
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(f"'vmin' ({vmin}) < 'vmax' ({vmax}) not fulfilled.")


# Check barplot and profile
def check_grid_axis(grid_axis=None):
    """"""
    list_valid = ["x", 'y', 'both']
    if grid_axis not in list_valid:
        raise ValueError(f"'grid_axis' ('{grid_axis}') not valid. Chose from following: {list_valid}")


# Check stat plot
def check_ylabel_fontweight(ylabel_fontweight=None, accept_none=True):
    """"""
    if accept_none and ylabel_fontweight is None:
        return
    name = "ylabel_fontweight"
    args = dict(name=name, val=ylabel_fontweight)
    list_weights = ['light', 'medium', 'bold']
    if type(ylabel_fontweight) in [float, int]:
        ut.check_non_negative_number(**args, min_val=0, max_val=1000, just_int=False)
    elif isinstance(ylabel_fontweight, str):
        if ylabel_fontweight not in list_weights:
            error = f"'{name}' ({ylabel_fontweight}) should be one of following: {list_weights}"
            raise ValueError(error)
    else:
        error = f"'{name}' ({ylabel_fontweight}) should be either numeric value in range 0-1000" \
                f"\n\tor one of following: {list_weights}"
        raise ValueError(error)


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


# Plotting functions
def _get_df_pos(df_feat=None, df_cat=None, y="subcategory", val_col="mean_dif",
                value_type="mean", normalize=False,
                tmd_len=20, jmd_n_len=10, jmd_c_len=10, start=1):
    """Helper method for plotting"""
    normalize, normalize_for_pos = check_normalize(normalize=normalize)
    cpp_plot = CPPPlots(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, start=start)
    df_pos = cpp_plot.get_df_pos(df=df_feat.copy(), y=y, value_type=value_type, val_col=val_col,
                                 normalize=normalize,
                                 normalize_for_pos=normalize_for_pos)
    # Sort according to given categories
    list_cat = list(df_cat[y].drop_duplicates())
    list_col = list(df_pos.T)
    sorted_col = [x for x in list_cat if x in list_col]
    df_pos = df_pos.T[sorted_col].T
    return df_pos


def _set_size_to_optimized_value(seq_size=None, tmd_fontsize=None, jmd_fontsize=None, opt_size=None):
    """Set sizes to given value if None"""
    if seq_size is None:
        seq_size = opt_size
    if tmd_fontsize is None:
        tmd_fontsize = opt_size
    if jmd_fontsize is None:
        jmd_fontsize = opt_size
    args_size = dict(seq_size=seq_size, tmd_fontsize=tmd_fontsize, jmd_fontsize=jmd_fontsize)
    return args_size

# TODO simplify checks & interface (end-to-end check with tests & docu)
#   TODO plot_functions test & refactor (end-to-end)


# II Main Functions
class CPP:
    """Create and filter features that are most discriminant between two sets of sequences.

    Attributes
    ----------
    df_scales: pd.DataFrame, default=SequenceFeature.load_scales
        DataFrame with default amino acid scales.
    df_cat: pd.DataFrame, default=SequenceFeature.load_categories
        DataFrame with default categories for physicochemical amino acid scales
    df_parts: pd.DataFrame
        DataFrame with sequence parts.
    split_kws: dict, default=SequenceFeature.get_split_kws
        Nested dictionary with parameter dictionary for each chosen split_type.
    accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if True).
    jmd_n_len: int, >=0, default=10
        Length of JMD-N.
    jmd_c_len: int, >=0, default=10
        Length of JMD-C.
    ext_len:int, >=0, default=4
        Length of TMD-extending part (starting from C and N terminal part of TMD).
        Conditions: ext_len<jmd_m_len and ext_len<jmd_c_len
    verbose: bool, default=True
            Whether to print progress information about the algorithm (if True).

    Notes
    -----
    The CPP.run() method performs all steps of the CPP algorithm:

    1. Creation of all possible features

    2. Two-step filtering:

        2.1 Threshold-based filtering

        2.2 Redundancy reduction

    """
    def __init__(self, df_scales=None, df_cat=None, df_parts=None, split_kws=None,
                 accept_gaps=False, jmd_n_len=10, jmd_c_len=10, ext_len=4, verbose=True):
        # Load default scales if not specified
        sf = SequenceFeature()
        if df_cat is None:
            df_cat = sf.load_categories(complete=True)
        if df_scales is None:
            df_scales = sf.load_scales(complete=True)
        if split_kws is None:
            split_kws = sf.get_split_kws()
        ut.check_bool(name="verbose", val=verbose)
        ut.check_df_parts(df_parts=df_parts, verbose=verbose)
        ut.check_df_scales(df_scales=df_scales, df_parts=df_parts, accept_gaps=accept_gaps)
        df_cat, df_scales = ut.check_df_cat(df_cat=df_cat, df_scales=df_scales, verbose=verbose)
        ut.check_split_kws(split_kws=split_kws)
        check_len_ext_and_jmd(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len)
        self.verbose = verbose
        self.accept_gaps = accept_gaps
        # Feature components: Scales + Part + Split
        self.df_cat = df_cat
        self.df_scales = df_scales
        self.df_parts = df_parts
        self.split_kws = split_kws
        # Default color dictionary
        self.dict_color = ut.DICT_COLOR
        # Set consistent length of JMD_N, JMD_C, TMD flanking amino acids (TMD-E)
        self.jmd_n_len = jmd_n_len
        self.jmd_c_len = jmd_c_len
        self.ext_len = ext_len



    # Adder methods for CPP analysis (used in run method)
    def add_scale_info(self, df_feat=None):
        """Add scale information to DataFrame. Scale information are–from general to specific–scale categories,
        sub categories, and scale names.

        Parameters
        ----------
        df_feat: pd.DataFrame
            Feature DataFrame to add scale categories.

        Returns
        -------
        df_feat: pd.DataFrame
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

    def add_stat(self, df_feat=None, labels=None, parametric=False, accept_gaps=False):
        """Add summary statistics for each feature to DataFrame.

        Parameters
        ----------
        df_feat: pd.DataFrame
            Feature DataFrame to add statistics.
        labels: array-like, shape (n_samples)
            Class labels for samples in df_parts attribute.
        parametric: bool, default=False
            Whether to use parametric (T-test) or non-parametric (U-test) test for p-value computation.
        accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if True).

        Returns
        -------
        df_feat: pd.DataFrame
            Feature DataFrame including statistics for comparing two given groups.

        Notes
        -----
        # TODO add link
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

    def add_positions(self, df_feat=None, tmd_len=20, start=1):
        """Add sequence positions to DataFrame.

        Parameters
        ----------
        df_feat: pd.DataFrame
            Feature DataFrame to add feature positions.
        tmd_len: int, >0
            Length of TMD.
        start: int, >=0
            Position label of first amino acid position (starting at N-terminus).

        Returns
        -------
        df_feat: pd.DataFrame
            Feature DataFrame including feature positions.

        Notes
        -----
        The length parameters define the total number of positions (jmd_n_len + tmd_len + jmd_c_len).
        """
        # Check input (length checked by SequenceFeaturePositions)
        df_feat = ut.check_df_feat(df_feat=df_feat)

        # Add positions of features
        sfp = SequenceFeaturePositions()
        dict_part_pos = sfp.get_dict_part_pos(tmd_len=tmd_len,
                                              jmd_n_len=self.jmd_n_len, jmd_c_len=self.jmd_c_len,
                                              ext_len=self.ext_len, start=start)
        df_feat["positions"] = sfp.get_positions(dict_part_pos=dict_part_pos, features=list(df_feat[ut.COL_FEATURE]))
        return df_feat

    @staticmethod
    def add_shap(df_feat=None, col_shap="shap_value", name_feat_impact="feat_impact"):
        """Convert SHAP values in feature impact/importance and add to DataFrame.

        Parameters
        ----------
        df_feat: pd.DataFrame
            Feature DataFrame to add feature impact.
        col_shap: str, default='shap_value'
            Column name of (`SHAP <https://shap.readthedocs.io/en/latest/index.html>`__) values in feature DataFrame
        name_feat_impact: str, default='feat_impact'
            Column name of feature impact or feature importance that should be added to feature DataFrame.

        Returns
        -------
        df_feat: pd.DataFrame
            Feature DataFrame including feature impact.

        Notes
        -----
        - SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of
        any machine learning model.
        - SHAP values represent a feature´s responsibility for a change in the model output.
        - Missing values are accepted in SHAP values.

        See also
        --------
        (`SHAP <https://shap.readthedocs.io/en/latest/index.html>`__)

        """
        # Check input
        df_feat = df_feat.copy()
        ut.check_str(name="name_feat_impact", val=name_feat_impact)
        ut.check_str(name="col_shap", val=col_shap)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_shap_value_for_feat_impact(df_feat=df_feat, col_shap=col_shap)
        check_feat_impact_in_df_feat(df_feat=df_feat, name_feat_impact=name_feat_impact)

        # Compute feature impact (accepting missing values)
        shap_values = np.array(df_feat[col_shap])
        feat_impact = shap_values / np.nansum(np.abs(shap_values)) * 100
        shap_loc = df_feat.columns.get_loc(col_shap)
        df_feat.insert(shap_loc + 1, name_feat_impact, feat_impact)
        return df_feat

    def add_sample_dif(self, df_feat=None, df_seq=None, labels=None, sample_name=str, ref_group=0, accept_gaps=False):
        """Add feature value difference between sample and reference group to DataFrame.

        Parameters
        ----------
        df_feat: pd.DataFrame
            Feature DataFrame to add sample difference.
        df_seq: pd.DataFrame
            DataFrame with sequences and sample names, in which the given sample name is included.
        labels: array-like, shape (n_samples)
            Class labels for samples in sequence DataFrame.
        sample_name: str
            Name of sample for which the feature value difference to a given reference group should be computed.
        ref_group: int, default=0
            Class label of reference group.
        accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if True).

        Returns
        -------
        df_feat: pd.DataFrame
            Feature DataFrame including feature value difference.
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_df_seq(df_seq=df_seq)
        ut.check_labels(labels=labels, df=df_seq, name_df="df_seq")
        check_ref_group(ref_group=ref_group, labels=labels)
        check_sample_in_df_seq(sample_name=sample_name, df_seq=df_seq)
        # Add sample difference to reference group
        sf = SequenceFeature()
        X = sf.feat_matrix(features=list(df_feat["feature"]),
                           df_parts=self.df_parts,
                           df_scales=self.df_scales,
                           accept_gaps=accept_gaps)
        mask = [True if x == ref_group else False for x in labels]
        i = list(df_seq[ut.COL_NAME]).index(sample_name)
        df_feat[f"dif_{sample_name}"] = X[i] - X[mask].mean()
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

    def _filtering(self, df=None, max_overlap=0.49, max_cor=0.5, n_filter=100, check_cat=True):
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
                    if overlap > max_overlap or pos.issubset(top_pos):
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
    def run(self, labels=None, parametric=False, n_filter=100, tmd_len=20, start=1, check_cat=True,
            n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.49, max_cor=0.5):
        """Perform CPP pipeline by creation and two-step filtering of features. CPP aims to identify a collection of
        non-redundant features that are most discriminant between a test and a reference group of sequences.

        Parameters
        ----------
        labels: array-like, shape (n_samples)
            Class labels for samples in sequence DataFrame (test=1, reference=0).
        parametric: bool, default=False
            Whether to use parametric (T-test) or non-parametric (U-test) test for p-value computation.
        n_filter: int, default=100
            Number of features to be filtered/selected by CPP algorithm.
        n_pre_filter: int, optional
            Number of feature to be pre-filtered by CPP algorithm. If None, a percentage of all features is used.

        tmd_len : int, >0
            Length of TMD used for positions.
        start : int, >=0
            Position label of first amino acid position (starting at N-terminus).

        pct_pre_filter: int, default=5
            Percentage of all features that should remain after the pre-filtering step.
        max_std_test: float [0-1], default=0.2
            The maximum standard deviation within the test group used as threshold for pre-filtering.
        max_overlap: float [0-1], default=0.49
            The maximum positional overlap of features used as threshold for filtering.
        max_cor: float [0-1], default=0.5
            The maximum pearson correlation of features used as threshold for filtering.

        accept_gaps: bool, default=False
            Whether to accept missing values by enabling omitting for computations (if True).


        Returns
        -------
        df_feat: pd.DataFrame, shape (n_feature, n_feature_information)
            DataFrame with an unique identifier, scale information, statistics, and positions for each feature.

        Notes
        -----
        The feature DataFrame contains following eleven columns including the unique feature id (1), scale
        information (2-4), statistical results used for filtering and ranking (5-10), and feature positions (11):

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
        ut.check_non_negative_number(name="n_filter", val=n_filter, min_val=1)
        ut.check_non_negative_number(name="n_pre_filter", val=n_pre_filter, min_val=1, accept_none=True)
        ut.check_non_negative_number(name="pct_pre_filter", val=pct_pre_filter, min_val=5, max_val=100)
        ut.check_non_negative_number(name="max_std_test", val=max_std_test, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_non_negative_number(name="max_overlap", val=max_overlap, min_val=0.0, max_val=1.0, just_int=False)
        ut.check_bool(name="verbose", val=self.verbose)
        # Settings and creation of objects
        args = dict(split_kws=self.split_kws, df_scales=self.df_scales)
        if self.verbose:
            sf = SequenceFeature()
            n_feat = len(sf.features(**args, list_parts=list(self.df_parts)))
            print(f"1. CPP creates {n_feat} features for {len(self.df_parts)} samples")
            ut.print_start_progress()
        # Pre-filtering: Select best n % of feature (filter_pct) based on ranking value
        sfs = SequenceFeatureStatistics()
        abs_mean_dif, std_test, features = sfs.pre_filtering_info(**args,
                                                                  df_parts=self.df_parts,
                                                                  y=labels,
                                                                  accept_gaps=self.accept_gaps,
                                                                  verbose=self.verbose)
        if n_pre_filter is None:
            n_pre_filter = int(len(features) * (pct_pre_filter / 100))
        if self.verbose:
            ut.print_finished_progress()
            print(f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest '{ut.COL_ABS_MEAN_DIF}'"
                  f" and 'max_std_test' <= {max_std_test}")
        df = self._pre_filtering(features=features,
                                 abs_mean_dif=abs_mean_dif,
                                 std_test=std_test,
                                 n=n_pre_filter,
                                 max_std_test=max_std_test)
        # Filtering using CPP algorithm
        df = self.add_stat(df_feat=df, labels=labels, parametric=parametric, accept_gaps=self.accept_gaps)
        if self.verbose:
            print(f"3. CPP filtering algorithm")
        df = self.add_positions(df_feat=df, tmd_len=tmd_len, start=start)
        df = self.add_scale_info(df_feat=df)
        df = self._filtering(df=df, n_filter=n_filter, check_cat=check_cat, max_overlap=max_overlap, max_cor=max_cor)
        df.reset_index(drop=True, inplace=True)
        if self.verbose:
            print(f"4. CPP returns df with {len(df)} unique features including general information and statistics")
        return df

    # Plotting methods
    def plot_heatmap(self, df_feat=None, y="subcategory", val_col="mean_dif", val_type="mean", normalize=False,
                     figsize=(10, 7), title=None, title_kws=None,
                     vmin=None, vmax=None, grid_on=True,
                     cmap="RdBu_r", cmap_n_colors=None, dict_color=None, cbar_kws=None,
                     add_jmd_tmd=True, tmd_len=20, start=1,
                     jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                     tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None, tmd_fontsize=None, jmd_fontsize=None,
                     xticks_top=False, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0, ytick_size=None,
                     add_legend_cat=True, legend_kws=None, legend_y_adjust=-0.05,
                     **kwargs):
        """Plot heatmap of selected value column for scale information (y-axis) against sequence position (x-axis).

        This is a wrapper function of :func:`seaborn.heatmap` to show differences between two sets of sequences on
        positional level (e.g., on level of amino acids for protein sequences).

        Parameters
        ----------
        df_feat : :class:`~pandas.DataFrame`, shape (n_feature, n_feature_information)
            DataFrame with an unique identifier, scale information, statistics, and positions for each feature.
        y : {'category', 'subcategory', 'scale_name'}, str, default='subcategory'
            Name of column in feature DataFrame for scale information (shown on x axis).
        val_col : {'mean_dif', 'feat_impact', 'abs_auc', 'std_test', ...} str, default='mean_dif'
            Name of column in feature DataFrame with numerical values to show.
        val_type : {'mean', 'sum', 'std'}, str, default='mean'
            How to aggregate numerical values given in 'val_col'.
        normalize : {True, False, 'positions'} bool or str, default=False
            Whether to use normalization for numerical values of 'val_col':

            - If False, value is set at all positions of a feature without posterior normalization over all features.
                Recomanded for 'val_col'='mean_dif' and 'value_type'='mean'.
            - If True, value is set at all positions of a feature and normalized over all features.
                Recomanded for 'val_col'='feat_impact' and 'value_type'='mean'.
            - If 'positions', values/number of positions is set at each position of a feature and normalize over all features.
                Recomanded for 'val_col'='feat_impact' and 'value_type'='mean' if impact of features comprising lower
                number of positions (e.g., Pattern) should be increased.

        figsize : tuple(float, float), default=(10,7)
            Width, height of figure in inches passd to :func:`matplotlib.pyplot.figure`.
        title : str, optional
            Title of figure used by :func:`matplotlib.pyplot.title`.
        title_kws : dict, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.title`.
        vmin, vmax : float, optional
            Values to anchor the colormap, otherwise, inferred from data and other keyword arguments.
        cmap : matplotlib colormap name or object, or list of colors, default='seismic'
            Name of color map assigning data values to color space. If 'SHAP', colors from
            (`SHAP <https://shap.readthedocs.io/en/latest/index.html>`__) will be used (recommended for feature impact).
        cmap_n_colors : int, optional
            Number of discrete steps in diverging or sequential color map.
        dict_color : dict, optional
            Map of colors for scale categories classifying scales shown on y-axis.
        cbar_kws : dict of key, value mappings, optional
            Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.

        add_jmd_tmd : bool, default=True
            Whether to add colored bar under heatmap indicating sequence parts (JMD-N, TMD, JMD-C).
        tmd_len : int, >0
            Length of TMD to be depiceted.
        start : int, >=0
            Position label of first amino acid position (starting at N-terminus).

        tmd_seq : str, optional
            Sequence of TMD. 'tmd_len' is set to length of TMD if sequence for TMD, JMD-N and JMD-C are given.
            Recommended if feature impact or mean difference should be depicted for one sample.
        jmd_n_seq : str, optional
            Sequence of JMD_N. 'jmd_n_len' is set to length of JMD_N if sequence for TMD, JMD-N and JMD-C are given.
            Recommended if feature impact or mean difference should be depicted for one sample.
        jmd_c_seq : str, optional
            Sequence of JMD_C. 'jmd_c_len' is set to length of JMD_C if sequence for TMD, JMD-N and JMD-C are given.
            Recommended if feature impact or mean difference should be depicted for one sample.
        tmd_color : str, default='mediumspringgreen'
            Color of TMD bar.
        jmd_color : str, default='blue'
            Color of JMD-N and JMD-C bar.
        tmd_seq_color : str, default='black'
            Color of TMD sequence.
        jmd_seq_color : str, defaault='white'
            Color of JMD-N and JMD-C sequence.
        seq_size : float, optional
            Font size of all sequence parts in points. If None, optimized automatically.
        tmd_fontsize : float, optional
            Font size of 'TMD' label in points. If None, optimized automatically.
        jmd_fontsize : float, optional
            Font size of 'JMD-N' and 'JMD-C' labels in points. If None, optimized automatically.

        xticks_top : bool, default=True
            Whether to show x ticks depicting positions on top of plot.
        xtick_size : float, default=11.0
            Size of x ticks in points. Passed as 'size' argument to :meth:`matplotlib.axes.Axes.set_xticklabels`.
        xtick_width : float, default=2.0
            Widht of x ticks in points. Passed as 'width' argument to :meth:`matplotlib.axes.Axes.tick_params`.
        xtick_length : float, default=5.0,
            Length of x ticks in points. Passed as 'length' argument to :meth:`matplotlib.axes.Axes.tick_params`.
        ytick_size : float, optional
            Size of scale information as y ticks in points. Passed to :meth:`matplotlib.axes.Axes.tick_params`.
            If None, optimized automatically.

        add_legend_cat : bool, default=True,
            Whether to add legend for categories under plot and classification of scales at y-axis.
        legend_kws : dict, optional
            Keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`

        kwargs : other keyword arguments
            All other keyword arguments passed to :meth:`matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        ax : matplotlib Axex
            Axes object with the heatmap.

        Warning
        -------
        - 'cmap_n_colors' only works properly if 'vmin' and 'vmax' match data.
        - 'tmd_seq_color' and 'jmd_seq_color' can only be applied if 'tmd_seq', 'jmd_n_seq', and 'jmd_c_seq' are given.

        See Also
        --------
        :func:`seaborn.heatmap`
        `SHAP <https://shap.readthedocs.io/en/latest/index.html>`__

        Examples
        --------

        Plot CPP feature heatmap:

        .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> import aaanalysis as aa
            >>> sf = aa.SequenceFeature()
            >>> df_seq, labels = sf.load_sequences(return_labels=True)
            >>> df_parts = sf.get_df_parts(df_seq=df_seq)
            >>> cpp = aa.CPP(df_parts=df_parts)
            >>> df_feat = cpp.run(labels=labels)
            >>> cpp.plot_heatmap(df_feat=df_feat)
            >>> plt.tight_layout()

        """
        # Group arguments
        args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq)
        args_size = check_args_size(seq_size=seq_size, tmd_fontsize=tmd_fontsize, jmd_fontsize=jmd_fontsize)
        args_len = check_args_len(tmd_len=tmd_len, jmd_n_len=self.jmd_n_len, jmd_c_len=self.jmd_c_len, **args_seq)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)

        # Checking input
        # Args checked by Matplotlib: title, cmap, cbar_kws, legend_kws
        ut.check_non_negative_number(name="start", val=start, min_val=0)
        ut.check_non_negative_number(name="ytick_size", val=ytick_size, accept_none=True, just_int=False, min_val=1)
        ut.check_non_negative_number(name="cmap_n_colors", val=cmap_n_colors, min_val=1, accept_none=True)
        ut.check_bool(name="add_jmd_tmd", val=add_jmd_tmd)
        ut.check_bool(name="xticks_top", val=xticks_top)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_dict(name="cbar_kws", val=cbar_kws, accept_none=True)
        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=val_col, type_check="numerical")
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat, df_cat=self.df_cat)
        check_value_type(val_type=val_type, count_in=False)
        check_vmin_vmax(vmin=vmin, vmax=vmax)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self.df_cat)

        # Get df positions
        df_feat = self.add_positions(df_feat=df_feat, tmd_len=args_len["tmd_len"], start=start)
        df_pos = _get_df_pos(df_feat=df_feat, df_cat=self.df_cat, y=y, val_col=val_col,
                             value_type=val_type, normalize=normalize, start=start, **args_len)
        # Plotting
        cpp_plot = CPPPlots(**args_len, start=start)
        cpp_plot.set_figsize(figsize=figsize)   # figsize is not used as argument in seaborn (but in pandas)
        try:
            subplots = "ax" in kwargs
            ax = cpp_plot.heatmap(df_pos=df_pos, vmin=vmin, vmax=vmax, grid_on=grid_on,
                                  cmap=cmap, cmap_n_colors=cmap_n_colors, cbar_kws=cbar_kws,
                                  x_shift=0.5, ytick_size=ytick_size, **args_xtick, **kwargs)

            if not grid_on:
                ax.axvline(x=self.jmd_n_len, ls="--" , color="black")
                ax.axvline(x=self.jmd_n_len + tmd_len, ls="--", color="black")
        except AttributeError as e:
            error_message = check_parameters(func=self.plot_heatmap, name_called_func="sns.heatmap", e=e)
            raise AttributeError(error_message)
        cpp_plot.set_title(title=title, title_kws=title_kws)
        # Autosize tmd sequence & annotation
        opt_size = cpp_plot.optimized_size(ax=ax, df_pos=df_pos)
        args_size = _set_size_to_optimized_value(**args_size, opt_size=opt_size)
        # Add tmd_jmd sequence if sequence is given
        if isinstance(tmd_seq, str):
            cpp_plot.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                     xticks_top=xticks_top, x_shift=0.5, xtick_size=xtick_size)
        # Add tmd_jmd bar
        elif add_jmd_tmd:
            args_part_size = {x: args_size[x] for x in args_size if x != "seq_size"}
            cpp_plot.add_tmd_jmd_bar(ax=ax, **args_part_color)
            cpp_plot.add_tmd_jmd_xticks(ax=ax, x_shift=0.5, **args_xtick)
            cpp_plot.add_tmd_jmd_text(ax=ax, x_shift=0, **args_part_size)
        # Add default x ticks on top of plot
        if xticks_top:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            cpp_plot.add_xticks(ax=ax2, xticks_position="top", x_shift=0.5, **args_xtick)
        # Add scale classification
        if add_legend_cat:
            ax = cpp_plot.add_legend_cat(ax=ax, df_pos=df_pos, df_cat=self.df_cat, y=y,
                                         dict_color=dict_color, legend_kws=legend_kws, legend_y_adjust=legend_y_adjust)

        # Set current axis to main axis object depending on tmd sequence given or not
        n = 2 if isinstance(tmd_seq, str) else 0
        plt.sca(plt.gcf().axes[n])
        return ax

    def plot_bargraph(self, df_feat=None, y="subcategory", val_col="mean_dif",
                      figsize=(7, 5), title=None, title_kws=None,
                      bar_color="steelblue", edge_color="none", bar_width=0.75,
                      add_jmd_tmd=True, tmd_len=20, start=1,
                      jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                      tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                      seq_size=None, tmd_fontsize=None, jmd_fontsize=None,
                      xticks_top=True, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                      ytick_size=None, ytick_width=2.0, ytick_length=5.0, ylim=None,
                      highlight_tmd_area=True, highlight_alpha=0.25,
                      grid=False, grid_axis="both",
                      **kwargs):
        """"""
        # Group arguments
        args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,)
        args_size = check_args_size(seq_size=seq_size, tmd_fontsize=tmd_fontsize, jmd_fontsize=jmd_fontsize)
        args_len = check_args_len(tmd_len=tmd_len, jmd_n_len=self.jmd_n_len, jmd_c_len=self.jmd_c_len, **args_seq)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)

        # Checking input
        # Args checked by Matplotlib: title
        # Args checked by internal plotting functions: ylim
        ut.check_non_negative_number(name="bar_width", val=bar_width, min_val=0, just_int=False)
        ut.check_non_negative_number(name="start", val=start, min_val=0)
        ut.check_non_negative_number(name="tmd_area_alpha", val=highlight_alpha, min_val=0, max_val=1, just_int=False)
        ut.check_bool(name="add_jmd_tmd", val=add_jmd_tmd)
        ut.check_bool(name="xticks_top", val=xticks_top)
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_bool(name="grid", val=grid)
        ut.check_color(name="bar_color", val=bar_color)
        ut.check_color(name="edge_color", val=edge_color, accept_none=True)

        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=val_col, type_check="numerical")
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_figsize(figsize=figsize)
        check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        check_grid_axis(grid_axis=grid_axis)

        # Get df positions
        df_feat = self.add_positions(df_feat=df_feat, tmd_len=args_len["tmd_len"], start=start)
        df_pos = _get_df_pos(df_feat=df_feat, df_cat=self.df_cat, y=y, val_col=val_col, value_type="count", **args_len)

        # Plotting
        cpp_plot = CPPPlots(**args_len, start=start)
        try:
            ax = cpp_plot.barplot(df_pos=df_pos, ylim=ylim, figsize=figsize,
                                  bar_color=bar_color, edge_color=edge_color, bar_width=bar_width,
                                  x_shift=0, **args_xtick, **kwargs)
        except AttributeError as e:
            error_message = check_parameters(func=self.plot_bargraph, name_called_func="pd.DataFrame.plot", e=e)
            raise AttributeError(error_message)
        cpp_plot.set_title(title=title, title_kws=title_kws)
        # Autosize tmd sequence & annotation
        opt_size = cpp_plot.optimized_size(ax=ax, df_pos=df_pos, label_term=False)
        args_size = _set_size_to_optimized_value(**args_size, opt_size=opt_size)
        # Add tmd_jmd sequence if sequence is given
        if type(tmd_seq) == str:
            cpp_plot.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                     xticks_top=False, heatmap=False, x_shift=0, xtick_size=xtick_size)
        # Add tmd_jmd bar
        elif add_jmd_tmd:
            args_part_size = {x: args_size[x] for x in args_size if x != "seq_size"}
            cpp_plot.add_tmd_jmd_bar(ax=ax, x_shift=-0.5, **args_part_color, add_white_bar=False)
            cpp_plot.add_tmd_jmd_xticks(ax=ax, x_shift=0, **args_xtick)
            cpp_plot.add_tmd_jmd_text(ax=ax, x_shift=-0.5, **args_part_size)
        # Adjust y ticks
        plt.yticks(size=ytick_size)
        plt.tick_params(axis="y", color="black", width=ytick_width, length=ytick_length, bottom=False)
        # Add default x ticks on top of plot
        if xticks_top:
            ax3 = ax.twiny()
            ax3.set_xlim(ax.get_xlim())
            cpp_plot.add_xticks(ax=ax3, xticks_position="top", x_shift=0, **args_xtick)
        else:
            sns.despine(top=True, right=True)
        # Set current axis to main axis object depending on tmd sequence given or not
        n = 1 if type(tmd_seq) == str else 0
        plt.sca(plt.gcf().axes[n])
        # Add grid
        if grid:
            ax.set_axisbelow(True)  # Grid behind datasets
            ax.grid(which="major", axis=grid_axis, linestyle="-")
        # Add tmd area
        if highlight_tmd_area:
            cpp_plot.highlight_tmd_area(ax=ax, x_shift=-0.5, tmd_color=tmd_color, alpha=highlight_alpha)
        # Set default ylabel
        ax = plt.gca()
        ylabel = f"Feature count"
        plt.ylabel(ylabel, size=args_size["seq_size"], weight="bold")
        return ax

    def plot_profile(self, df_feat=None, y="category", val_col="mean_dif", val_type="count", normalize=False,
                     figsize=(7, 5), title=None, title_kws=None,
                     dict_color=None, edge_color="none", bar_width=0.75,
                     add_jmd_tmd=True, tmd_len=20, start=1,
                     jmd_n_seq=None, tmd_seq=None, jmd_c_seq=None,
                     tmd_color="mediumspringgreen", jmd_color="blue", tmd_seq_color="black", jmd_seq_color="white",
                     seq_size=None, tmd_fontsize=None, jmd_fontsize=None,
                     xticks_top=True, xtick_size=11.0, xtick_width=2.0, xtick_length=5.0,
                     ytick_size=None, ytick_width=2.0, ytick_length=5.0, ylim=None,
                     highlight_tmd_area=True, highlight_alpha=0.15,
                     grid=False, grid_axis="both",
                     add_legend_cat=True, legend_kws=None,
                     shap_plot=False,
                     **kwargs):
        """"""
        # Group arguments
        args_seq = dict(jmd_n_seq=jmd_n_seq, tmd_seq=tmd_seq, jmd_c_seq=jmd_c_seq,)
        args_size = check_args_size(seq_size=seq_size, tmd_fontsize=tmd_fontsize, jmd_fontsize=jmd_fontsize)
        args_len = check_args_len(tmd_len=tmd_len, jmd_n_len=self.jmd_n_len, jmd_c_len=self.jmd_c_len, **args_seq)
        args_xtick = check_args_xtick(xtick_size=xtick_size, xtick_width=xtick_width, xtick_length=xtick_length)
        args_part_color = check_part_color(tmd_color=tmd_color, jmd_color=jmd_color)
        args_seq_color = check_seq_color(tmd_seq_color=tmd_seq_color, jmd_seq_color=jmd_seq_color)

        # Checking input
        # Args checked by Matplotlib: title, legend_kws
        # Args checked by internal plotting functions: ylim
        ut.check_non_negative_number(name="bar_width", val=bar_width, min_val=0, just_int=False)
        ut.check_non_negative_number(name="start", val=start, min_val=0)
        ut.check_non_negative_number(name="tmd_area_alpha", val=highlight_alpha, min_val=0, max_val=1, just_int=False)
        ut.check_bool(name="add_jmd_tmd", val=add_jmd_tmd)
        ut.check_bool(name="xticks_top", val=xticks_top)
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_bool(name="grid", val=grid)
        ut.check_bool(name="shap_plot", val=shap_plot)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_color(name="edge_color", val=edge_color, accept_none=True)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)

        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=val_col, type_check="numerical")
        ut.check_y_categorical(df=df_feat, y=y)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        check_value_type(val_type=val_type, count_in=True)
        check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        check_figsize(figsize=figsize)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self.df_cat)
        check_args_ytick(ytick_size=ytick_size, ytick_width=ytick_width, ytick_length=ytick_length)
        check_grid_axis(grid_axis=grid_axis)

        # Get df positions
        df_feat = self.add_positions(df_feat=df_feat, tmd_len=args_len["tmd_len"], start=start)
        df_pos = _get_df_pos(df_feat=df_feat, df_cat=self.df_cat, y=y, val_col=val_col,
                             value_type=val_type, normalize=normalize, start=start, **args_len)
        # Plotting
        cpp_plot = CPPPlots(**args_len, start=start)
        try:
            ax = cpp_plot.profile(df_pos=df_pos, figsize=figsize, ylim=ylim,
                                  dict_color=dict_color, edge_color=edge_color, bar_width=bar_width,
                                  add_legend=add_legend_cat, legend_kws=legend_kws, shap_plot=shap_plot,
                                  **args_xtick, **kwargs)
        except AttributeError as e:
            error_message = check_parameters(func=self.plot_profile, name_called_func="pd.DataFrame.plot", e=e)
            raise AttributeError(error_message)
        cpp_plot.set_title(title=title, title_kws=title_kws)

        # Autosize tmd sequence & annotation
        opt_size = cpp_plot.optimized_size(ax=ax, df_pos=df_pos, label_term=False)
        args_size = _set_size_to_optimized_value(**args_size, opt_size=opt_size)
        # Add tmd_jmd sequence if sequence is given
        if type(tmd_seq) == str:
            cpp_plot.add_tmd_jmd_seq(ax=ax, **args_seq, **args_size, **args_part_color, **args_seq_color,
                                     xticks_top=False, heatmap=False, x_shift=0, xtick_size=xtick_size)
        # Add tmd_jmd bar
        elif add_jmd_tmd:
            args_part_size = {x: args_size[x] for x in args_size if x != "seq_size"}
            cpp_plot.add_tmd_jmd_bar(ax=ax, x_shift=-0.5, **args_part_color, add_white_bar=False)
            cpp_plot.add_tmd_jmd_xticks(ax=ax, x_shift=0, **args_xtick)
            cpp_plot.add_tmd_jmd_text(ax=ax, x_shift=-0.5, **args_part_size)
        # Adjust y ticks
        plt.yticks(size=ytick_size)
        plt.tick_params(axis="y", color="black", width=ytick_width, length=ytick_length, bottom=False)
        # Add default x ticks on top of plot
        if xticks_top:
            ax3 = ax.twiny()
            ax3.set_axisbelow(True)
            ax3.set_xlim(ax.get_xlim())
            cpp_plot.add_xticks(ax=ax3, xticks_position="top", x_shift=0, **args_xtick)
        else:
            sns.despine(top=True, right=True)
        # Set current axis to main axis object depending on tmd sequence given or not
        n = 1 if type(tmd_seq) == str else 0
        plt.sca(plt.gcf().axes[n])
        # Add grid
        if grid:
            ax.set_axisbelow(True)  # Grid behind datasets
            ax.grid(which="major", axis=grid_axis, linestyle="-")
        # Add tmd area
        if highlight_tmd_area:
            cpp_plot.highlight_tmd_area(ax=ax, x_shift=-0.5, tmd_color=tmd_color, alpha=highlight_alpha)
        # Set default ylabel
        ax = plt.gca()
        ylabel = "Feature impact" if shap_plot else f"Feature count (-/+ {val_col})"
        plt.ylabel(ylabel, size=args_size["seq_size"], weight="bold")
        return ax

    def plot_stat(self, df_feat=None, figsize=(7, 5),
                  val_col="abs_mean_dif", percent_v=True, ylim_v=None,
                  color_v="silver", ylabel_v=None,
                  pval_col="p_val_mann_whitney", neg_log_p=True, ylim_p=None,
                  color_p="black", ylabel_p=None, min_p=0.001,
                  ylabel_fontsize=12, ylabel_fontweight="medium",
                  add_cat=True, add_legend_cat=True, legend_kws=None, legend_y_adjust=-0.05, dict_color=None,
                  highlight_cat=False, highlight_alpha=0.075, **kwargs):
        """
        Show p-value and effect size (e.g., mean differences) over all features in descending order.

        Parameters
        ----------
        df_feat:
        val_col:
        pval_col:
        percent_v:
        neg_log_p:
        ylim_v:
        ylim_p:
        color_v:
        color_p:
        ylabel_v:
        ylabel_p:
        min_p:
        ylabel_fontsize:
        ylabel_fontweight: {0-1000, 'light', 'medium', 'bold'}
        add_cat:
        add_legend_cat:
        legend_kws:
        dict_color:
        highlight_cat:
        highlight_alpha:

        Returns
        -------

        """
        # TODO check (test) ylabel_v/p, ylabel_weight
        # Checking input
        # Args checked by Matplotlib: title, legend_kws
        ut.check_non_negative_number(name="highlight_alpha", val=highlight_alpha, min_val=0, max_val=1, just_int=False)
        ut.check_non_negative_number(name="ylabel_fontsize", val=ylabel_fontsize, min_val=1, accept_none=True, just_int=False)
        ut.check_non_negative_number(name="min_p,", val=min_p, min_val=0, max_val=1, just_int=False)
        ut.check_bool(name="highlight_cat", val=highlight_cat)
        ut.check_bool(name="percent_v", val=percent_v)
        ut.check_bool(name="add_cat", val=add_cat)
        ut.check_bool(name="add_legend_cat", val=add_legend_cat)
        ut.check_dict(name="legend_kws", val=legend_kws, accept_none=True)
        ut.check_color(name="color_v", val=color_v)
        ut.check_color(name="color_p", val=color_p)
        ut.check_str(name="ylabel_v", val=ylabel_v, accept_none=True)
        ut.check_str(name="ylabel_p", val=ylabel_p, accept_none=True)

        check_figsize(figsize=figsize)
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=val_col, type_check="numerical")
        ut.check_col_in_df(df=df_feat, name_df="df_feat", col=pval_col, type_check="numerical")
        ut.check_ylim(df=df_feat, ylim=ylim_v, val_col=val_col)
        ut.check_ylim(df=df_feat, ylim=ylim_p, val_col=pval_col)
        dict_color = check_dict_color(dict_color=dict_color, df_cat=self.df_cat)
        check_ylabel_fontweight(ylabel_fontweight=ylabel_fontweight)
        # Plotting
        plt.figure(figsize=figsize)
        ax = cpp_statistics(df=df_feat, df_cat=self.df_cat, dict_color=dict_color,
                            col_p=pval_col, neg_log_p=neg_log_p, ylim_p=ylim_p, color_p=color_p, ylabel_p=ylabel_p,
                            min_p=min_p,
                            col_v=val_col, percent_v=percent_v, ylim_v=ylim_v, color_v=color_v, ylabel_v=ylabel_v,
                            ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight,
                            add_cat=add_cat, add_legend_cat=add_legend_cat,
                            legend_kws=legend_kws, legend_y_adjust=legend_y_adjust,
                            highlight_cat=highlight_cat, highlight_alpha=highlight_alpha, **kwargs)
        return ax

