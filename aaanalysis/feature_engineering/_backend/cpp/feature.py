"""
Script for SequenceFeature() object that combines scales, splits, and parts to create
    feature names, feature values, or a feature matrix for ML or CPP pipelines.
"""
import os
import pandas as pd
import numpy as np
import math
from itertools import repeat
import multiprocessing as mp
import warnings

from aaanalysis.feature_engineering._backend.cpp._feature_pos import SequenceFeaturePositions
from aaanalysis.feature_engineering._backend.cpp._split import Split, SplitRange
from aaanalysis.feature_engineering._backend.cpp._part import Parts

import aaanalysis as aa
import aaanalysis.utils as ut

# TODO simplify and check


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


# Functions to create feature (part + split + scale)
def _get_feature_components(feat_name=None, dict_all_scales=None):
    """Convert feature name into three feature components of part, split, and scale given as dictionary"""
    if feat_name is None or dict_all_scales is None:
        raise ValueError("'feature_name' and 'dict_all_scales' must be given")
    part, split, scale = feat_name.split("-")
    if scale not in dict_all_scales:
        raise ValueError("'scale' from 'feature_name' is not in 'dict_all_scales")
    dict_scale = dict_all_scales[scale]
    return part, split, dict_scale


def _feature_value(df_parts=None, split=None, dict_scale=None, accept_gaps=False):
    """Helper function to create feature values for feature matrix"""
    sp = Split()
    # Get vectorized split function
    split_type, split_kwargs = ut.check_split(split=split)
    f_split = getattr(sp, split_type.lower())
    # Vectorize split function using anonymous function
    vf_split = np.vectorize(lambda x: f_split(seq=x, **split_kwargs))
    # Get vectorized scale function
    vf_scale = ut.get_vf_scale(dict_scale=dict_scale, accept_gaps=accept_gaps)
    # Combine part split and scale to get feature values
    part_split = vf_split(df_parts)
    feature_value = np.round(vf_scale(part_split), 5)  # feature values
    return feature_value


def _feature_matrix(feat_names, dict_all_scales, df_parts, accept_gaps):
    """Helper function to create feature matrix via multiple processing"""
    feat_matrix = np.empty([len(df_parts), len(feat_names)])
    for i, feat_name in enumerate(feat_names):
        part, split, dict_scale = _get_feature_components(feat_name=feat_name,
                                                          dict_all_scales=dict_all_scales)
        check_parts_in_df_parts(df_parts=df_parts, part=part)
        feat_matrix[:, i] = _feature_value(split=split,
                                           dict_scale=dict_scale,
                                           df_parts=df_parts[part.lower()],
                                           accept_gaps=accept_gaps)
    return feat_matrix
    

# II Main Functions
class SequenceFeature:
    """Retrieve and create sequence feature components (Part, Split, and Scale).

    Notes
    -----
    Part: Feature Component
        A continuous subset of a sequence like a protein domain (e.g, transmembrane domain of membrane proteins).

    Split: Feature Component
        Principle to obtain a distinct subset of amino acids from a sequence part like a segment or a pattern.

    Scale: Feature Component
        A physicochemical scale assigning  each amino acid a numerical value between 0 and 1.

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

    # Basic datastructures for features
    @staticmethod
    def get_df_parts(df_seq=None, list_parts=None, jmd_n_len=None, jmd_c_len=None, ext_len=None, all_parts=False):
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
        ut.check_args_len(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len, accept_tmd_none=True)
        df_seq = ut.check_df_seq(df_seq=df_seq, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        seq_info_in_df = set(ut.COLS_SEQ_TMD_POS_KEY).issubset(set(df_seq))
        pa = Parts()
        dict_parts = {}
        for i, row in df_seq.iterrows():
            entry = row[ut.COL_ENTRY]
            if jmd_c_len is not None and jmd_n_len is not None and seq_info_in_df:
                seq, start, stop = row[ut.COLS_SEQ_TMD_POS_KEY].values
                parts = pa.create_parts(seq=seq, tmd_start=start, tmd_stop=stop,
                                        jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
                jmd_n, tmd, jmd_c = parts.jmd_n, parts.tmd, parts.jmd_c
            else:
                jmd_n, tmd, jmd_c = row[ut.COLS_PARTS].values
            dict_part_seq = pa.get_dict_part_seq(tmd_seq=tmd, jmd_n_seq=jmd_n, jmd_c_seq=jmd_c, ext_len=ext_len)
            dict_part_seq = {part: dict_part_seq[part] for part in list_parts}
            dict_parts[entry] = dict_part_seq
        df_parts = pd.DataFrame.from_dict(dict_parts, orient="index")
        return df_parts

    @staticmethod
    def get_split_kws(n_split_min=1, n_split_max=15, steps_pattern=None, n_min=2, n_max=4, len_max=15,
                      steps_periodicpattern=None, split_types=None):
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
        split_types = check_split_types(split_types=split_types)
        args_int = dict(n_split_min=n_split_min, n_split_max=n_split_max, n_min=n_min, n_max=n_max, len_max=len_max)
        check_split_int_args(kwargs_int=args_int)
        args_list = dict(steps_pattern=steps_pattern, steps_periodicpattern=steps_periodicpattern)
        check_split_list_args(kwargs_list=args_list)
        if steps_pattern is None:
            # Differences between interacting amino acids in helix (without gaps) include 6, 7 ,8 to include gaps
            steps_pattern = [3, 4]
        if steps_periodicpattern is None:
            steps_periodicpattern = [3, 4]      # Differences between interacting amino acids in helix (without gaps)
        split_kws = {ut.STR_SEGMENT: dict(n_split_min=n_split_min, n_split_max=n_split_max),
                     ut.STR_PATTERN: dict(steps=steps_pattern, n_min=n_min, n_max=n_max, len_max=len_max),
                     ut.STR_PERIODIC_PATTERN: dict(steps=steps_periodicpattern)}
        split_kws = {x: split_kws[x] for x in split_types}
        ut.check_split_kws(split_kws=split_kws)
        return split_kws

    def get_features(self, list_parts=None, split_kws=None, df_scales=None, all_parts=False):
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
        list_parts = ut.check_list_parts(list_parts=list_parts, all_parts=all_parts)
        ut.check_split_kws(split_kws=split_kws)
        ut.check_df_scales(df_scales=df_scales, accept_none=True)
        if df_scales is None:
            df_scales = aa.load_scales()
        if split_kws is None:
            split_kws = self.get_split_kws()
        scales = list(df_scales)
        spr = SplitRange()
        features = []
        for split_type in split_kws:
            args = split_kws[split_type]
            labels_s = getattr(spr, "labels_" + split_type.lower())(**args)
            features.extend(["{}-{}-{}".format(p.upper(), s, sc)
                             for p in list_parts
                             for s in labels_s
                             for sc in scales])
        return features

    @staticmethod
    def feat_matrix(features=None, df_parts=None, df_scales=None, accept_gaps=False,
                    n_jobs=None, verbose=False, return_labels=False):
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
            if n_vals > 1000*1000:
                warning = f"Feature matrix with n={n_vals}>=10^6 values will be created, which will take some time.\n" \
                          "It is recommended to create a feature matrix for a pre-selected number features " \
                          "so that 10^6 values are not exceeded."
                warnings.warn(warning)
        # Create feature matrix using parallel processing
        dict_all_scales = ut.get_dict_all_scales(df_scales=df_scales)
        n_processes = min([os.cpu_count(), len(features)]) if n_jobs is None else n_jobs
        feat_chunks = np.array_split(features, n_processes)
        args = zip(feat_chunks, repeat(dict_all_scales), repeat(df_parts), repeat(accept_gaps))
        with mp.get_context("spawn").Pool(processes=n_processes) as pool:
            result = pool.starmap(_feature_matrix, args)
        feat_matrix = np.concatenate(result, axis=1)
        if return_labels:
            if verbose:
                print("Tuple of (feat_matrix, labels) will be returned")
            labels = df_parts.index.tolist()
            return feat_matrix, labels  # X, y
        else:
            if verbose:
                print("Only feat_matrix (without labels) will be returned")
            return feat_matrix  # X

    # Additional feature related methods
    @staticmethod
    def feat_names(features=None, df_cat=None, tmd_len=20, jmd_c_len=10, jmd_n_len=10, ext_len=0, start=1):
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
        # Check input (length checked in SequenceFeaturePositions)
        features = ut.check_features(features=features)
        ut.check_df_cat(df_cat=df_cat)
        if df_cat is None:
            df_cat = aa.load_scales(name=ut.STR_SCALE_CAT)
        # Get feature names
        sfp = SequenceFeaturePositions()
        dict_part_pos = sfp.get_dict_part_pos(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                              ext_len=ext_len, start=start)
        list_positions = sfp.get_positions(dict_part_pos=dict_part_pos, features=features)
        dict_scales = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SCALE_NAME]))
        feat_names = []
        for feat_id, pos in zip(features, list_positions):
            part, split, scale = feat_id.split("-")
            split_type = split.split("(")[0]
            if split_type == ut.STR_SEGMENT and len(pos.split(",")) > 2:
                pos = pos.split(",")[0] + "..." + pos.split(",")[-1]
            if split_type == ut.STR_PERIODIC_PATTERN:
                step = split.split("+")[1].split(",")[0]
                pos = pos.split(",")[0] + ".." + step + ".." + pos.split(",")[-1]
            feat_names.append(f"{dict_scales[scale]} [{pos}]")
        return feat_names

    # Feature: Part + Split + Scale
    # For what used? Not redudant with feature matrix?
    # TODO Add functions (modify df_feat)
    @staticmethod
    def add_feat_value(df_parts=None, split=None, dict_scale=None, accept_gaps=False):
        """Create feature values for all sequence parts by combining Part, Split, and Scale.

        Parameters
        ----------
        df_parts: :class:`pandas.DataFrame`
            DataFrame with sequence parts.
        split: str
            Name of Split following given convention.
        dict_scale: dict
            Dictionary mapping a numerical value to each letter of given sequences
        accept_gaps: bool, default = False
            Whether to accept missing values by enabling omitting for computations (if True).

        Returns
        -------
        feature_value: array-like, shape (n_samples, n_parts)
            Average scale values over sequence parts.

        Notes
        -----
        A split name should has the form of PART-SPLIT-SCALE, where following structures
        are given for the three split types:

        - Segment(i-th,n_split)
            with i-th<=n_split and
            where 'i-th' and 'n_split' indicate the i-th Segment resp. the number of Segments.

        - Pattern(N/C,p1,p2,...,pn)
            with p1<p2<...<pn indicating amino acid positions and
            'N/C' whether the splits starts from the N resp. C-terminal sequence end.

        - PeriodicPattern(N/C,i+step1/step2,start)
            where 'step1/step2' indicates the step size of each odd resp. even step and
            'start' gives the first position starting from the N- or C-terminal sequence end.

        All numbers should be non-negative integers. Examples for each split type
        are as follows: 'Segment(5,7)', 'Pattern(C,1,2)', 'PeriodicPattern(N,i+2/3,1)'.
        """
        ut.check_df_parts(df_parts=df_parts)
        ut.check_split(split=split)
        check_dict_scale(dict_scale=dict_scale, df_parts=df_parts, accept_gaps=accept_gaps)
        feature_value = _feature_value(df_parts=df_parts,
                                       split=split,
                                       dict_scale=dict_scale,
                                       accept_gaps=accept_gaps)
        return feature_value

    @staticmethod
    def add_dif(df_feat=None, df_seq=None, labels=None, sample_name=str, ref_group=0,
                accept_gaps=False, jmd_n_len=10, jmd_c_len=10, df_parts=None, df_scales=None):
        """
        Add feature value difference between sample and reference group to DataFrame.

        Parameters
        ----------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame (CPP output) to add sample difference.
        df_seq: :class:`pandas.DataFrame`
            DataFrame with sequences and sample names, in which the given sample name is included.
        labels: array-like, shape (n_samples)
            Class labels for samples in sequence DataFrame.
        sample_name: str
            Name of sample for which the feature value difference to a given reference group should be computed.
        ref_group: int, default = 0
            Class label of reference group.
        accept_gaps: bool, default = False
            Whether to accept missing values by enabling omitting for computations (if True).

        Returns
        -------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame including feature value difference.
        """
        # Check input
        df_feat = ut.check_df_feat(df_feat=df_feat)
        ut.check_df_seq(df_seq=df_seq, jmd_c_len=jmd_c_len, jmd_n_len=jmd_n_len)
        ut.check_labels_(labels=labels, df=df_seq, name_df="df_seq")
        check_ref_group(ref_group=ref_group, labels=labels)
        check_sample_in_df_seq(sample_name=sample_name, df_seq=df_seq)
        # Add sample difference to reference group
        sf = SequenceFeature()
        X = sf.feat_matrix(features=list(df_feat["feature"]),
                           df_parts=df_parts,
                           df_scales=df_scales,
                           accept_gaps=accept_gaps)
        mask = [True if x == ref_group else False for x in labels]
        i = list(df_seq[ut.COL_NAME]).index(sample_name)
        df_feat[f"dif_{sample_name}"] = X[i] - X[mask].mean()
        return df_feat

    @staticmethod
    def add_position(df_feat=None, features=None, start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=4,
                     part_split=False):
        """Create list with positions for given feature names

        Parameters
        ----------
        df_feat: :class:`pandas.DataFrame`
            Feature DataFrame (CPP output) to add sample difference.
        features: str, list of strings, pd.Series
            Ids of features for which feature names should be created.
        start: int, >=0, default = 1
            Position label of first amino acid position (starting at N-terminus).
        tmd_len: int, >0, default = 20
            Length of TMD.
        jmd_n_len : int, >=0, default = 10
            Length of JMD-N.
        jmd_c_len : int, >=0, default = 10
            Length of JMD-C.
        ext_len : int, >=0, default = 4
            Length of TMD-extending part (starting from C and N terminal part of TMD).
            Conditions: ext_len < jmd_m_len and ext_len < jmd_c_len.

        Returns
        -------
        feat_positions: list
            list with positions for each feature in feat_names

        Notes
        -----
        The length parameters define the total number of positions (jmd_n_len + tmd_len + jmd_c_len).
        """
        # TODO add sequence, generalize check functions for tmd_len ...
        features = ut.check_features(features=features)
        ut.check_number_range(name="tmd_len", val=tmd_len, just_int=True, min_val=1)
        args = dict(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len, start=start)
        for name in args:
            ut.check_number_range(name=name, val=args[name], just_int=True, min_val=0)
        sfp = SequenceFeaturePositions()
        dict_part_pos = sfp.get_dict_part_pos(tmd_len=tmd_len, **args)
        feat_positions = sfp.get_positions(dict_part_pos=dict_part_pos, features=features)
        return feat_positions
