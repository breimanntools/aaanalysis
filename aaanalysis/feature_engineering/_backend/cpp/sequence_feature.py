"""
This is a script for the backend of the SequenceFeature() object, a supportive class for the CPP feature engineering.
"""
import aaanalysis.utils as ut
from ._split import SplitRange
from ._utils_cpp import get_positions_, get_feature_matrix_


# I Helper Functions


# II Main functions
# Parts and splits
def get_split_kws_(n_split_min=1, n_split_max=15, steps_pattern=None, n_min=2, n_max=4, len_max=15,
                   steps_periodicpattern=None, split_types=None):
    """Get split kws for CPP class"""
    if steps_pattern is None:
        # Differences between interacting amino acids in helix (without gaps) include 6, 7 ,8 to include gaps
        steps_pattern = [3, 4]
    if steps_periodicpattern is None:
        steps_periodicpattern = [3, 4]  # Differences between interacting amino acids in helix (without gaps)
    split_kws = {ut.STR_SEGMENT: dict(n_split_min=n_split_min, n_split_max=n_split_max),
                 ut.STR_PATTERN: dict(steps=steps_pattern, n_min=n_min, n_max=n_max, len_max=len_max),
                 ut.STR_PERIODIC_PATTERN: dict(steps=steps_periodicpattern)}
    split_kws = {x: split_kws[x] for x in split_types}
    ut.check_split_kws(split_kws=split_kws)
    return split_kws


# Features
def feature_matrix_(features=None, df_parts=None, df_scales=None, accept_gaps=False, n_jobs=None):
    """Create feature matrix for given feature ids and sequence parts."""
    # Create feature matrix using parallel processing
    feat_matrix = get_feature_matrix_(features=features,
                                      df_parts=df_parts,
                                      df_scales=df_scales,
                                      accept_gaps=accept_gaps,
                                      n_jobs=n_jobs)


def get_features_(list_parts=None, split_kws=None, df_scales=None):
    """Create list of all feature ids for given Parts, Splits, and Scales"""
    scales = list(df_scales)
    spr = SplitRange()
    features = []
    for split_type in split_kws:
        args = split_kws[split_type]
        labels_s = getattr(spr, "labels_" + split_type.lower())(**args)
        features.extend(["{}-{}-{}".format(p.upper(), s, sc) for p in list_parts for s in labels_s for sc in scales])
    return features


def get_feature_names_(features=None, df_cat=None, tmd_len=20, jmd_c_len=10, jmd_n_len=10, start=1):
    """Convert feature ids (PART-SPLIT-SCALE) into feature names (scale name [positions])."""
    # Get feature names
    feat_positions = get_positions_(features=features, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                    start=start)
    dict_scales = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SUBCAT]))
    feat_names = []
    for feat_id, pos in zip(features, feat_positions):
        part, split, scale = feat_id.split("-")
        split_type = split.split("(")[0]
        if split_type == ut.STR_SEGMENT and len(pos.split(",")) > 2:
            pos = pos.split(",")[0] + "-" + pos.split(",")[-1]
        if split_type == ut.STR_PERIODIC_PATTERN:
            step = split.split("+")[1].split(",")[0]
            pos = pos.split(",")[0] + ".." + step + ".." + pos.split(",")[-1]
        feat_names.append(f"{dict_scales[scale]} [{pos}]")
    return feat_names


