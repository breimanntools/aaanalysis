"""This is a script with the backend for the CPP class"""
from ._utils_cpp import get_dict_part_pos, _get_positions, get_feat_matrix
import aaanalysis.utils as ut


# I Helper functions


# II Main functions
# Feature information methods (can be included to df_feat for individual sequences)
def get_positions(features=None, start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=4):
    """Create list with positions for given feature names"""
    # TODO add sequence, generalize check functions for tmd_len ...
    args = dict(jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, ext_len=ext_len, start=start)
    dict_part_pos = get_dict_part_pos(tmd_len=tmd_len, **args)
    feat_positions = _get_positions(dict_part_pos=dict_part_pos, features=features)
    return feat_positions


def get_dif(features=None, df_parts=None, df_scales=None, accept_gaps=False,
            list_names=None, sample_name=str, labels=None, ref_group=0):
    """
    Add feature value difference between sample and reference group to DataFrame.
    """
    X = get_feat_matrix(features=features,
                        df_parts=df_parts,
                        df_scales=df_scales,
                        accept_gaps=accept_gaps)
    mask = [True if x == ref_group else False for x in labels]
    i = list_names.index(sample_name)
    feat_dif = X[i] - X[mask].mean()
    return feat_dif
