"""
This is a script for the backend of the AAlogo class methods.
"""
import warnings
import logomaker

import aaanalysis.utils as ut


# I Helper Functions
def _retrieve_tmd_aligned(df_parts=None, tmd_len=None, start_n=True):
    """Align TMD sequences to uniform length by N- or C-terminal gap padding."""
    if ut.COL_TMD not in list(df_parts):
        return df_parts
    _df_tmd = df_parts[ut.COL_TMD].copy()
    if tmd_len is None:
        tmd_len = _df_tmd.apply(len).max()
    if start_n:
        _df_tmd = _df_tmd.apply(lambda seq: seq + "-" * max(0, tmd_len - len(seq)))
        _df_tmd = _df_tmd.apply(lambda seq: seq[:tmd_len])
    else:
        _df_tmd = _df_tmd.apply(lambda seq: "-" * max(0, tmd_len - len(seq)) + seq)
        _df_tmd = _df_tmd.apply(lambda seq: seq[-tmd_len:])
    df_parts = df_parts.copy()
    df_parts[ut.COL_TMD] = _df_tmd
    return df_parts


def _get_sequences(df_parts=None):
    """Concatenate sequence parts row-wise and return list of combined sequences."""
    df_merged = df_parts.apply(lambda x: "".join(x), axis=1)
    list_sequences = df_merged.tolist()
    return list_sequences


# II Main Functions
def get_df_logo_(df_parts=None, logo_type="probability",
                 tmd_len=None, start_n=True,
                 characters_to_ignore=".-", pseudocount=0.0):
    """Compute sequence logo matrix for given sequence parts."""
    df_parts = _retrieve_tmd_aligned(df_parts=df_parts, tmd_len=tmd_len, start_n=start_n)
    list_sequences = _get_sequences(df_parts=df_parts)
    args_logo = dict(to_type=logo_type, characters_to_ignore=characters_to_ignore,
                     pseudocount=pseudocount)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_logo = logomaker.alignment_to_matrix(list_sequences, **args_logo)
    return df_logo


def get_df_logo_info_(df_parts=None, tmd_len=None, start_n=True,
                      characters_to_ignore=".-", pseudocount=0.0):
    """Compute per-position information content (in bits) from sequence parts."""
    df_logo = get_df_logo_(df_parts=df_parts, logo_type="information",
                           tmd_len=tmd_len, start_n=start_n,
                           characters_to_ignore=characters_to_ignore,
                           pseudocount=pseudocount)
    df_logo_info = df_logo.sum(axis=1)
    df_logo_info.index.name = "pos"
    return df_logo_info


def get_conservation_(df_logo_info=None, value_type="mean"):
    """Summarize per-position information content into a single conservation score."""
    dict_agg = {
        "min": df_logo_info.min,
        "mean": df_logo_info.mean,
        "median": df_logo_info.median,
        "max": df_logo_info.max,
    }
    cons_val = dict_agg[value_type]()
    return cons_val