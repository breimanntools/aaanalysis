"""
This is a script for ...
"""
import logomaker
import warnings
from typing import Optional, Literal, Dict, Union, List, Tuple, Type

import aaanalysis.utils as ut


# I Helper function
def check_df_logo_info(df_logo_info=None):
    """Check if df_logo_info has correct format"""
    ut.check_df(name="df_logo_info", df=df_logo_info,
                check_series=True, accept_none=False, accept_nan=False)
    # Additional check specific to logo info: index name
    if df_logo_info.index.name != "pos":
        raise ValueError("Index name must be 'pos'")


def _adjust_tmd(df_parts=None, tmd_len=None, start_n=False, ):
    """Adjust TMD to have similar length for df logo"""
    if ut.COL_TMD in list(df_parts):
        _df_tmd = df_parts[ut.COL_TMD]
        if tmd_len is None:
            tmd_len = _df_tmd.apply(len).max()
        if start_n:
            _df_tmd = _df_tmd.apply(lambda seq: seq + "-" * max(0, (tmd_len - len(seq))))
            _df_tmd = _df_tmd.apply(lambda seq: seq[:tmd_len])
        else:
            _df_tmd = _df_tmd.apply(lambda seq: "-" * max(0, (tmd_len - len(seq))) + seq)
            _df_tmd = _df_tmd.apply(lambda seq: seq[-tmd_len:])
        df_parts[ut.COL_TMD] = _df_tmd
    return df_parts


def _get_sequences(df_parts=None):
    """Merge parts to one sequence and get list of sequences"""
    # Merge part columns to one dataframe
    df_merge = df_parts.apply(lambda x: ''.join(x), axis=1)
    list_sequences = df_merge.tolist()
    return list_sequences


# TODO refactor into backend, add checks, add docstring, tests, with Claude
# II Main function
class AALogo:
    """
    UNDER CONSTRUCTION - AALogo class for computing sequence logo matrices and conservation scores.
    """

    def __init__(self,
                 logo_type: Literal["probability", "weight", "counts", "information"] = "probability"
                 ):
        list_to_type = ["probability", "weight", "counts", "information"]
        ut.check_str_options(name="to_type", val=logo_type, list_str_options=list_to_type)
        self._logo_type = logo_type

    def get_df_logo(self,
                    df_parts=None,
                    labels=None,
                    label_test=1,
                    tmd_len=None,
                    start_n=True,
                    characters_to_ignore='.-',
                    pseudocount=0.0):

        """Generate sequence logo matrix for JMD-N, TMD, and JMD-C regions."""
        ut.check_df_parts(df_parts=df_parts)
        if labels is not None:
            n_samples = len(df_parts)
            labels = ut.check_labels(labels=labels, len_requiered=n_samples)
            df_parts = df_parts[labels == label_test]
        args_logo = dict(to_type=self._logo_type, characters_to_ignore=characters_to_ignore,
                         pseudocount=pseudocount)
        list_parts = [x for x in ut.COLS_SEQ_PARTS if x in list(df_parts)]
        if len(list_parts) == 0:
            raise ValueError(f"'df_parts' should contain at least on of the following parts: {ut.COLS_SEQ_PARTS}")
        _df_parts = df_parts[list_parts].copy()
        _df_parts = _adjust_tmd(df_parts=_df_parts, tmd_len=tmd_len, start_n=start_n)
        list_sequences = _get_sequences(df_parts=_df_parts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_logo = logomaker.alignment_to_matrix(list_sequences, **args_logo)
        return df_logo

    def get_df_logo_info(self,
                         df_parts=None,
                         labels=None,
                         label_test=1,
                         tmd_len=None,
                         start_n=True,
                         characters_to_ignore='.-',
                         pseudocount=0.0):
        """"""
        logo_type = self._logo_type
        self._logo_type = "information"
        df_logo = self.get_df_logo(df_parts=df_parts, labels=labels, label_test=label_test, tmd_len=tmd_len,
                                   start_n=start_n, characters_to_ignore=characters_to_ignore, pseudocount=pseudocount)
        self._logo_type = logo_type
        df_logo_info = df_logo.sum(axis=1)  # vals_sum_per_pos
        return df_logo_info

    @staticmethod
    def get_conservation(df_logo_info=None,
                         value_type: Literal["min", "mean", "median", "max"] = "mean"):
        """Compute conservation scores from sequence logos, ranging from 0 (no conservation) to
        4.248 (completely conserved)."""
        check_df_logo_info(df_logo_info=df_logo_info)
        # Compute the statistic for each scale
        if value_type == "min":
            cons_val = df_logo_info.min()
        elif value_type == "mean":
            cons_val = df_logo_info.mean()
        elif value_type == "median":
            cons_val = df_logo_info.median()
        else:
            cons_val = df_logo_info.max()
        return cons_val
