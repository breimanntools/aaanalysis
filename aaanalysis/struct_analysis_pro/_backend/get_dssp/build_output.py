"""
This is a script for the backend of get_dssp; assembles the returned DataFrame
by appending ``ss`` and ``dssp_ok`` columns to a copy of ``df_seq``.
"""
from typing import List, Optional
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
# (none)


# II Main Functions
def build_get_dssp_output(df_seq: pd.DataFrame,
                          ss_per_row: List[Optional[List[str]]],
                          ok_per_row: List[bool]) -> pd.DataFrame:
    """Return a copy of ``df_seq`` with ``ss`` and ``dssp_ok`` columns appended."""
    if len(ss_per_row) != len(df_seq) or len(ok_per_row) != len(df_seq):
        raise RuntimeError(
            f"Internal shape mismatch in build_get_dssp_output: "
            f"len(df_seq)={len(df_seq)}, len(ss_per_row)={len(ss_per_row)}, "
            f"len(ok_per_row)={len(ok_per_row)}")
    df_out = df_seq.copy()
    df_out[ut.COL_SS] = ss_per_row
    df_out[ut.COL_DSSP_OK] = ok_per_row
    return df_out
