"""
This is a script for reading and writing df_seq to the fasta file format, which is the most commonly used format
in computational biology. This fasta format enables a smooth interaction with the biopython package.
"""
import time
import pandas as pd
import numpy as np
from typing import Optional, List

import aaanalysis.utils as ut


# I Helper Functions


# II Main Functions
# TODO finish, docu, test, example ...

def to_fasta(df=None,
             file_path=None,
             col_id="entry",
             col_seq="sequence",
             col_db=None,
             cols_info=None,
             sep="|"):
        """ """
        # Check input
        ut.check_file_path(file_path=file_path)
        ut.check_str(name="col_id", val=col_id, accept_none=False)
        ut.check_str(name="col_seq", val=col_seq, accept_none=False)
        cols_info = ut.check_list_like(name="cols_info", val=cols_info, accept_str=True, accept_none=True)
        cols_requiered = [col_id, col_seq]
        if cols_info is not None:
            cols_requiered += cols_info
        ut.check_df(df=df, name="df", cols_requiered=cols_requiered, accept_none=False, accept_nan=False)
        # Create faste
        if ".fasta" not in file_path:
            file_path += ".fasta"
        fasta = open(file_path, "w")
        for i, row in df.iterrows():
            seq = row[col_seq]
            entry = row[col_id]
            if cols_info is not None:
                if isinstance(cols_info, list):
                    str_info = ", ".join([str(row[c]) for c in cols_info])
                else:
                    str_info = row[cols_info]
                fasta.write(">" + entry + "," + str_info + "\n" + seq + "\n")
            else:
                fasta.write(">" + entry + "\n" + seq + "\n")
        fasta.close()


