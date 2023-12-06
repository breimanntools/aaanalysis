"""
This is a script for reading and writing df_seq to the fasta file format, which is the most commonly used format
in computational biology. This fasta format enables a smooth interaction with the biopython package.
See https://bioperl.org/formats/sequence_formats/FASTA_sequence_format for description of input fasta format.
"""
import time
import pandas as pd
import numpy as np

import aaanalysis.utils as ut



# I Helper Functions
# TODO add more parsers for often used data formats in computational biology (make overview)


# II Main Functions
def read_fasta():
    """"""
    # TODO implement a fasta parser to df_seq


def to_fasta(df=None, fasta_name=None, col_id=None, col_seq=None, cols_info=None):
        """"""
        # Check input
        ut.check_str(name="col_id", val=col_id, accept_none=False)
        ut.check_str(name="col_seq", val=col_seq, accept_none=False)
        cols_info = ut.check_list_like(name="cols_info", val=cols_info, accept_str=True, accept_none=True)
        cols_requiered = [col_id, col_seq]
        if cols_info is not None:
            cols_requiered += cols_info
        ut.check_df(df=df, name="df", cols_requiered=cols_requiered, accept_none=False, accept_nan=False)
        # Create faste
        if ".fasta" not in fasta_name:
            fasta_name += ".fasta"
        fasta = open(fasta_name, "w")
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



def to_df_scales(df=None):
    """"""
    # TODO implement parser from df to df_seq (remove not necessary columns and adjust naming)