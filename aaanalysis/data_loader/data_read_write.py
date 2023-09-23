"""
This is a script for reading and writing df_seq to the fasta file format, which is the most commonly used format
in computational biology. This fasta format enables a smooth interaction with the biopython package.
See https://bioperl.org/formats/sequence_formats/FASTA_sequence_format for description of input fasta format.
"""
import time
import pandas as pd
import numpy as np


# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


# I Helper Functions


# II Main Functions
def read_fasta():
    """"""
    # TODO implement a fasta parser to df_seq


def to_fasta(df_seq=None):
    """"""
    # TODO implement a writer to fasta from df_seq

def to_df_scales(df=None):
    """"""
    # TODO implement parser from df to df_seq (remove not necessary columns and adjust naming)