"""
This is a script for reading FASTA files to DataFrames (df_seq). FASTA files are the most commonly used format
in computational biology. This function should enable a smooth interaction with the biopython package.
"""
import pandas as pd
from typing import Optional, List
import warnings

import aaanalysis.utils as ut


# I Helper Functions
# Post check functions
def post_check_unique_entries(list_entries=None, col_id=None):
    """Check if entries are unique"""
    list_duplicates = list(set([x for x in list_entries if list_entries.count(x) > 1]))
    if len(list_duplicates) > 0:
        str_warning = (f"Entries from '{col_id}' should be unique. "
                       f"\nFollowing entries are duplicated: {list_duplicates}")
        warnings.warn(str_warning)


def post_check_col_db(df_seq=None, col_db=None, sep="|"):
    """Check if database column is in DataFrame"""
    columns = list(df_seq)
    if col_db is not None and col_db not in columns:
        str_warning = f"'col_db' ('{col_db}') not in 'df_seq'. Check if 'sep' ('{sep}') is matching."
        warnings.warn(str_warning)


def _get_entries_from_fasta(file_path, col_id, col_seq, col_db, sep):
    """Read information from FASTA file and convert to DataFrame"""
    list_entries = []
    dict_current_entry = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if dict_current_entry:
                    # Save the previous sequence before starting a new one
                    list_entries.append(dict_current_entry)
                # Parse the header and prepare a new entry
                list_info = line[1:].split(sep)
                if col_db and len(list_info) > 1:
                    dict_current_entry = {col_id: list_info[1], col_seq: "", col_db: list_info[0]}
                    list_info = list_info[1:]
                else:
                    dict_current_entry = {col_id: list_info[0], col_seq: ""}
                if len(list_info) > 1:
                    for i in range(1, len(list_info[1:])+1):
                        dict_current_entry[f'info{i}'] = list_info[i]
            else:
                dict_current_entry[col_seq] += line
        if dict_current_entry:
            list_entries.append(dict_current_entry)
    df = pd.DataFrame(list_entries)
    return df


# II Main Functions
def read_fasta(file_path: str,
               col_id: str = "entry",
               col_seq: str = "sequence",
               col_db: Optional[str] = None,
               cols_info: Optional[List[str]] = None,
               sep: str = "|"
               ) -> pd.DataFrame:
    """
    Read an FASTA file into a DataFrame.

    Translation of FASTA file by extracting identifiers and further information from headers
    as well as subsequent sequences.

    Parameters
    ----------
    file_path : str
        Path to the FASTA file.
    col_id : str, default='entry'
        Column name for the sequence identifiers in the resulting DataFrame.
    col_seq : str, default='sequence'
        Column name for the sequences in the resulting DataFrame.
    sep : str, default='|'
        Separator used for splitting identifier and additional information in the FASTA headers.
    cols_info : List[str], optional
        Specifies custom column names for the additional info extracted from headers.
        If not provided, defaults to 'info1', 'info2', etc.
    col_db : str, optional
        Column name for databases. First entry of FASTA header if given.

    Returns
    -------
    pandas.DataFrame
        A DataFrame (``df_seq``) where each row corresponds to a sequence entry from the FASTA file.

    Notes
    -----
    Each ``FASTA`` file entry consists of two parts:

    - **FASTA header**: Starting with '>', the header contains the main id and additional information,
      all separated by a specified separator.
    - **Sequence**: Sequence of specific entry, directly following the header

    ``df_seq`` includes at least these columns:

    - 'entry': Protein identifier, either the UniProt accession number or an id based on index.
    - 'sequence': Amino acid sequence.

    See Also
    --------
    * Further information and examples on FASTA format in
      `BioPerl documentation <https://bioperl.org/formats/sequence_formats/FASTA_sequence_format>`_.
    * Use the FASTA format to create a `BioPython SeqIO object <https://biopython.org/wiki/SeqIO>`_,
      which supports various file formats in computational biology.

    Examples
    --------
    .. include:: examples/read_fasta.rst
    """
    # Check input
    ut.check_file_path(file_path=file_path)
    ut.check_str(name="col_id", val=col_id, accept_none=False)
    ut.check_str(name="col_seq", val=col_seq, accept_none=False)
    ut.check_str(name="col_db", val=col_db, accept_none=True)
    cols_info = ut.check_list_like(name="cols_info", val=cols_info, accept_str=True, accept_none=True)
    ut.check_str(name="sep", val=sep, accept_none=False)
    # Read fasta
    df_seq = _get_entries_from_fasta(file_path, col_id, col_seq, col_db, sep)
    # Adjust column names
    columns = list(df_seq)
    if cols_info is not None:
        n_info = len(columns)-2
        if len(cols_info) >= n_info:
            cols_info = cols_info[0:n_info]
        else:
            cols_info = cols_info + [f"info{i}" for i in range(1, n_info-len(cols_info)+1)]
        if col_db:
            columns = [col_id, col_seq, col_db] + cols_info[0:-1]
        else:
            columns = [col_id, col_seq] + cols_info
        df_seq.columns = columns
    # Post check
    post_check_unique_entries(list_entries=df_seq[col_id].to_list(), col_id=col_id)
    post_check_col_db(df_seq=df_seq, col_db=col_db, sep=sep)
    return df_seq
