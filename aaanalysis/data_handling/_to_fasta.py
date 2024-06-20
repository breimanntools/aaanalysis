"""
This is a script for writing df_seq to the FASTA file format. FASTA files are the most commonly used format
in computational biology. This function should enable a smooth interaction with the biopython package.
"""
import pandas as pd
from typing import Optional, List

import aaanalysis.utils as ut
from ._backend.parse_fasta import save_entries_to_fasta


# I Helper Functions


# II Main Functions
def to_fasta(df_seq: pd.DataFrame = None,
             file_path: str = None,
             col_id: str = "entry",
             col_seq: str = "sequence",
             sep: str = "|",
             col_db: Optional[str] = None,
             cols_info: Optional[List[str]] = None,
             ) -> None:
    """
    Write sequence DataFrame to a FASTA file.

    Saving a DataFrame to a FASTA file that includes sequence identifiers, the sequences themselves,
    and additional selected information.

    Parameters
    ----------
    df_seq : pd.DataFrame
        DataFrame containing the identifiers, sequences and additional information.
    file_path : str
        Path where the FASTA file will be saved.
    col_id : str, default='entry'
        Column name in df for the sequence identifiers.
    col_seq : str, default='sequence'
        Column name in df for the sequences.
    sep : str, default='|'
        Separator used to divide different pieces of information in the FASTA header.
    col_db : str, optional
        Column name in df for the database source of the sequence.
    cols_info : list of str, optional
        List of column names for additional information to include in the FASTA header.

    Notes
    -----
    The FASTA header for each sequence is composed as follows:
    >[col_db](optional)|[col_id]|[info1]|...|[infoN]
    followed by the sequence on the next line.

    See Also
    --------
    * :func:`read_fasta`: the respective FASTA reading function.
    * Further information and examples on FASTA format in
      `BioPerl documentation <https://bioperl.org/formats/sequence_formats/FASTA_sequence_format>`_.
    * Use the FASTA format to create a `BioPython SeqIO object <https://biopython.org/wiki/SeqIO>`_,
      which supports various file formats in computational biology.

    Examples
    --------
    .. include:: examples/to_fasta.rst
    """
    # Check input
    ut.check_is_fasta(file_path=file_path)
    ut.check_str(name="col_id", val=col_id, accept_none=False)
    ut.check_str(name="col_seq", val=col_seq, accept_none=False)
    ut.check_str(name="col_db", val=col_db, accept_none=True)
    cols_info = ut.check_list_like(name="cols_info", val=cols_info, accept_str=True, accept_none=True)
    required_columns = [col_id, col_seq] + (cols_info if cols_info is not None else [])
    ut.check_df(df=df_seq, name="df_seq", cols_requiered=required_columns, accept_none=False, accept_nan=False)
    ut.check_str(name="sep", val=sep, accept_none=False)

    # Writing to FASTA
    save_entries_to_fasta(df_seq=df_seq, file_path=file_path,
                          col_id=col_id, col_seq=col_seq,
                          sep=sep,
                          cols_info=cols_info, col_db=col_db)

