"""
This is a script for writing an AAanalysis output table (df_feat, df_seq, the feature
matrix, ...) to a delimited text file together with a JSON metadata sidecar, so the
exported table stays interpretable outside of Python.
"""
import os
from typing import Optional, Dict, Any

import pandas as pd

import aaanalysis.utils as ut
from .._provenance import get_provenance
from ._backend.export_table import save_table_with_metadata


# I Helper Functions
def check_match_file_path_sep(file_path: str, sep: Optional[str]) -> str:
    """Check the file extension is a supported table format; return the separator to use."""
    file_ext = os.path.splitext(file_path)[1].lower()
    list_ext = list(ut.DICT_TABLE_SEP)
    if file_ext not in list_ext:
        raise ValueError(f"'file_path' ('{file_path}') should end with one of {list_ext}, "
                         f"but ends with '{file_ext}'.")
    list_sep_valid = ut.DICT_TABLE_SEP[file_ext]
    if sep is None:
        return list_sep_valid[0]
    if sep not in list_sep_valid:
        raise ValueError(f"'sep' ({sep!r}) is not valid for a '{file_ext}' file, which "
                         f"accepts {list_sep_valid}.")
    return sep


def check_match_df_name(df: pd.DataFrame, name: str) -> None:
    """Check that 'df' carries the columns the schema named by 'name' requires."""
    dict_cols = ut.DICT_DF_SCHEMAS[name].get("columns")
    if dict_cols is None:
        return
    cols_required = [c for c, rec in dict_cols.items() if rec["required"]]
    cols_missing = [c for c in cols_required if c not in df.columns]
    if len(cols_missing) > 0:
        raise ValueError(f"'df' does not match the '{name}' schema: the required columns "
                         f"{cols_missing} are missing. Set 'name' to the schema that "
                         f"describes 'df' (one of {list(ut.DICT_DF_SCHEMAS)}).")


def check_folder_of_file_path(file_path: str) -> None:
    """Check that the directory the table is written into exists."""
    folder_path = os.path.dirname(file_path)
    if folder_path != "" and not os.path.isdir(folder_path):
        raise ValueError(f"The directory of 'file_path' ('{folder_path}') does not exist. "
                         f"Create it before exporting.")


# II Main Functions
def to_table(df: pd.DataFrame,
             file_path: str,
             name: str = "df_feat",
             sep: Optional[str] = None,
             random_state: Optional[int] = None,
             ) -> Dict[str, Any]:
    """
    Write an AAanalysis output table to a delimited text file with a metadata sidecar.

    Two artifacts are written: the table itself (``.csv`` or ``.tsv``, without the row
    index) and a JSON sidecar named after it (``<stem>.meta.json``). The sidecar
    documents every exported column with the meaning, dtype, and allowed values from
    the shipped data schemas, and records the AAanalysis version and input fingerprint,
    so a table can still be read correctly long after the session that produced it.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    df : pd.DataFrame
        Table to export, such as the CPP feature table (``df_feat``), a sequence table
        (``df_seq``), or the feature matrix ``X`` wrapped in a DataFrame. Its required
        columns must match the schema selected by ``name``.
    file_path : str
        Path of the table file to write. Its extension selects the format and must be
        ``.csv`` or ``.tsv``; the directory must already exist.
    name : str, default='df_feat'
        Name of the data schema describing ``df``, used to document the exported
        columns. One of the keys of the shipped schemas (see :ref:`df_schemas`).
    sep : str, optional
        Column separator. ``None`` derives it from the file extension (``','`` for
        ``.csv``, a tab for ``.tsv``). A ``.csv`` file also accepts ``';'``.
    random_state : int, optional
        Seed under which ``df`` was computed, recorded in the provenance part of the
        sidecar. ``None`` means no seed was in effect.

    Returns
    -------
    dict_metadata : dict
        The JSON-serializable sidecar record, with the following keys:

        - ``schema_version``: version of the sidecar layout itself.
        - ``name``, ``description``: the schema ``df`` was documented against.
        - ``file_name``, ``file_name_metadata``, ``sep``, ``n_rows``, ``n_cols``:
          how the table was written.
        - ``columns``: per column, its meaning, contract dtype, written pandas dtype
          (``dtype_pandas``), and where defined its allowed values or range.
        - ``columns_undocumented``: columns outside the schema contract (empty for a
          table that matches its schema).
        - ``categories``, ``parts``: the AAontology category and sequence part
          vocabularies the table's values are drawn from.
        - ``provenance``: the record of :func:`get_provenance`, including the package
          version and a fingerprint of ``df``.

    Notes
    -----
    * The row index is not written, so the exported table has no unnamed first column.
      Rows keep their order; compare a filtered frame after ``reset_index(drop=True)``.
    * Load the table back with pandas. Plain inference already restores the AAanalysis
      output tables exactly::

          df_feat = pd.read_csv("df_feat.csv")

      To restore the written dtypes regardless of inference, pass them from the
      sidecar::

          dict_metadata = json.load(open("df_feat.meta.json"))
          dtypes = {c: r["dtype_pandas"] for c, r in dict_metadata["columns"].items()}
          df_feat = pd.read_csv("df_feat.csv", dtype=dtypes)

    * Only delimited text is written. Columns holding non-scalar objects (lists,
      arrays, nested frames) are stringified by pandas and do not round-trip.

    See Also
    --------
    * :func:`to_fasta`: the respective writing function for sequence data.
    * :func:`get_provenance`: the provenance record embedded in the sidecar.
    * :ref:`df_schemas`: the column contracts the sidecar serializes.

    Examples
    --------
    .. include:: examples/to_table.rst
    """
    # Check input
    ut.check_df(df=df, name="df", accept_none=False)
    ut.check_str(name="file_path", val=file_path, accept_none=False)
    ut.check_str_options(name="name", val=name, list_str_options=list(ut.DICT_DF_SCHEMAS))
    ut.check_str(name="sep", val=sep, accept_none=True)
    ut.check_number_range(name="random_state", val=random_state, min_val=0,
                          just_int=True, accept_none=True)
    sep = check_match_file_path_sep(file_path=file_path, sep=sep)
    check_folder_of_file_path(file_path=file_path)
    check_match_df_name(df=df, name=name)

    # Export table and metadata sidecar
    dict_provenance = get_provenance(random_state=random_state, data=df)
    dict_metadata = save_table_with_metadata(df=df, file_path=file_path, name=name,
                                             sep=sep, dict_provenance=dict_provenance)
    return dict_metadata
