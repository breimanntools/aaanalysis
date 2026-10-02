"""
This is a script for the backend of the standardized table export (to_table). It assembles
the JSON metadata sidecar from the shipped DataFrame schemas and writes both artifacts.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import aaanalysis.utils as ut


# I Helper Functions
def _field_from_columns(spec=None, col=None):
    """Return the schema field record of ``col`` for a column-table frame, else None."""
    dict_cols = spec.get("columns")
    if dict_cols is None:
        return None
    return dict_cols.get(col)


def _field_from_df_feat(col=None):
    """Return a field record from the simple df_feat contract (covers optional columns)."""
    record = ut.DICT_DF_FEAT.get(col)
    if record is None:
        return None
    dtype, required, nullable, description = record
    return {"dtype": dtype, "required": required, "nullable": nullable,
            "unique": False, "description": description}


def _field_from_dynamic(spec=None):
    """Return the shared field record of a frame whose columns are dynamic (df_parts)."""
    dynamic = spec.get("dynamic_columns")
    if dynamic is None:
        return None
    return {"dtype": dynamic["dtype"], "required": False, "nullable": dynamic["nullable"],
            "unique": False, "description": dynamic["description"]}


def _field_from_matrix(spec=None):
    """Return the shared field record of a matrix/array frame (X, df_scales, df_logo)."""
    matrix = spec.get("matrix")
    if matrix is None:
        return None
    description = (f"{matrix['values'][0].upper()}{matrix['values'][1:]} "
                   f"(one column per {matrix['columns']}).")
    return {"dtype": matrix["dtype"], "required": False, "nullable": False,
            "unique": False, "description": description}


def _field_undocumented(name=None):
    """Return the fallback record for a column the named schema does not cover."""
    return {"dtype": "unknown", "required": False, "nullable": True, "unique": False,
            "description": (f"Column outside the '{name}' schema contract; its meaning is "
                            f"defined by whatever produced it, not by AAanalysis.")}


def build_column_records_(df=None, name=None) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``({column: field record}, undocumented columns)`` for every column of ``df``.

    Every record carries a non-empty ``description``, so an exported table is readable
    without the package. The schema record is looked up in this order: the named frame's
    column table, the simple df_feat contract (which also names the optional / post-hoc
    columns), the frame's dynamic-column or matrix record, and finally a fallback that
    states the column is outside the contract.
    """
    spec = ut.DICT_DF_SCHEMAS[name]
    dict_columns = {}
    cols_undocumented = []
    for col in df.columns:
        record = _field_from_columns(spec=spec, col=col)
        if record is None and name == "df_feat":
            record = _field_from_df_feat(col=col)
        if record is None:
            record = _field_from_dynamic(spec=spec)
        if record is None:
            record = _field_from_matrix(spec=spec)
        if record is None:
            record = _field_undocumented(name=name)
            cols_undocumented.append(str(col))
        # 'dtype' is the contract's human dtype; 'dtype_pandas' is what was written out
        # and what pd.read_csv(..., dtype=...) restores.
        record = dict(record)
        record["dtype_pandas"] = str(df[col].dtype)
        dict_columns[str(col)] = record
    return dict_columns, cols_undocumented


def build_table_metadata_(df=None, name=None, file_name=None, sep=None,
                          dict_provenance=None) -> Dict[str, Any]:
    """Assemble the JSON-serializable sidecar record of an exported table."""
    spec = ut.DICT_DF_SCHEMAS[name]
    dict_columns, cols_undocumented = build_column_records_(df=df, name=name)
    dict_metadata = {"schema_version": ut.STR_SCHEMA_VERSION,
                     "name": name,
                     "description": spec["description"],
                     "file_name": file_name,
                     "file_name_metadata": os.path.splitext(file_name)[0] + ut.STR_SUFFIX_META,
                     "sep": sep,
                     "n_rows": int(len(df)),
                     "n_cols": int(len(df.columns)),
                     "columns": dict_columns,
                     "columns_undocumented": cols_undocumented,
                     "categories": list(ut.LIST_CAT),
                     "parts": list(ut.LIST_ALL_PARTS),
                     "provenance": dict_provenance}
    return dict_metadata


def get_metadata_path_(file_path=None) -> str:
    """Return the sidecar path written next to ``file_path`` ('<stem>.meta.json')."""
    return os.path.splitext(file_path)[0] + ut.STR_SUFFIX_META


# II Main Functions
def save_table_with_metadata(df=None, file_path=None, name=None, sep=None,
                             dict_provenance=None) -> Dict[str, Any]:
    """Write the delimited table and its JSON metadata sidecar; return the sidecar record."""
    dict_metadata = build_table_metadata_(df=df, name=name, sep=sep,
                                          file_name=os.path.basename(file_path),
                                          dict_provenance=dict_provenance)
    df.to_csv(file_path, sep=sep, index=False)
    with open(get_metadata_path_(file_path=file_path), "w", encoding="utf-8") as file_meta:
        json.dump(dict_metadata, file_meta, indent=2, ensure_ascii=False)
        file_meta.write("\n")
    return dict_metadata
