import os
import pandas as pd
import re
import platform

# Folder and File Constants
SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_SOURCE = os.path.dirname(os.path.abspath(__file__))
FOLDER_REF = FOLDER_SOURCE + SEP + "index" + SEP
FOLDER_RESOURCES = FOLDER_SOURCE + SEP + "_resources" + SEP
FOLDER_TABLES = FOLDER_RESOURCES + "tables" + SEP

FILE_REF = FOLDER_REF + "references.rst"
FILE_TABLE = FOLDER_RESOURCES + "tables.rst"
FILE_MAPPER = FOLDER_TABLES + "0_mapper.xlsx"
LIST_TABLES = list(sorted([x for x in os.listdir(FOLDER_TABLES) if x != "0_mapper.xlsx"]))

COL_MAP_TABLE = "Table"
COL_DESCRIPTION = "Description"
COL_REF = "Reference"


# Helper Functions
def _f_xlsx(on=True, file=None, ending=".xlsx"):
    """"""
    if on:
        if ending not in file:
            return file.split(".")[0] + ending
        else:
            return file
    else:
        if ending not in file:
            return file
        else:
            return file.split(".")[0]


def _check_references(table_name=None, table_refs=None, list_refs=None):
    missing_references = [ref for ref in table_refs if ref not in list_refs]
    if len(missing_references) > 0:
        raise ValueError(f"The following references are missing from '{table_name}': {missing_references}")


def _check_tables(list_tables):
    """"""
    f = lambda x: _f_xlsx(on=False, file=x)
    if list_tables != LIST_TABLES:
        list_missing_map = [f(x) for x in list_tables if x not in list_tables]
        list_missing_tables = [f(x) for x in LIST_TABLES if x not in list_tables]
        if len(list_missing_map) > 0:
            raise ValueError(f"Following tables miss in 0_mapper.xlsx: {list_missing_map}")
        if len(list_missing_tables) > 0:
            raise ValueError(f"Following tables miss in tables folder: {list_missing_tables}")


def _convert_excel_to_rst(df):
    header = df.columns.tolist()
    rows = df.values.tolist()
    columns = [header] + rows
    rst_output = ".. list-table::\n   :header-rows: 1\n   :widths: " + " ".join(["10"] * len(header)) + "\n\n"
    for row in columns:
        rst_output += "   * - " + "\n     - ".join(map(str, row)) + "\n"

    return rst_output


# Main Functionality
def generate_table_rst():
    with open(FILE_REF, 'r') as f:
        list_refs = f.read()
    list_refs = re.findall(r'\.\. \[([^\]]+)\]', list_refs)
    df_mapper = pd.read_excel(FILE_MAPPER)
    list_tables = [_f_xlsx(on=True, file=x) for x in sorted(df_mapper[COL_MAP_TABLE])]
    _check_tables(list_tables)
    rst_content = _convert_excel_to_rst(df_mapper)
    rst_content = f"Tables for the Project\n======================\n\n.. contents::\n    :local:\n    :depth: 1\n\nOverview of Tables\n------------------\n{rst_content}"
    for index, row in df_mapper.iterrows():
        table_name = row[COL_MAP_TABLE]
        description = row[COL_DESCRIPTION]
        df = pd.read_excel(FOLDER_TABLES + _f_xlsx(on=True, file=table_name))
        table_refs = df[COL_REF].tolist()
        _check_references(table_name=table_name, table_refs=table_refs, list_refs=list_refs)
        rst_content += f"\n{description}\n{'-' * len(description)}\n"
        rst_content += _convert_excel_to_rst(df)

    with open(FILE_TABLE, 'w') as f:
        f.write(rst_content)
