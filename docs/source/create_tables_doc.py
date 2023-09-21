import os
import pandas as pd
import re
import platform

# Folder and File Constants
SEP = "\\" if platform.system() == "Windows" else "/"
FOLDER_SOURCE = os.path.dirname(os.path.abspath(__file__)) + SEP
FOLDER_IND = FOLDER_SOURCE + "index" + SEP
FOLDER_TABLES = FOLDER_IND + "tables" + SEP

FILE_REF = FOLDER_IND + "references.rst"
FILE_TABLE_TEMPLATE = FOLDER_IND + "tables_template.rst"
FILE_TABLE_SAVED = FOLDER_IND + "tables.rst"
FILE_MAPPER = FOLDER_TABLES + "t0_mapper.xlsx"
LIST_TABLES = list(sorted([x for x in os.listdir(FOLDER_TABLES) if x != "0_mapper.xlsx"]))

COL_MAP_TABLE = "Table"
COL_DESCRIPTION = "Description"
COL_REF = "Reference"

COLUMN_WIDTH = 8
STR_REMOVE = "_XXX" # Check with tables_template.rst for consistency
STR_ADD_TABLE = "ADD-TABLE"


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
    rst_output = ".. list-table::\n   :header-rows: 1\n   :widths: " + " ".join([f"{COLUMN_WIDTH}"] * len(header)) + "\n\n"
    # Include the header
    rst_output += "   * - " + "\n     - ".join(header) + "\n"
    # Include the rows
    for row in rows:
        new_row = []
        for col, val in zip(header, row):
            if col == "Reference":  # Special handling for the 'Reference' column
                new_row.append(f":ref:`{val} <{val}>`")
            else:
                new_row.append(str(val))
        rst_output += "   * - " + "\n     - ".join(new_row) + "\n"
    return rst_output


# Main Functionality
def generate_table_rst():
    # Read the existing references
    with open(FILE_REF, 'r') as f:
        list_refs = f.read()
    list_refs = re.findall(r'\.\. \[([^\]]+)\]', list_refs)

    # Read the existing template
    with open(FILE_TABLE_TEMPLATE, 'r') as f:
        template_lines = f.readlines()

    # Read the mapper file and convert it to reStructuredText format
    df_mapper = pd.read_excel(FILE_MAPPER)
    overview_table_rst = _convert_excel_to_rst(df_mapper)

    # Generate the tables and store them in a dictionary
    tables_dict = {"t0_mapper": overview_table_rst}
    for index, row in df_mapper.iterrows():
        table_name = row[COL_MAP_TABLE]
        df = pd.read_excel(FOLDER_TABLES + _f_xlsx(on=True, file=table_name))
        # Check the references for each table
        table_refs = df[COL_REF].tolist()
        _check_references(table_name=table_name, table_refs=table_refs, list_refs=list_refs)
        table_rst = _convert_excel_to_rst(df)
        tables_dict[table_name] = table_rst

    # Initialize variables
    rst_content = ""
    table_name = ""
    # Loop through the lines of the template
    for line in template_lines:
        # Check for hooks like ".. _1_overview_benchmarks:"
        match = re.search(r'\.\. _(\w+):', line)
        if not match:
            if STR_ADD_TABLE in line and table_name in tables_dict:
                rst_content += "\n" + tables_dict[table_name] + "\n"
            else:
                rst_content += line
        else:
            line_with_new_marker = line.replace(STR_REMOVE, "")
            rst_content += line_with_new_marker
            table_name = match.group(1).replace(STR_REMOVE, "")

    # Write the new content to the output .rst file
    with open(FILE_TABLE_SAVED, 'w') as f:
        f.write(rst_content)
