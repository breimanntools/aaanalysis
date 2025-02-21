"""
This is a script for displaying pd.DataFrames as HTML output for jupyter notebooks.
"""
from typing import Optional, Union
import pandas as pd
from IPython.display import display, HTML

from aaanalysis import utils as ut


# Helper functions
def _adjust_df(df=None, char_limit = 50):
    df = df.copy()
    list_index = df.index
    # Adjust index if it consists solely of integers
    if all(isinstance(i, int) for i in list_index):
        df.index = [x + 1 for x in df.index]

    # Function to truncate strings longer than char_limit
    def truncate_string(s):
        return str(s)[:int(char_limit/2)] + '...' + str(s)[-int(char_limit/2):] if len(str(s)) > char_limit else s

    # Apply truncation to each cell in the DataFrame
    if char_limit is not None:
        for col in df.columns:
            df[col] = df[col].apply(lambda x: truncate_string(x) if isinstance(x, str) else x)
    return df


def _check_show(name="row_to_show", val=None, df=None):
    """Check if valid string or int"""
    if val is None:
        return None # Skip test
    rows_or_columns = list(df.T) if "row" in name else list(df)
    n = len(rows_or_columns)
    str_row_or_column = "row" if "row" in name else "column"
    if isinstance(val, str):
        ut.check_str(name=name, val=val, accept_none=True)
        if val not in rows_or_columns:
            raise ValueError(f"'{name}' ('{val}') should be one of: {rows_or_columns}")
    elif isinstance(val, int):
        ut.check_number_range(name=name, val=val, accept_none=True, min_val=0, max_val=n, just_int=True)
    else:
        raise ValueError(f"'{name}' ('{val}') should be int (<{n}) or one of following {str_row_or_column} names: {rows_or_columns}")


def _select_row(df=None, row_to_show=None):
    """Select row"""
    if row_to_show is not None:
        if isinstance(row_to_show, int):
            df = df.iloc[[row_to_show]]
        elif isinstance(row_to_show, str):
            df = df.loc[[row_to_show]]
    return df


def _select_col(df=None, col_to_show=None):
    """Select column"""
    if col_to_show is not None:
        if isinstance(col_to_show, int):
            df = df.iloc[:, [col_to_show]]
        elif isinstance(col_to_show, str):
            df = df[[col_to_show]]
    return df


# Main functions
def display_df(df: pd.DataFrame = None,
               max_width_pct: int = 100,
               max_height: int = 300,
               char_limit: int = 30,
               show_shape=False,
               n_rows: Optional[int] = None,
               n_cols: Optional[int] = None,
               row_to_show: Optional[Union[int, str]] = None,
               col_to_show: Optional[Union[int, str]] = None,
               ):
    """
    Display DataFrame with specific style as HTML output for jupyter notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be displayed as HTML output.
    max_width_pct: int, default=100
        Maximum width in percentage of main page for table.
    max_height : int, default=300
       Maximum height in pixels of table.
    char_limit : int, default=30
        Maximum number of characters to display in a cell.
    show_shape : bool, default=False
        If ``True``, shape of ``df`` is printed.
    n_rows : int, optional
        Display only the first n rows. If negative, last n rows will be shown.
    n_cols : int, optional
        Display only the first n columns. If negative, last n columns will be shown.
    row_to_show : int or str, optional
        Display only the specified row.
    col_to_show : int or str, optional
        Display only the specified column.

    Examples
    --------
    .. include:: examples/display_df.rst
    """
    # Check input
    ut.check_df(name="df", df=df, accept_none=False)
    ut.check_number_range(name="max_width_pct", val=max_width_pct, min_val=1, max_val=100, accept_none=False, just_int=True)
    ut.check_number_range(name="max_height", val=max_height, min_val=1, accept_none=False, just_int=True)
    ut.check_number_range(name="char_limit", val=char_limit, min_val=1, accept_none=True, just_int=True)
    n_rows_, n_cols_ = len(df), len(df.T)
    ut.check_number_range(name="n_rows", val=n_rows, min_val=-n_rows_, max_val=n_rows_, accept_none=True, just_int=True)
    ut.check_number_range(name="n_cols", val=n_cols, min_val=-n_cols_, max_val=n_cols_, accept_none=True, just_int=True)
    _check_show(name="show_only_col", val=col_to_show, df=df)
    _check_show(name="show_only_row", val=row_to_show, df=df)
    # Show shape before filtering
    if show_shape:
        print(f"DataFrame shape: {df.shape}")
    # Filtering
    df = df.copy()
    df = _select_col(df=df, col_to_show=col_to_show)
    df = _select_row(df=df, row_to_show=row_to_show)
    if row_to_show is None and n_rows is not None:
        if n_rows > 0:
            df = df.head(n_rows)
        else:
            df = df.tail(abs(n_rows))
    if col_to_show is None and n_cols is not None:
        if n_cols > 0:
            df = df.T.head(n_cols).T
        else:
            df = df.T.tail(abs(n_cols)).T
    # Style dataframe
    df = _adjust_df(df=df, char_limit=char_limit)
    styled_df = (
        df.style
        .set_table_attributes(f"style='display:block; max-height: {max_height}px; max-width: {max_width_pct}%; overflow-x: auto; overflow-y: auto;'")
        .set_table_styles([
            # Explicitly set background and text color for headers
            {'selector': 'thead th', 'props': [('background-color', 'white'), ('color', 'black')]},
            # Style for odd and even rows
            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f2f2f2')]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', 'white')]},
            # General styling for table cells
            {'selector': 'th, td', 'props': [('padding', '5px'), ('white-space', 'nowrap')]},
        ])
    )
    display(HTML(styled_df.to_html()))
