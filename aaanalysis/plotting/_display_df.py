"""
This is a script for displaying pd.DataFrames as HTML output for jupyter notebooks.
"""
from typing import Optional
import pandas as pd
from IPython.display import display, HTML

from aaanalysis import utils as ut

# Helper functions
def _adjust_df(df=None, char_limit = 50):
    df = df.copy()
    list_index = df.index
    if sum([type(i) is not int for i in list_index]) == 0:
        df.index = [x+1 for x in df.index]
    if char_limit is not None:
        f = lambda x: str(x)[:int(char_limit/2)] + '...' + str(x)[-int(char_limit/2):]
        df = df.map(lambda x: f(x) if isinstance(x, str) and len(str(x)) > char_limit else x)
    return df


# Main functions
def display_df(df: pd.DataFrame = None,
               fontsize: int = 12,
               max_width_pct: int = 100,
               max_height: int = 300,
               char_limit: int = 25,
               show_shape=False,
               n_rows: Optional[int] = None,
               n_cols: Optional[int] = None,
               n_round=3,
               ):
    """
    Display DataFrame with specific style as HTML output for jupyter notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be displayed as HTML output.
    fontsize : int, default=12
        Relative font size in points of table font.
    max_width_pct: int, default=100
        Maximum width in percentage of main page for table.
    max_height : int, default=300
       Maximum height in pixels of table.
    char_limit : int, default=25
        Maximum number of characters to display in a cell.
    show_shape : bool, default=False
        If ``True``, shape of ``df`` is printed.
    n_rows : int, optional
        Number of rows.
    n_cols : int, optional
        Number of rows.
    n_round : int, default=3
        Rounding to a variable number of decimal places.
    """
    # Check input
    ut.check_df(name="df", df=df, accept_none=False)
    ut.check_number_range(name="fontsize", val=fontsize, min_val=1, accept_none=False, just_int=True)
    ut.check_number_range(name="max_width_pct", val=max_width_pct, min_val=1, max_val=100, accept_none=False, just_int=True)
    ut.check_number_range(name="max_height", val=max_height, min_val=1, accept_none=False, just_int=True)
    ut.check_number_range(name="char_limit", val=char_limit, min_val=1, accept_none=True, just_int=True)
    ut.check_number_range(name="n_rows", val=n_rows, min_val=1, max_val=len(df), accept_none=True, just_int=True)
    ut.check_number_range(name="n_cols", val=n_cols, min_val=1, max_val=len(df.T), accept_none=True, just_int=True)
    # Show shape before filtering
    if show_shape:
        print(f"DataFrame shape: {df.shape}")
    # Filtering
    df = df.copy()
    if n_rows is not None:
        df = df.head(n_rows)
    if n_cols is not None:
        df = df.T.head(n_cols).T
    if n_round is not None:
        df = df.round(n_round)
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
            # Font size for the table (does not work, overriden by CSS files)
            {'selector': 'table', 'props': [('font-size', f'{fontsize}px')]},
        ])
    )
    display(HTML(styled_df.to_html()))