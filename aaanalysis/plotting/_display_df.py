"""
This is a script for displaying pd.DataFrames as HTML output for jupyter notebooks.
"""
from typing import Optional
import pandas as pd
from IPython.display import display, HTML

from aaanalysis import utils as ut

# Helper functions
def _adjust_df(df=None, char_limit = 50):
    """"""
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
               char_limit: Optional[int] = None,
               show_shape=False,
               ):
    """Display DataFrame with specific style as HTML output for jupyter notebooks.

    Parameters
    ----------
    df
        DataFrame to be displayed as HTML output.
    fontsize
        Relative font size in points of table font.
    max_width_pct
        Maximum width in percentage of main page for table.
    max_height
       Maximum height in pixels of table.
    char_limit
        Maximum number of characters to display in a cell.
    show_shape
        If ``True``, shape of ``df`` is printed.
    """
    # Check input
    ut.check_df(name="df", df=df, accept_none=False)
    ut.check_number_range(name="fontsize", val=fontsize, min_val=1, accept_none=False, just_int=True)
    ut.check_number_range(name="max_width_pct", val=max_width_pct, min_val=1, max_val=100, accept_none=False, just_int=True)
    ut.check_number_range(name="max_height", val=max_height, min_val=1, accept_none=False, just_int=True)
    ut.check_number_range(name="char_limit", val=char_limit, min_val=1, accept_none=True, just_int=True)
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
    if show_shape:
        print(f"DataFrame shape: {df.shape}")
    display(HTML(styled_df.to_html()))