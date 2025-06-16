"""
This is a script for the frontend of the AALogoPlot class.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple, Type, Literal

import aaanalysis.utils as ut
from ._backend._aalogo.aalogo_plot import single_logo_, multi_logo_

# Settings
DICT_LOGO_LABELS = {"probability": "Probability [%]",
                    "weight": "Weight",
                    "counts": "Counts",
                    "information": "Bits"}


# I Helper Functions
def check_parts_len(jmd_n_len=None, jmd_c_len=None, df_logo=None):
    """Check of lengths of parts are matching to """
    len_seq = len(df_logo)
    ut.check_number_range(name="sequence length", val=len_seq, accept_none=False, min_val=1, just_int=True)
    ut.check_number_range(name="jmd_n_len", val=jmd_n_len, accept_none=True, min_val=0, just_int=True)
    ut.check_number_range(name="jmd_c_len", val=jmd_c_len, accept_none=True, min_val=0, just_int=True)
    jmd_c_len = 0 if jmd_c_len is None else jmd_c_len
    jmd_n_len = 0 if jmd_n_len is None else jmd_n_len
    tmd_len = len_seq - (jmd_n_len + jmd_c_len)
    if tmd_len < 1:
        raise ValueError(f"'jmd_n_len' ({jmd_n_len}) + 'jmd_c_len' ({jmd_c_len}) should be "
                         f"< logo length from 'df_logo' ({len_seq})")
    return tmd_len, jmd_n_len, jmd_c_len


# TODO add if target site to be positions (without P or with, e.g., kinases)
# TODO add checks, complete docstring (consistency with cpp_plot functions!), tests for both plots
# II Main Functions
class AALogoPlot:
    """
    UNDER CONSTRUCTION - AALogoPlot class for visualizing sequence logos.

    This class supports single and multiple sequence logo visualizations using different encoding types.
    """

    def __init__(self,
                 logo_type: Literal["probability", "weight", "counts", "information"] = "probability",
                 jmd_n_len: int = 10,
                 jmd_c_len: int = 10,
                 verbose: bool = True
                 ):
        """
        Parameters
        ----------
        logo_type : {'probability', 'weight', 'counts', 'information'}, default='probability'
            Type of sequence logo representation:

            - 'probability': Normalized probability distribution of amino acids.
            - 'weight': Weighted representation.
            - 'counts': Raw counts of amino acids.
            - 'information': Information content in bits.

        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len: int, default=10
            Length of JMD-C (>=0).
        verbose: bool, default=True
            If ``True``, verbose outputs are enabled.

        See Also
        --------
        * `logomaker <https://logomaker.readthedocs.io/en/latest/>`_: Python package for creating sequence logos.
        """
        # Check input
        verbose = ut.check_verbose(verbose)
        list_to_type = ["probability", "weight", "counts", "information"]
        ut.check_str_options(name="to_type", val=logo_type, list_str_options=list_to_type)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        # General settings
        self._verbose = verbose
        # Set type of sequence logo
        self._y_label = DICT_LOGO_LABELS[logo_type]
        # Set consistent length of JMD-N and JMD-C
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len

    def single_logo(self,
                    # Data and Plot Type
                    df_logo: pd.DataFrame = None,
                    df_logo_info: pd.Series = None,
                    info_bar_color: str = "gray",
                    info_bar_ylim: Optional[Tuple[float, float]] = None,
                    target_p1_site: Optional[int] = None,
                    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 4),
                    height_ratio: Tuple[Union[int, float], Union[int, float]] = (1, 6),
                    fontsize_labels: Union[int, float] = None,
                    name_data: Optional[str] = None,
                    name_data_pos: Literal["top", "right", "bottom", "left"] = "top",
                    name_data_color: str = "black",
                    name_data_fontsize: Union[int, float] = None,
                    logo_font_name: str = "Verdana",
                    logo_color_scheme: str = "weblogo_protein",
                    logo_stack_order: Literal["big_on_top", "small_on_top", "fixed"] = "big_on_top",
                    logo_width: float = 0.96,
                    logo_vpad: float = 0.05,
                    logo_vsep: float = 0.0,
                    # Appearance of Parts (TMD-JMD)
                    start: int = 1,
                    tmd_color: str = "mediumspringgreen",
                    jmd_color: str = "blue",
                    fontsize_tmd_jmd: Optional[Union[int, float]] = None,
                    weight_tmd_jmd: Literal['normal', 'bold'] = "normal",
                    highlight_tmd_area: bool = True,
                    highlight_alpha: float = 0.15,
                    xtick_size: Optional[Union[int, float]] = None,
                    xtick_width: Union[int, float] = 2.0,
                    xtick_length: Union[int, float] = 11.0,
                    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a single sequence logo with an optional split view for bit-score information.

        Parameters
        ----------
        df_logo : pd.DataFrame
            DataFrame containing sequence logo data (letter heights per position).
        df_logo_info : pd.Series, optional
            Additional bit-score information to display as a separate subplot.
        info_bar_color : str, default='gray'
            Color of the bit-score bars.
        info_bar_ylim : tuple, optional
            Y-axis limits for bit-score bars. If ``None``, y-axis limits are set automatically.
        target_p1_site : int, optional
            Position marker for a specific target site in the sequence.
        figsize : tuple, default=(8, 3.5)
            Figure dimensions (width, height) in inches.
        height_ratio : tuple, default=(1, 6)
            Ratio of heights between the bit-score subplot and the main logo.
        fontsize_labels : int or float, optional
            Font size for axis labels.
        name_data : str, optional
            Name to annotate the dataset being visualized.
        name_data_pos : {'top', 'right', 'bottom', 'left'}, default='top'
            Position of the dataset name label.
        name_data_color : str, default='black'
            Color of the dataset name label.
        name_data_fontsize : int or float, optional
            Font size of the dataset name label.
        logo_font_name : str, default='Verdana'
            Font used for sequence letters.
        logo_color_scheme : str, default='weblogo_protein'
            Color scheme applied to the amino acid symbols.
        logo_stack_order : {'big_on_top', 'small_on_top', 'fixed'}, default='big_on_top'
            Stacking order of letters in the sequence logo.
        logo_width : float, default=0.96
            Width of the sequence logo plot area (relative scale).
        logo_vpad : float, default=0.05
            Vertical padding between letters.
        logo_vsep : float, default=0.0
            Vertical separation between stacked letters.
        start : int, default=1
            Position label of first residue position (starting at N-terminus).
        tmd_color : str, default='mediumspringgreen'
            Color for TMD.
        jmd_color : str, default='blue'
            Color for JMDs.
        fontsize_tmd_jmd : int or float, optional
            Font size (>=0) for the part labels: 'JMD-N', 'TMD', 'JMD-C'. If ``None``, optimized automatically.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for the part labels: 'JMD-N', 'TMD', 'JMD-C'.
        xtick_size : int or float, optional
            Size of x-tick labels (>0).
        xtick_width : int or float, default=2.0
            Width of the x-ticks (>0).
        xtick_length : int or float, default=11.0
            Length of the x-ticks (>0).
        highlight_tmd_area : bool, default=True
            If ``True``, highlights the TMD area in the plot.
        highlight_alpha : float, default=0.15
            Transparency level for TMD area highlighting.

        Returns
        -------
        fig : plt.Figure
            The figure object containing the sequence logo plot.
        axes : plt.Axes
            The axes object containing the sequence logo plot.

        """
        # Check primary input
        ut.check_figsize(figsize=figsize)
        tmd_len, jmd_n_len, jmd_c_len = check_parts_len(df_logo=df_logo, jmd_n_len=self._jmd_n_len,
                                                        jmd_c_len=self._jmd_c_len)

        # Plot single logo
        fig, axes = single_logo_(df_logo=df_logo,
                                 df_logo_info=df_logo_info,
                                 info_bar_color=info_bar_color,
                                 info_bar_ylim=info_bar_ylim,
                                 target_p1_site=target_p1_site,
                                 figsize=figsize,
                                 height_ratio=height_ratio,
                                 fontsize_labels=fontsize_labels,
                                 y_label=self._y_label,
                                 name_data=name_data,
                                 name_data_pos=name_data_pos,
                                 name_data_color=name_data_color,
                                 name_data_fontsize=name_data_fontsize,
                                 logo_font_name=logo_font_name,
                                 logo_color_scheme=logo_color_scheme,
                                 logo_stack_order=logo_stack_order,
                                 logo_width=logo_width,
                                 logo_vpad=logo_vpad,
                                 logo_vsep=logo_vsep,
                                 start=start,
                                 tmd_len=tmd_len,
                                 jmd_n_len=jmd_n_len,
                                 jmd_c_len=jmd_c_len,
                                 tmd_color=tmd_color,
                                 jmd_color=jmd_color,
                                 xtick_size=xtick_size,
                                 xtick_width=xtick_width,
                                 xtick_length=xtick_length,
                                 fontsize_tmd_jmd=fontsize_tmd_jmd,
                                 weight_tmd_jmd=weight_tmd_jmd,
                                 highlight_tmd_area=highlight_tmd_area,
                                 highlight_alpha=highlight_alpha)
        return fig, axes

    def multi_logo(self,
                   # Data and Plot Type
                   list_df_logo: List[pd.DataFrame] = None,
                   target_p1_site: Optional[int] = None,
                   figsize_per_logo: Tuple[Union[int, float], Union[int, float]] = (8, 3),
                   fontsize_labels: Union[int, float] = None,
                   list_name_data: Optional[List[str]] = None,
                   name_data_pos: Literal["top", "right", "bottom", "left"] = "top",
                   list_name_data_color: Optional[Union[str, List[str]]] = "black",
                   name_data_fontsize: Union[int, float] = None,
                   logo_font_name: str = "Verdana",
                   logo_color_scheme: str = "weblogo_protein",
                   logo_stack_order: Literal["big_on_top", "small_on_top", "fixed"] = "big_on_top",
                   logo_width: float = 0.96,
                   logo_vpad: float = 0.05,
                   logo_vsep: float = 0.0,
                   # Appearance of Parts (TMD-JMD)
                   start: int = 1,
                   tmd_color: str = "mediumspringgreen",
                   jmd_color: str = "blue",
                   fontsize_tmd_jmd: Optional[Union[int, float]] = None,
                   weight_tmd_jmd: Literal['normal', 'bold'] = "normal",
                   highlight_tmd_area: bool = True,
                   highlight_alpha: float = 0.15,
                   xtick_size: Optional[Union[int, float]] = None,
                   xtick_width: Union[int, float] = 2.0,
                   xtick_length: Union[int, float] = 11.0,
                   ):
        """Plot multiple sequence logos for comparison, with adjustable JMD/TMD sizes."""
        # Check primary input
        ut.check_figsize(figsize=figsize_per_logo)
        df_logo1 = list_df_logo[0]
        tmd_len, jmd_n_len, jmd_c_len = check_parts_len(df_logo=df_logo1,
                                                        jmd_n_len=self._jmd_n_len,
                                                        jmd_c_len=self._jmd_c_len)

        # Plot multi logo
        fig, axes = multi_logo_(list_df_logo=list_df_logo,
                                target_p1_site=target_p1_site,
                                figsize_per_logo=figsize_per_logo,
                                y_label=self._y_label,
                                fontsize_labels=fontsize_labels,
                                list_name_data=list_name_data,
                                name_data_pos=name_data_pos,
                                list_name_data_color=list_name_data_color,
                                name_data_fontsize=name_data_fontsize,
                                logo_font_name=logo_font_name,
                                logo_color_scheme=logo_color_scheme,
                                logo_stack_order=logo_stack_order,
                                logo_width=logo_width,
                                logo_vpad=logo_vpad,
                                logo_vsep=logo_vsep,
                                start=start,
                                tmd_len=tmd_len,
                                jmd_n_len=jmd_n_len,
                                jmd_c_len=jmd_c_len,
                                tmd_color=tmd_color,
                                jmd_color=jmd_color,
                                fontsize_tmd_jmd=fontsize_tmd_jmd,
                                weight_tmd_jmd=weight_tmd_jmd,
                                highlight_tmd_area=highlight_tmd_area,
                                highlight_alpha=highlight_alpha,
                                xtick_size=xtick_size,
                                xtick_width=xtick_width,
                                xtick_length=xtick_length)
        return fig, axes

