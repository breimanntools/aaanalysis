"""
This is a script for the frontend of the AAlogoPlot class.
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, List, Tuple, Literal

import aaanalysis.utils as ut
from ._backend._aalogo.aalogo_plot import single_logo_, multi_logo_

# Settings
DICT_LOGO_LABELS = {"probability": "Probability [%]",
                    "weight": "Weight",
                    "counts": "Counts",
                    "information": "Bits"}


# I Helper Functions
def check_df_logo(df_logo=None):
    """Check that df_logo is a valid non-empty DataFrame with numeric values."""
    ut.check_df(name="df_logo", df=df_logo, accept_none=False, accept_nan=False)
    if len(df_logo) == 0:
        raise ValueError("'df_logo' must not be empty.")


def check_list_df_logo(list_df_logo=None):
    """Check that list_df_logo is a non-empty list of valid logo DataFrames."""
    if not isinstance(list_df_logo, list) or len(list_df_logo) == 0:
        raise ValueError("'list_df_logo' must be a non-empty list of DataFrames.")
    for i, df in enumerate(list_df_logo):
        ut.check_df(name=f"list_df_logo[{i}]", df=df, accept_none=False, accept_nan=False)
    n_pos = len(list_df_logo[0])
    for i, df in enumerate(list_df_logo[1:], start=1):
        if len(df) != n_pos:
            raise ValueError(
                f"All DataFrames in 'list_df_logo' must have the same number of positions. "
                f"list_df_logo[0] has {n_pos}, list_df_logo[{i}] has {len(df)}.")


def check_parts_len(jmd_n_len=None, jmd_c_len=None, df_logo=None):
    """Validate jmd_n_len and jmd_c_len against df_logo length and return derived tmd_len."""
    len_seq = len(df_logo)
    ut.check_number_range(name="sequence length", val=len_seq, accept_none=False, min_val=1, just_int=True)
    ut.check_number_range(name="jmd_n_len", val=jmd_n_len, accept_none=True, min_val=0, just_int=True)
    ut.check_number_range(name="jmd_c_len", val=jmd_c_len, accept_none=True, min_val=0, just_int=True)
    jmd_n_len = 0 if jmd_n_len is None else jmd_n_len
    jmd_c_len = 0 if jmd_c_len is None else jmd_c_len
    tmd_len = len_seq - (jmd_n_len + jmd_c_len)
    if tmd_len < 1:
        raise ValueError(
            f"'jmd_n_len' ({jmd_n_len}) + 'jmd_c_len' ({jmd_c_len}) must be "
            f"< logo length from 'df_logo' ({len_seq}).")
    return tmd_len, jmd_n_len, jmd_c_len


def check_df_logo_info(df_logo_info=None, df_logo=None):
    """Check that df_logo_info is a valid Series with the same length as df_logo."""
    ut.check_df(name="df_logo_info", df=df_logo_info, check_series=True,
                accept_none=False, accept_nan=False)
    if df_logo is not None and len(df_logo_info) != len(df_logo):
        raise ValueError(
            f"'df_logo_info' length ({len(df_logo_info)}) must match "
            f"'df_logo' length ({len(df_logo)}).")


def check_info_bar_ylim(info_bar_ylim=None):
    """Check that info_bar_ylim is None or a valid (min, max) tuple of numbers."""
    if info_bar_ylim is None:
        return
    if (not isinstance(info_bar_ylim, tuple) or len(info_bar_ylim) != 2
            or not all(isinstance(v, (int, float)) for v in info_bar_ylim)):
        raise ValueError("'info_bar_ylim' must be a tuple of two numbers (min, max).")
    if info_bar_ylim[0] >= info_bar_ylim[1]:
        raise ValueError(
            f"'info_bar_ylim[0]' ({info_bar_ylim[0]}) must be < 'info_bar_ylim[1]' ({info_bar_ylim[1]}).")


def check_height_ratio(height_ratio=None):
    """Check that height_ratio is a tuple of two positive numbers."""
    if (not isinstance(height_ratio, tuple) or len(height_ratio) != 2
            or not all(isinstance(v, (int, float)) and v > 0 for v in height_ratio)):
        raise ValueError("'height_ratio' must be a tuple of two positive numbers.")


def check_list_name_data(list_name_data=None, list_df_logo=None):
    """Check that list_name_data matches the number of logos if provided."""
    if list_name_data is None:
        return
    if not isinstance(list_name_data, list):
        raise ValueError("'list_name_data' must be a list of strings or None.")
    if list_df_logo is not None and len(list_name_data) != len(list_df_logo):
        raise ValueError(
            f"'list_name_data' length ({len(list_name_data)}) must match "
            f"'list_df_logo' length ({len(list_df_logo)}).")


def check_list_name_data_color(list_name_data_color=None, list_df_logo=None):
    """Check that list_name_data_color is a valid string or matching list of strings."""
    if isinstance(list_name_data_color, str):
        return
    if isinstance(list_name_data_color, list):
        if list_df_logo is not None and len(list_name_data_color) != len(list_df_logo):
            raise ValueError(
                f"'list_name_data_color' length ({len(list_name_data_color)}) must match "
                f"'list_df_logo' length ({len(list_df_logo)}).")
        return
    raise ValueError("'list_name_data_color' must be a string or list of strings.")


# II Main Functions
class AAlogoPlot:
    """
    Amino Acid Logo Plot (**AAlogoPlot**) class for visualizing sequence logos.

    Supports single and stacked multiple sequence logo visualizations with automatic
    TMD/JMD part annotations. The ``logo_type`` set at initialization controls only
    the y-axis label; the logo data itself is provided as a pre-computed ``df_logo``
    from :class:`AAlogo`.
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
            Type of sequence logo encoding, used to set the y-axis label:

            - ``'probability'``: y-axis label is ``'Probability [%]'``.
            - ``'weight'``: y-axis label is ``'Weight'``.
            - ``'counts'``: y-axis label is ``'Counts'``.
            - ``'information'``: y-axis label is ``'Bits'``.

        jmd_n_len : int, default=10
            Length of the JMD-N region (>=0). Used together with ``jmd_c_len`` to derive
            the TMD length from the logo DataFrame for part annotations.
        jmd_c_len : int, default=10
            Length of the JMD-C region (>=0).
        verbose : bool, default=True
            If ``True``, verbose outputs are enabled.

        See Also
        --------
        * :class:`AAlogo`: the corresponding data computation class.
        * `logomaker <https://logomaker.readthedocs.io/en/latest/>`_: the underlying logo rendering package.
        """
        # Check input
        verbose = ut.check_verbose(verbose)
        list_logo_types = ["probability", "weight", "counts", "information"]
        ut.check_str_options(name="logo_type", val=logo_type, list_str_options=list_logo_types)
        jmd_n_len = ut.check_jmd_n_len(jmd_n_len=jmd_n_len)
        jmd_c_len = ut.check_jmd_c_len(jmd_c_len=jmd_c_len)
        # Set attributes
        self._verbose = verbose
        self._y_label = DICT_LOGO_LABELS[logo_type]
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len

    def single_logo(self,
                    df_logo: pd.DataFrame = None,
                    df_logo_info: Optional[pd.Series] = None,
                    info_bar_color: str = "gray",
                    info_bar_ylim: Optional[Tuple[float, float]] = None,
                    target_p1_site: Optional[int] = None,
                    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 4),
                    height_ratio: Tuple[Union[int, float], Union[int, float]] = (1, 6),
                    fontsize_labels: Optional[Union[int, float]] = None,
                    name_data: Optional[str] = None,
                    name_data_pos: Literal["top", "right", "bottom", "left"] = "top",
                    name_data_color: str = "black",
                    name_data_fontsize: Optional[Union[int, float]] = None,
                    logo_font_name: str = "Verdana",
                    logo_color_scheme: str = "weblogo_protein",
                    logo_stack_order: Literal["big_on_top", "small_on_top", "fixed"] = "big_on_top",
                    logo_width: float = 0.96,
                    logo_vpad: float = 0.05,
                    logo_vsep: float = 0.0,
                    start: int = 1,
                    tmd_color: str = "mediumspringgreen",
                    jmd_color: str = "blue",
                    fontsize_tmd_jmd: Optional[Union[int, float]] = None,
                    weight_tmd_jmd: Literal["normal", "bold"] = "normal",
                    highlight_tmd_area: bool = True,
                    highlight_alpha: float = 0.15,
                    xtick_size: Optional[Union[int, float]] = None,
                    xtick_width: Union[int, float] = 2.0,
                    xtick_length: Union[int, float] = 11.0,
                    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single sequence logo with optional bit-score bar and TMD/JMD annotations.

        Parameters
        ----------
        df_logo : pd.DataFrame, shape (n_positions, n_amino_acids)
            Logo matrix as returned by :meth:`AAlogo.get_df_logo`. Rows are residue
            positions, columns are amino acids.
        df_logo_info : pd.Series, shape (n_positions,), optional
            Per-position information content as returned by :meth:`AAlogo.get_df_logo_info`.
            If provided, a bit-score bar is rendered above the main logo.
        info_bar_color : str, default='gray'
            Color of the bit-score bars in the optional top panel.
        info_bar_ylim : tuple of float, optional
            Y-axis limits ``(min, max)`` for the bit-score bar. If ``None``, set automatically.
        target_p1_site : int, optional
            If set, replaces the standard JMD/TMD x-axis with P-site notation
            (P1, P2, ..., P1', P2', ...) anchored at this position index.
        figsize : tuple of (int or float), default=(8, 4)
            Figure size ``(width, height)`` in inches.
        height_ratio : tuple of (int or float), default=(1, 6)
            Height ratio ``(info_bar, logo)`` when ``df_logo_info`` is provided.
        fontsize_labels : int or float, optional
            Font size for axis labels. If ``None``, uses the package default.
        name_data : str, optional
            Dataset name to annotate on the plot.
        name_data_pos : {'top', 'right', 'bottom', 'left'}, default='top'
            Position of the ``name_data`` annotation.
        name_data_color : str, default='black'
            Color of the ``name_data`` annotation.
        name_data_fontsize : int or float, optional
            Font size of the ``name_data`` annotation.
        logo_font_name : str, default='Verdana'
            Font name for amino acid letter rendering.
        logo_color_scheme : str, default='weblogo_protein'
            Color scheme for amino acid letters (passed to ``logomaker``).
        logo_stack_order : {'big_on_top', 'small_on_top', 'fixed'}, default='big_on_top'
            Stacking order of letters within each position column.
        logo_width : float, default=0.96
            Relative width of each letter column (0 to 1).
        logo_vpad : float, default=0.05
            Vertical padding between stacked letters.
        logo_vsep : float, default=0.0
            Vertical separation between stacked letters.
        start : int, default=1
            Residue number assigned to the first position of JMD-N.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD x-tick marks, part bar, and highlight.
        jmd_color : str, default='blue'
            Color for JMD x-tick marks and part bar.
        fontsize_tmd_jmd : int or float, optional
            Font size for the 'JMD-N', 'TMD', 'JMD-C' part labels. If ``None``, auto-sized.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for the 'JMD-N', 'TMD', 'JMD-C' part labels.
        highlight_tmd_area : bool, default=True
            If ``True``, shades the TMD region with a semi-transparent rectangle.
        highlight_alpha : float, default=0.15
            Opacity of the TMD highlight (0 = fully transparent, 1 = opaque).
        xtick_size : int or float, optional
            Font size for x-tick labels. If ``None``, uses the package default.
        xtick_width : int or float, default=2.0
            Line width of x-tick marks.
        xtick_length : int or float, default=11.0
            Length of x-tick marks in points.

        Returns
        -------
        fig : plt.Figure
            Figure object.
        axes : plt.Axes or tuple of plt.Axes
            Single ``Axes`` when ``df_logo_info`` is ``None``; tuple ``(ax_logo, ax_info)``
            when ``df_logo_info`` is provided.

        See Also
        --------
        * :meth:`AAlogoPlot.multi_logo`: for stacked multi-group comparison.
        * :class:`AAlogo`: to compute ``df_logo`` and ``df_logo_info``.

        Examples
        --------
        .. include:: examples/AAlogoplot_single_logo.rst
        """
        # Check input
        check_df_logo(df_logo=df_logo)
        if df_logo_info is not None:
            check_df_logo_info(df_logo_info=df_logo_info, df_logo=df_logo)
            check_info_bar_ylim(info_bar_ylim=info_bar_ylim)
            check_height_ratio(height_ratio=height_ratio)
        ut.check_figsize(figsize=figsize)
        ut.check_str_options(name="name_data_pos", val=name_data_pos,
                             list_str_options=["top", "right", "bottom", "left"])
        ut.check_str_options(name="logo_stack_order", val=logo_stack_order,
                             list_str_options=["big_on_top", "small_on_top", "fixed"])
        ut.check_str_options(name="weight_tmd_jmd", val=weight_tmd_jmd,
                             list_str_options=["normal", "bold"])
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_number_range(name="highlight_alpha", val=highlight_alpha,
                              min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="logo_width", val=logo_width,
                              min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="xtick_width", val=xtick_width, min_val=0, just_int=False)
        ut.check_number_range(name="xtick_length", val=xtick_length, min_val=0, just_int=False)
        tmd_len, jmd_n_len, jmd_c_len = check_parts_len(df_logo=df_logo,
                                                         jmd_n_len=self._jmd_n_len,
                                                         jmd_c_len=self._jmd_c_len)
        # Plot
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
                   list_df_logo: List[pd.DataFrame] = None,
                   target_p1_site: Optional[int] = None,
                   figsize_per_logo: Tuple[Union[int, float], Union[int, float]] = (8, 3),
                   fontsize_labels: Optional[Union[int, float]] = None,
                   list_name_data: Optional[List[str]] = None,
                   name_data_pos: Literal["top", "right", "bottom", "left"] = "top",
                   list_name_data_color: Optional[Union[str, List[str]]] = "black",
                   name_data_fontsize: Optional[Union[int, float]] = None,
                   logo_font_name: str = "Verdana",
                   logo_color_scheme: str = "weblogo_protein",
                   logo_stack_order: Literal["big_on_top", "small_on_top", "fixed"] = "big_on_top",
                   logo_width: float = 0.96,
                   logo_vpad: float = 0.05,
                   logo_vsep: float = 0.0,
                   start: int = 1,
                   tmd_color: str = "mediumspringgreen",
                   jmd_color: str = "blue",
                   fontsize_tmd_jmd: Optional[Union[int, float]] = None,
                   weight_tmd_jmd: Literal["normal", "bold"] = "normal",
                   highlight_tmd_area: bool = True,
                   highlight_alpha: float = 0.15,
                   xtick_size: Optional[Union[int, float]] = None,
                   xtick_width: Union[int, float] = 2.0,
                   xtick_length: Union[int, float] = 11.0,
                   ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot multiple sequence logos stacked vertically for group comparison.

        All logos share the same y-axis scale. TMD/JMD annotations are shown only
        on the bottom subplot to avoid repetition.

        Parameters
        ----------
        list_df_logo : list of pd.DataFrame, each shape (n_positions, n_amino_acids)
            List of logo matrices, one per group. All must have the same number of positions.
        target_p1_site : int, optional
            If set, replaces the standard JMD/TMD x-axis with P-site notation.
        figsize_per_logo : tuple of (int or float), default=(8, 3)
            Figure size ``(width, height)`` per subplot in inches. Total figure height
            is ``figsize_per_logo[1] * len(list_df_logo)``.
        fontsize_labels : int or float, optional
            Font size for axis labels.
        list_name_data : list of str, optional
            Dataset names, one per logo. Length must match ``list_df_logo``.
        name_data_pos : {'top', 'right', 'bottom', 'left'}, default='top'
            Position of the name annotation on each subplot.
        list_name_data_color : str or list of str, default='black'
            Color(s) for name annotations. A single string applies to all;
            a list must match ``list_df_logo`` in length.
        name_data_fontsize : int or float, optional
            Font size for name annotations.
        logo_font_name : str, default='Verdana'
            Font name for amino acid letter rendering.
        logo_color_scheme : str, default='weblogo_protein'
            Color scheme for amino acid letters (passed to ``logomaker``).
        logo_stack_order : {'big_on_top', 'small_on_top', 'fixed'}, default='big_on_top'
            Stacking order of letters within each position column.
        logo_width : float, default=0.96
            Relative width of each letter column (0 to 1).
        logo_vpad : float, default=0.05
            Vertical padding between stacked letters.
        logo_vsep : float, default=0.0
            Vertical separation between stacked letters.
        start : int, default=1
            Residue number assigned to the first position of JMD-N.
        tmd_color : str, default='mediumspringgreen'
            Color for TMD annotations.
        jmd_color : str, default='blue'
            Color for JMD annotations.
        fontsize_tmd_jmd : int or float, optional
            Font size for part labels. If ``None``, auto-sized.
        weight_tmd_jmd : {'normal', 'bold'}, default='normal'
            Font weight for part labels.
        highlight_tmd_area : bool, default=True
            If ``True``, shades the TMD region in each subplot.
        highlight_alpha : float, default=0.15
            Opacity of the TMD highlight.
        xtick_size : int or float, optional
            Font size for x-tick labels (bottom subplot only).
        xtick_width : int or float, default=2.0
            Line width of x-tick marks.
        xtick_length : int or float, default=11.0
            Length of x-tick marks in points.

        Returns
        -------
        fig : plt.Figure
            Figure object.
        axes : list of plt.Axes
            One ``Axes`` per logo.

        See Also
        --------
        * :meth:`AAlogoPlot.single_logo`: for a single-group visualization.
        * :class:`AAlogo`: to compute ``df_logo`` for each group.

        Examples
        --------
        .. include:: examples/AAlogoplot_multi_logo.rst
        """
        # Check input
        check_list_df_logo(list_df_logo=list_df_logo)
        ut.check_figsize(figsize=figsize_per_logo)
        check_list_name_data(list_name_data=list_name_data, list_df_logo=list_df_logo)
        check_list_name_data_color(list_name_data_color=list_name_data_color,
                                   list_df_logo=list_df_logo)
        ut.check_str_options(name="name_data_pos", val=name_data_pos,
                             list_str_options=["top", "right", "bottom", "left"])
        ut.check_str_options(name="logo_stack_order", val=logo_stack_order,
                             list_str_options=["big_on_top", "small_on_top", "fixed"])
        ut.check_str_options(name="weight_tmd_jmd", val=weight_tmd_jmd,
                             list_str_options=["normal", "bold"])
        ut.check_bool(name="highlight_tmd_area", val=highlight_tmd_area)
        ut.check_number_range(name="highlight_alpha", val=highlight_alpha,
                              min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="logo_width", val=logo_width,
                              min_val=0, max_val=1, just_int=False)
        ut.check_number_range(name="xtick_width", val=xtick_width, min_val=0, just_int=False)
        ut.check_number_range(name="xtick_length", val=xtick_length, min_val=0, just_int=False)
        tmd_len, jmd_n_len, jmd_c_len = check_parts_len(df_logo=list_df_logo[0],
                                                         jmd_n_len=self._jmd_n_len,
                                                         jmd_c_len=self._jmd_c_len)
        # Plot
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