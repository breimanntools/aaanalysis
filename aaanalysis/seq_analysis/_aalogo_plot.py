"""
This is a script for the frontend of the AAlogoPlot class.
"""
import inspect
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, List, Tuple, Literal

import aaanalysis.utils as ut
from ._aalogo import AAlogo
from ._backend._aalogo.aalogo_plot import single_logo_, multi_logo_

# Settings
DICT_LOGO_LABELS = {"probability": "Probability [%]",
                    "weight": "Weight",
                    "counts": "Counts",
                    "information": "Bits"}


# I Helper Functions
def check_df_logo(df_logo=None) -> None:
    """Check that df_logo is a valid non-empty DataFrame with numeric values."""
    ut.check_df(name="df_logo", df=df_logo, accept_none=False, accept_nan=False)
    if len(df_logo) == 0:
        raise ValueError("'df_logo' must not be empty.")


def _valid_aal_kws_keys(include_info=True):
    """Return AAlogo.get_df_logo[/_info] parameter names valid in an aal_kws dict."""
    keys = set(inspect.signature(AAlogo.get_df_logo).parameters) - {"self"}
    if include_info:
        keys &= set(inspect.signature(AAlogo.get_df_logo_info).parameters) - {"self"}
    return keys


def check_aal_kws_keys(name=None, aal_kws=None, include_info=True) -> None:
    """Check that every key in an aal_kws dict is a valid AAlogo argument name."""
    valid_keys = _valid_aal_kws_keys(include_info=include_info)
    invalid_keys = set(aal_kws) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"'{name}' contains invalid keys {sorted(invalid_keys)}; allowed keys are "
            f"{sorted(valid_keys)}.")


def check_aal_kws(aal_kws=None, df_logo=None, df_logo_info=None) -> None:
    """Check aal_kws is a valid-key dict not combined with df_logo/df_logo_info."""
    if aal_kws is None:
        return
    if not isinstance(aal_kws, dict):
        raise ValueError(
            f"'aal_kws' ({aal_kws}) should be a dictionary of keyword arguments "
            f"shared by 'AAlogo.get_df_logo' and 'AAlogo.get_df_logo_info'.")
    check_aal_kws_keys(name="aal_kws", aal_kws=aal_kws, include_info=True)
    if df_logo is not None or df_logo_info is not None:
        raise ValueError(
            "'aal_kws' is mutually exclusive with 'df_logo' and 'df_logo_info': "
            "provide either 'aal_kws' (logo data computed internally) or "
            "precomputed 'df_logo'/'df_logo_info', not both.")


def check_logo_input_source(df_logo=None, df_logo_info=None, aal_kws=None,
                            df_parts=None, labels=None, tmd_len=None) -> None:
    """Check exactly one logo-data source is given (precomputed, aal_kws, or df_parts)."""
    if labels is not None and df_parts is None:
        raise ValueError("'labels' requires 'df_parts' to be given as well.")
    if tmd_len is not None and df_parts is None:
        raise ValueError("'tmd_len' requires 'df_parts' to be given as well.")
    has_df_parts = df_parts is not None
    if has_df_parts:
        if df_logo is not None or df_logo_info is not None:
            raise ValueError(
                "'df_parts'/'labels' are mutually exclusive with 'df_logo' and "
                "'df_logo_info': provide either 'df_parts' (logo data computed "
                "internally) or precomputed 'df_logo'/'df_logo_info', not both.")
        if aal_kws is not None:
            raise ValueError(
                "'df_parts'/'labels' are mutually exclusive with 'aal_kws': pass the "
                "logo inputs either directly via 'df_parts' (with 'labels', "
                "'label_test', 'tmd_len') or bundled in 'aal_kws', not both.")
    elif df_logo is None and aal_kws is None:
        raise ValueError(
            "No logo data provided: pass precomputed 'df_logo' (optionally with "
            "'df_logo_info'), or compute it internally via 'df_parts' (with "
            "'labels', 'label_test', 'tmd_len') or 'aal_kws'.")


def check_list_aal_kws(list_aal_kws=None, list_df_logo=None, list_df_logo_info=None) -> None:
    """Check list_aal_kws is a list of valid dicts not combined with precomputed data."""
    if list_aal_kws is None:
        return
    if not isinstance(list_aal_kws, list) or len(list_aal_kws) == 0:
        raise ValueError(
            f"'list_aal_kws' ({list_aal_kws}) should be a non-empty list of "
            f"dictionaries, one per logo, holding 'AAlogo.get_df_logo' arguments.")
    for i, aal_kws in enumerate(list_aal_kws):
        if not isinstance(aal_kws, dict):
            raise ValueError(
                f"'list_aal_kws[{i}]' ({aal_kws}) should be a dictionary of "
                f"'AAlogo.get_df_logo' keyword arguments.")
        check_aal_kws_keys(name=f"list_aal_kws[{i}]", aal_kws=aal_kws,
                           include_info=True)
    if list_df_logo is not None or list_df_logo_info is not None:
        raise ValueError(
            "'list_aal_kws' is mutually exclusive with 'list_df_logo' and "
            "'list_df_logo_info': provide either 'list_aal_kws' (logo data computed "
            "internally) or precomputed 'list_df_logo'/'list_df_logo_info', not both.")


def check_list_df_logo(list_df_logo=None) -> None:
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


def check_list_df_logo_info(list_df_logo_info=None, list_df_logo=None) -> None:
    """Check list_df_logo_info is a list of Series matching list_df_logo in count and length."""
    if not isinstance(list_df_logo_info, list) or len(list_df_logo_info) == 0:
        raise ValueError("'list_df_logo_info' must be a non-empty list of Series.")
    if len(list_df_logo_info) != len(list_df_logo):
        raise ValueError(
            f"'list_df_logo_info' length ({len(list_df_logo_info)}) must match "
            f"'list_df_logo' length ({len(list_df_logo)}).")
    for i, df_logo_info in enumerate(list_df_logo_info):
        check_df_logo_info(df_logo_info=df_logo_info, df_logo=list_df_logo[i])


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


def check_df_logo_info(df_logo_info=None, df_logo=None) -> None:
    """Check that df_logo_info is a valid Series with the same length as df_logo."""
    ut.check_df(name="df_logo_info", df=df_logo_info, check_series=True,
                accept_none=False, accept_nan=False)
    if df_logo is not None and len(df_logo_info) != len(df_logo):
        raise ValueError(
            f"'df_logo_info' length ({len(df_logo_info)}) must match "
            f"'df_logo' length ({len(df_logo)}).")


def check_info_bar_ylim(info_bar_ylim=None) -> None:
    """Check that info_bar_ylim is None or a valid (min, max) tuple of numbers."""
    if info_bar_ylim is None:
        return
    if (not isinstance(info_bar_ylim, tuple) or len(info_bar_ylim) != 2
            or not all(isinstance(v, (int, float)) for v in info_bar_ylim)):
        raise ValueError("'info_bar_ylim' must be a tuple of two numbers (min, max).")
    if info_bar_ylim[0] >= info_bar_ylim[1]:
        raise ValueError(
            f"'info_bar_ylim[0]' ({info_bar_ylim[0]}) must be < 'info_bar_ylim[1]' ({info_bar_ylim[1]}).")


def check_height_ratio(height_ratio=None) -> None:
    """Check that height_ratio is a tuple of two positive numbers."""
    if (not isinstance(height_ratio, tuple) or len(height_ratio) != 2
            or not all(isinstance(v, (int, float)) and v > 0 for v in height_ratio)):
        raise ValueError("'height_ratio' must be a tuple of two positive numbers.")


def check_list_name_data(list_name_data=None, list_df_logo=None) -> None:
    """Check that list_name_data matches the number of logos if provided."""
    if list_name_data is None:
        return
    if not isinstance(list_name_data, list):
        raise ValueError("'list_name_data' must be a list of strings or None.")
    if list_df_logo is not None and len(list_name_data) != len(list_df_logo):
        raise ValueError(
            f"'list_name_data' length ({len(list_name_data)}) must match "
            f"'list_df_logo' length ({len(list_df_logo)}).")


def check_list_name_data_color(list_name_data_color=None, list_df_logo=None) -> None:
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
    Amino Acid logo Plot (**AAlogoPlot**) class for visualizing sequence logos.

    Renders single and stacked multiple sequence logos (via the logomaker [Tareen20]_
    package) with automatic target middle domain (TMD) / juxta middle domain (JMD) part
    annotations. ``jmd_n_len`` and ``jmd_c_len`` are set at initialization and used by all
    plot methods to derive the TMD region length from the logo DataFrame. The ``logo_type``
    set at initialization controls
    only the y-axis label; the logo data itself is provided as a pre-computed
    ``df_logo`` from :class:`AAlogo`.

    .. versionadded:: 1.0.3

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
            Length of the juxta middle domain (JMD)-N region (>=0). Used together with
            ``jmd_c_len`` to derive the target middle domain (TMD) length from the logo
            DataFrame for part annotations.
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
        self._logo_type = logo_type
        self._y_label = DICT_LOGO_LABELS[logo_type]
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len

    def single_logo(self,
                    df_logo: Optional[pd.DataFrame] = None,
                    df_logo_info: Optional[pd.Series] = None,
                    aal_kws: Optional[dict] = None,
                    df_parts: Optional[pd.DataFrame] = None,
                    labels: Optional[ut.ArrayLike1D] = None,
                    label_test: int = 1,
                    tmd_len: Optional[int] = None,
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
        Plot a single sequence logo with optional bit-score bar and target middle domain
        (TMD) / juxta middle domain (JMD) annotations.

        Renders a pre-computed logo matrix from :class:`AAlogo` as a letter-stack
        sequence logo using logomaker [Tareen20]_, and draws colored TMD/JMD part
        annotations beneath the x-axis. An optional bit-score bar panel can be shown
        above the logo when ``df_logo_info`` is provided. See
        :meth:`AAlogoPlot.multi_logo` for stacking multiple logos for group comparison.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        df_logo : pd.DataFrame, shape (n_positions, n_amino_acids), optional
            Logo matrix as returned by :meth:`AAlogo.get_df_logo`. Rows are residue
            positions, columns are amino acids. Required unless ``aal_kws`` is given,
            in which case it is computed internally and must be ``None``.
        df_logo_info : pd.Series, shape (n_positions,), optional
            Per-position information content as returned by :meth:`AAlogo.get_df_logo_info`.
            If provided, a bit-score bar is rendered above the main logo. Must be ``None``
            when ``aal_kws`` is given (it is then computed internally and the bar is shown).
        aal_kws : dict, optional
            :meth:`AAlogo.get_df_logo` / :meth:`AAlogo.get_df_logo_info` keyword arguments.
            If given, ``df_logo`` and ``df_logo_info`` are computed internally and both must
            be ``None``. Mutually exclusive with ``df_logo`` and ``df_logo_info`` (see Notes).
        df_parts : pd.DataFrame, shape (n_samples, n_parts), optional
            Sequence parts DataFrame with at least one of the standard part columns
            (``jmd_n``, ``tmd``, ``jmd_c``), as passed to :meth:`AAlogo.get_df_logo`. If given,
            ``df_logo`` and ``df_logo_info`` are computed internally (so the bit-score bar is
            shown). Mutually exclusive with ``df_logo`` / ``df_logo_info`` and with ``aal_kws``
            (see Notes).
        labels : array-like, shape (n_samples,), optional
            Class labels for the samples in ``df_parts``. If provided, only samples with
            ``label_test`` are included in the logo computation. Only used together with
            ``df_parts``.
        label_test : int, default=1
            Class label of the test group to select from ``labels``. Only used together with
            ``df_parts``.
        tmd_len : int, optional
            Fixed length (>=1) to align all target middle domain (TMD) sequences before
            computing the logo. If ``None``, the maximum TMD length in ``df_parts`` is used.
            Only used together with ``df_parts``.
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
        axes : plt.Axes or tuple of (plt.Axes, plt.Axes)
            When ``df_logo_info`` is ``None``: a single ``Axes`` for the logo panel.
            When ``df_logo_info`` is provided: a tuple ``(ax_logo, ax_info)`` where
            ``ax_info`` is the bit-score bar above the logo.

        Notes
        -----
        * There are three ways to supply the logo data, and exactly one must be used:

          1. precomputed ``df_logo`` (optionally with ``df_logo_info``);
          2. the raw inputs ``df_parts`` (with ``labels``, ``label_test``, ``tmd_len``);
          3. an ``aal_kws`` dict bundling the :class:`AAlogo` getter arguments.

          Mixing sources (e.g. ``df_parts`` together with ``df_logo`` or ``aal_kws``)
          raises ``ValueError``.
        * ``df_parts`` is the most direct shortcut: when given (and ``df_logo`` is
          ``None``), ``AAlogoPlot`` instantiates :class:`AAlogo` with this plot's
          ``logo_type`` and computes both ``df_logo`` (via :meth:`AAlogo.get_df_logo`)
          and ``df_logo_info`` (via :meth:`AAlogo.get_df_logo_info`) from
          ``df_parts``/``labels``/``label_test``/``tmd_len``, then renders the logo
          with the bit-score bar.
        * ``aal_kws`` is the equivalent shortcut as a single dict, useful when the
          getter arguments are assembled programmatically. It holds the arguments
          shared by both methods, e.g. ``df_parts``, ``labels``, ``label_test``,
          ``tmd_len``, ``start_n``, ``characters_to_ignore``, and ``pseudocount``.
          Unknown keys raise ``ValueError``. Example: ``aal_kws=dict(df_parts=df_parts,
          labels=labels, label_test=1, tmd_len=20)``.

        See Also
        --------
        * :meth:`AAlogoPlot.multi_logo`: for stacked multi-group comparison.
        * :class:`AAlogo`: to compute ``df_logo`` and ``df_logo_info``.

        Examples
        --------
        .. include:: examples/aal_plot_single_logo.rst
        """
        # Check input
        check_logo_input_source(df_logo=df_logo, df_logo_info=df_logo_info,
                                aal_kws=aal_kws, df_parts=df_parts, labels=labels,
                                tmd_len=tmd_len)
        check_aal_kws(aal_kws=aal_kws, df_logo=df_logo, df_logo_info=df_logo_info)
        # Build aal_kws from the direct df_parts inputs (same internal compute path)
        if df_parts is not None:
            aal_kws = dict(df_parts=df_parts, labels=labels,
                           label_test=label_test, tmd_len=tmd_len)
        if aal_kws is not None:
            aal = AAlogo(logo_type=self._logo_type)
            df_logo = aal.get_df_logo(**aal_kws)
            df_logo_info = aal.get_df_logo_info(**aal_kws)
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
        if self._verbose:
            ut.print_out(f"Plotting single logo (TMD length={tmd_len}, "
                         f"JMD-N={jmd_n_len}, JMD-C={jmd_c_len})")
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
                   list_df_logo: Optional[List[pd.DataFrame]] = None,
                   list_df_logo_info: Optional[List[pd.Series]] = None,
                   list_aal_kws: Optional[List[dict]] = None,
                   info_bar_color: str = "gray",
                   info_bar_ylim: Optional[Tuple[float, float]] = None,
                   height_ratio: Tuple[Union[int, float], Union[int, float]] = (1, 6),
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
        Plot multiple sequence logos stacked vertically for group comparison, each with
        an optional bit-score bar on top.

        All logos share the same y-axis scale, and (when shown) all bit-score bars share a
        common scale so they are comparable across groups. Target middle domain (TMD) /
        juxta middle domain (JMD) annotations are shown only on the bottom subplot to avoid
        repetition.

        .. versionadded:: 1.1.0

        Parameters
        ----------
        list_df_logo : list of pd.DataFrame, each shape (n_positions, n_amino_acids), optional
            List of logo matrices, one per group. All must have the same number of positions.
            Required unless ``list_aal_kws`` is given, in which case it is computed
            internally and must be ``None``.
        list_df_logo_info : list of pd.Series, each shape (n_positions,), optional
            Per-position information content, one per group, as returned by
            :meth:`AAlogo.get_df_logo_info`. If provided, a bit-score bar is rendered above
            each logo. Length and per-group positions must match ``list_df_logo``. Must be
            ``None`` when ``list_aal_kws`` is given (it is then computed internally and the
            bars are shown).
        list_aal_kws : list of dict, optional
            Per-group :meth:`AAlogo.get_df_logo` keyword arguments, one dict per group. If
            given, ``list_df_logo`` and ``list_df_logo_info`` are computed internally and
            must be ``None``. Mutually exclusive with ``list_df_logo`` (see Notes).
        info_bar_color : str, default='gray'
            Color of the bit-score bars in the optional top panels.
        info_bar_ylim : tuple of float, optional
            Shared y-axis limits ``(min, max)`` for all bit-score bars. If ``None``, set
            automatically from the global maximum so bars stay comparable across groups.
        height_ratio : tuple of (int or float), default=(1, 6)
            Height ratio ``(info_bar, logo)`` of each group when ``list_df_logo_info`` is
            provided.
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
        axes : list of plt.Axes or list of tuple of (plt.Axes, plt.Axes)
            When no bit-score bars are shown: one ``Axes`` per logo. When
            ``list_df_logo_info`` (or ``list_aal_kws``) is given: one
            ``(ax_logo, ax_info)`` tuple per group, where ``ax_info`` is the bar above.

        Notes
        -----
        * ``list_aal_kws`` is a convenience shortcut that skips the manual :class:`AAlogo`
          step for each group: ``AAlogoPlot`` instantiates :class:`AAlogo` with this plot's
          ``logo_type`` and computes both ``df_logo`` (via :meth:`AAlogo.get_df_logo`) and
          ``df_logo_info`` (via :meth:`AAlogo.get_df_logo_info`) per dict, so the bit-score
          bars appear automatically. Each dict holds that group's arguments, e.g.
          ``df_parts``, ``labels``, ``label_test``, ``tmd_len``, ``start_n``,
          ``characters_to_ignore``, and ``pseudocount`` (typically the same ``df_parts``
          with a different ``label_test`` per group). Passing both ``list_aal_kws`` and
          ``list_df_logo`` / ``list_df_logo_info`` raises ``ValueError``, as do unknown keys.
          Example: ``list_aal_kws=[dict(df_parts=df_parts, labels=labels, label_test=1),
          dict(df_parts=df_parts, labels=labels, label_test=0)]``.

        See Also
        --------
        * :meth:`AAlogoPlot.single_logo`: for a single-group visualization.
        * :class:`AAlogo`: to compute ``df_logo`` for each group.

        Examples
        --------
        .. include:: examples/aal_plot_multi_logo.rst
        """
        # Check input
        check_list_aal_kws(list_aal_kws=list_aal_kws, list_df_logo=list_df_logo,
                           list_df_logo_info=list_df_logo_info)
        if list_aal_kws is not None:
            aal = AAlogo(logo_type=self._logo_type)
            list_df_logo = [aal.get_df_logo(**aal_kws) for aal_kws in list_aal_kws]
            list_df_logo_info = [aal.get_df_logo_info(**aal_kws) for aal_kws in list_aal_kws]
        check_list_df_logo(list_df_logo=list_df_logo)
        if list_df_logo_info is not None:
            check_list_df_logo_info(list_df_logo_info=list_df_logo_info,
                                    list_df_logo=list_df_logo)
            check_info_bar_ylim(info_bar_ylim=info_bar_ylim)
            check_height_ratio(height_ratio=height_ratio)
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
        if self._verbose:
            ut.print_out(f"Plotting {len(list_df_logo)} logos (TMD length={tmd_len}, "
                         f"JMD-N={jmd_n_len}, JMD-C={jmd_c_len})")
        # Plot
        fig, axes = multi_logo_(list_df_logo=list_df_logo,
                                list_df_logo_info=list_df_logo_info,
                                info_bar_color=info_bar_color,
                                info_bar_ylim=info_bar_ylim,
                                height_ratio=height_ratio,
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