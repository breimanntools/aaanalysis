"""
TODO google motif meme
"""

import logomaker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple, Type, Literal

import aaanalysis.utils as ut


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


def add_bit_score_bar(ax_info=None, df_logo_info=None, bar_color="gray", show_right=True):
    """"""
    ax_info.bar(df_logo_info.index, df_logo_info.values, color=bar_color)
    ax_info.spines["top"].set_visible(False)
    if show_right:
        ax_info.spines["left"].set_visible(False)
        ax_info.yaxis.set_label_position("right")
        ax_info.yaxis.tick_right()
    else:
        ax_info.spines["right"].set_visible(False)
    ax_info.xaxis.set_tick_params(labelbottom=False)
    ax_info.set_ylabel("Bits")


def add_name_test(ax=None, name_test=None, name_data_pos=None, fontsize=None, color="black"):
    """Add the name_test to the plot at the specified position."""
    args = dict(transform=ax.transAxes, fontsize=fontsize, color=color, multialignment="center")
    if name_data_pos == "top":
        ax.text(0.5, 1.02, name_test, ha="center", va="bottom", **args)
    elif name_data_pos == "right":
        ax.text(1.02, 0.5, name_test, ha='left', va='center', **args)
    elif name_data_pos == "bottom":
        ax.text(0.5, -0.25, name_test, ha='center', va='top', **args)
    elif name_data_pos == "left":
        ax.text(-0.12, 0.5, name_test, ha='right', va='center', **args)


# II Main Functions
class AALogoPlot:
    """
    AALogoPlot class for visualizing sequence logos.
    """

    def __init__(self,
                 logo_type="counts",
                 jmd_n_len: int = 10,
                 jmd_c_len: int = 10,
                 verbose: bool = True
                 ):
        """
        Parameters
        ----------
        jmd_n_len : int, default=10
            Length of JMD-N (>=0).
        jmd_c_len: int, default=10
            Length of JMD-C (>=0).
        verbose: bool, default=True
            If ``True``, verbose outputs are enabled.

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
        self._logo_type = logo_type
        self._y_label = DICT_LOGO_LABELS[logo_type]
        # Set consistent length of JMD-N and JMD-C
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len

    def plot_single_logo(self,
                         # Data and Plot Type
                         df_logo: pd.DataFrame = None,
                         df_logo_info: pd.Series = None,
                         name_data: Optional[str] = None,
                         name_data_pos: Literal["top", "right", "bottom", "left"] = "top",
                         name_data_color: str = "black",
                         name_data_fontsize: Union[int, float] = None,
                         target_site: Optional[int] = None,
                         figsize: Tuple[Union[int, float], Union[int, float]] = (8, 3.5),
                         logo_width: float = 0.96,
                         logo_vpad: float = 0.05,
                         logo_vsep: float = 0.0,
                         logo_font_name: str = "Verdana",
                         logo_color_scheme: str = "weblogo_protein",
                         logo_stack_order: Literal["big_on_top", "small_on_top", "fixed"] = "big_on_top",

                         # Appearance of Parts (TMD-JMD)
                         start: int = 1,
                         tmd_color: str = "mediumspringgreen",
                         jmd_color: str = "blue",
                         fontsize_tmd_jmd: Union[int, float] = None,
                         weight_tmd_jmd: Literal['normal', 'bold'] = "normal",
                         fontsize_labels: Union[int, float] = 12,
                         add_xticks_pos: bool = False,
                         highlight_tmd_area: bool = True,
                         highlight_alpha: float = 0.15,

                         # Bit-score barplot
                         bar_color: str = "gray",
                         ):
        """Plot a single sequence logo with an optional split view for bit-score information."""
        # Check input
        ut.check_figsize(figsize=figsize)
        tmd_len, jmd_n_len, jmd_c_len = check_parts_len(df_logo=df_logo, jmd_n_len=self._jmd_n_len,
                                                        jmd_c_len=self._jmd_c_len)

        # Set figure and axes layout (ensure zero spacing)
        if df_logo_info is not None:
            _args = dict(nrows=2, gridspec_kw={"height_ratios": [0.4, 3]}, figsize=figsize, sharex=True)
            fig, (ax_info, ax_logo) = plt.subplots(**_args)
        else:
            fig, ax_logo = plt.subplots(figsize=figsize)

        # Plot sequence logo
        logomaker.Logo(df_logo, ax=ax_logo, figsize=figsize, font_name=logo_font_name,
                       color_scheme=logo_color_scheme, width=logo_width,
                       vpad=logo_vpad, vsep=logo_vsep,
                       stack_order=logo_stack_order)

        # Add TMD-JMD elements TODO adjust bar_height, TMD, JMD text position
        args_parts = dict(ax=ax_logo, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        ut.add_tmd_jmd_bar(**args_parts, x_shift=-0.5, jmd_color=jmd_color, tmd_color=tmd_color, bar_height_factor=2)
        ut.add_tmd_jmd_xticks(**args_parts, x_shift=0, xtick_size=15, xtick_length=15, start=start)
        ut.add_tmd_jmd_text(**args_parts, x_shift=-0.5, weight_tmd_jmd=weight_tmd_jmd)

        #ut.highlight_tmd_area(**args_parts, x_shift=-0.5)

        # Adjust labels and formatting
        ax_logo.set_ylabel(self._y_label)
        sns.despine(ax=ax_logo, top=False)

        if df_logo_info is not None:
            add_bit_score_bar(ax_info=ax_info, df_logo_info=df_logo_info, bar_color=bar_color)
        if name_data is not None:
            fs = ut.plot_gco() if name_data_fontsize is None else name_data_fontsize
            ax = ax_info if name_data_pos == "top" else ax_logo
            add_name_test(ax=ax, name_test=name_data, name_data_pos=name_data_pos, color=name_data_color,
                          fontsize=fs)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

    def plot_comparative_logo(self,
                              list_df_logo=None,
                              list_names=None,
                              parts_to_plot=["JMD_N", "TMD", "JMD_C"],
                              figsize_per_plot=(8, 3.5),
                              start_tmd_at_n=True):
        """Plot multiple sequence logos for comparison, with adjustable JMD/TMD sizes."""
        num_datasets = len(list_df_logo)

        figsize = (figsize_per_plot[0] * num_datasets, figsize_per_plot[1])
        fig, axes = plt.subplots(1, num_datasets, figsize=figsize, sharey=True)

        if num_datasets == 1:
            axes = [axes]


