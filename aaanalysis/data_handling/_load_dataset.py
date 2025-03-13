"""
This is a script for protein benchmark loading function. Please define new loading functions by their loaded data
by introducing a new data table in docs/source/index/tables_templates.rst.
"""
from typing import Optional, Literal, Union
import os
import pandas as pd
import numpy as np
import re
import warnings
from pandas import DataFrame
import aaanalysis.utils as ut

# Constants
FOLDER_BENCHMARKS = ut.FOLDER_DATA + "benchmarks" + ut.SEP
LIST_NON_CANONICAL_OPTIONS = ["remove", "keep", "gap"]
LIST_CLEAVAGE_SITE_DATA = ["AA_CASPASE3", "AA_FURIN", "AA_MMP2"]


# I Helper Functions
# Check functions
def check_name_of_dataset(name="Overview", folder_in=None):
    """Check if name of dataset is valid"""
    if name == "Overview":
        return
    list_datasets = [x.split(".")[0] for x in os.listdir(folder_in)
                     if "." in x and not x.startswith(".")]
    if name not in list_datasets:
        list_aa = [x for x in list_datasets if 'AA' in x]
        list_seq = [x for x in list_datasets if 'SEQ' in x]
        list_dom = [x for x in list_datasets if 'DOM' in x]
        raise ValueError(f"'name' ({name}) is not valid. Chose one of the following:"
                         f"\n Amino acid datasets: {list_aa}"
                         f"\n Sequence datasets: {list_seq}"
                         f"\n Domain datasets: {list_dom}")


def check_min_max_val(min_len=None, max_len=None):
    """Check if min_val and max_val are valid and match"""
    ut.check_number_range(name="min_len", val=min_len, min_val=1, accept_none=True, just_int=True)
    ut.check_number_range(name="max_len", val=max_len, min_val=1, accept_none=True, just_int=True)
    if min_len is None or max_len is None:
        return
    if isinstance(min_len, int) and isinstance(max_len, int) and min_len > max_len:
        raise ValueError(f"'min_len' ({min_len}) should not be smaller than 'max_len' ({max_len})")


def check_aa_window_size(aa_window_size=None, is_cs_dataset=False):
    """Check if aa_window size is a positive odd integer"""
    if aa_window_size is None:
        return
    ut.check_number_range(name="aa_window_size", val=aa_window_size, min_val=1, just_int=True)
    if aa_window_size % 2 == 0 and not is_cs_dataset:
        raise ValueError(f"'aa_window_size' ({aa_window_size}) must be an odd number. "
                         f"Only the following cleavage site datasets can have odd or even sizes: {LIST_CLEAVAGE_SITE_DATA}")


def post_check_df_seq(df_seq=None, n=None, name=None):
    """Check if length of df_seq is valid"""
    max_n = df_seq[ut.COL_LABEL].value_counts().min()
    warning_message = f"'n' ({n}) is too high since the smaller class for '{name}' contains {max_n} samples." \
                      f"\nThis maximum value depends on the filtering settings used."
    # Validation of sequence and domain datasets
    if n is not None and len(df_seq) != n*2:
        warnings.warn(warning_message)


# Helper functions
def _is_aa_level(name=None):
    return name.split("_")[0] == "AA"


def _is_cleavage_site_dataset(name=None):
    return name in LIST_CLEAVAGE_SITE_DATA


def _adjust_non_canonical_aa(df=None, non_canonical_aa="remove"):
    """Adjust non-canonical amino acids"""
    if non_canonical_aa == "keep":
        return df
    # Get all non-canonical amino acids
    f = lambda x: set(str(x))
    vf = np.vectorize(f)
    char_seq = set().union(*vf(df.values).flatten())
    list_non_canonical_aa = [x for x in char_seq if x not in ut.LIST_CANONICAL_AA]
    if non_canonical_aa == "remove":
        pattern = '|'.join(list_non_canonical_aa)  # Joining list into a single regex pattern
        df = df[~df[ut.COL_SEQ].str.contains(pattern, regex=True)]
    else:
        df[ut.COL_SEQ] = [re.sub(f'[{"".join(list_non_canonical_aa)}]', ut.STR_AA_GAP, x) for x in df[ut.COL_SEQ]]
    return df


def _get_aa_window_even(df_seq=None, aa_window_size=9):
    """Get amino acid windows from df_seq"""
    min_seq_len = df_seq[ut.COL_SEQ].apply(len).min()
    if df_seq[ut.COL_SEQ].apply(len).min() <= aa_window_size:
        raise ValueError(f"'aa_window_size' ({aa_window_size}) should be smaller than shortest sequence ({min_seq_len})")
    list_aa = []
    list_labels = []
    list_entries = []
    for entry, seq, _labels in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ], df_seq[ut.COL_LABEL]):
        for i, in zip(range(len(seq))):
            n_half = int(aa_window_size/2)
            start_pos, stop_pos = max(0, i-n_half), min(i+n_half, len(seq))
            aa_window = seq[start_pos:stop_pos]
            entry_pos = f"{entry}_{i}|{i+1}"
            list_aa.append(aa_window)
            list_entries.append(entry_pos)
        labels = [int(x) for x in _labels.split(",")]
        labels = [1 if l == 1 and labels[i-1] == 1 else 0 for i, l in enumerate(labels)]
        list_labels.extend(labels)
    df_seq = pd.DataFrame({ut.COL_ENTRY: list_entries, ut.COL_SEQ: list_aa, ut.COL_LABEL: list_labels})
    df_seq = df_seq[df_seq[ut.COL_SEQ].apply(len) == aa_window_size]    # Remove too short windows at sequence edges
    return df_seq


def _get_aa_window_odd(df_seq=None, aa_window_size=9):
    """Get amino acid windows from df_seq"""
    min_seq_len = df_seq[ut.COL_SEQ].apply(len).min()
    if df_seq[ut.COL_SEQ].apply(len).min() <= aa_window_size:
        raise ValueError(f"'aa_window_size' ({aa_window_size}) should be smaller than shortest sequence ({min_seq_len})")
    n_pre = n_post = int((aa_window_size - 1) / 2)  # checked to be an odd number before
    list_aa = []
    list_labels = []
    list_entries = []
    for entry, seq, labels in zip(df_seq[ut.COL_ENTRY], df_seq[ut.COL_SEQ], df_seq[ut.COL_LABEL]):
        for i in range(len(seq)):
            start_pre, end_pre = max(i - n_pre, 0), i
            start_post, end_post = i + 1, i + n_post + 1
            aa_window = seq[start_pre:end_pre] + seq[i] + seq[start_post:end_post]
            entry_pos = f"{entry}_pos{i}"
            list_aa.append(aa_window)
            list_entries.append(entry_pos)
        list_labels.extend(labels.split(","))
    df_seq = pd.DataFrame({ut.COL_ENTRY: list_entries, ut.COL_SEQ: list_aa, ut.COL_LABEL: list_labels})
    df_seq = df_seq[df_seq[ut.COL_SEQ].apply(len) == aa_window_size]    # Remove too short windows at sequence edges
    return df_seq


# II Main Functions
def load_dataset(name: str = "Overview",
                 n: Optional[int] = None,
                 random: bool = False,
                 non_canonical_aa: Literal["remove", "keep", "gap"] = "remove",
                 min_len: Optional[int] = None,
                 max_len: Optional[int] = None,
                 aa_window_size: Union[int, None] = 9,
                 ) -> DataFrame:
    """
    Load protein benchmarking datasets.

    The benchmarks are categorized into amino acid ('AA'), domain ('DOM'), and sequence ('SEQ') level datasets.
    By default, an overview table is provided (``name='Overview'``). For in-depth details, refer to [Breimann24a]_.

    Parameters
    ----------
    name : str, default='Overview'
        The name of the loaded dataset, from the 'Dataset' column in the overview table.
    n : int, optional
        Number of proteins per class, selected by index. If ``None``, the whole dataset will be returned.
    random : bool, default=False
        If ``True``, ``n`` randomly selected proteins per class will be chosen.
    non_canonical_aa : {'remove', 'keep', 'gap'}, default='remove'
        Options for handling non-canonical amino acids:

        - ``remove``: Remove sequences containing non-canonical amino acids.
        - ``keep``: Don't remove sequences containing non-canonical amino acids.
        - ``gap``: Non-canonical amino acids are replaced by the gap symbol ('X').

    min_len : int, optional
        Minimum length of sequences for filtering.
    max_len : int, optional
        Maximum length of sequences for filtering.
    aa_window_size : int, default=9
        Length of amino acid window, only used for the amino acid dataset level (``name='AA_'``). Disabled if ``None``.
        Must be odd, except for cleavage site datasets (e.g., 'AA_CASPASE3', 'AA_FURIN', 'AA_MMP2').

    Returns
    -------
    pandas.DataFrame
        A DataFrame of either the selected sequence dataset (``df_seq``) or
        overview on all benchmark datasets (``df_overview``).

    Notes
    -----
    ``df_seq`` includes these columns:

    - 'entry': Protein identifier, either the UniProt accession number or an id based on index.
    - 'sequence': Amino acid sequence.
    - 'label': Binary classification label (0 for negatives, 1 for positives).
    - 'tmd_start', 'tmd_stop': Start and stop positions of TMD (present only at the domain level).
    - 'jmd_n', 'tmd', 'jmd_c': Sequences for JMD_N, TMD, and JMD_C respectively.

    See Also
    --------
    * Overview of all benchmarks in :ref:`t1_overview_benchmarks`.
    * Step-by-step guide in the `Data Loading Tutorial <tutorial2a_data_loader.html>`_.

    Examples
    --------
    .. include:: examples/load_dataset.rst
    """
    # Check input
    check_name_of_dataset(name=name, folder_in=FOLDER_BENCHMARKS)
    ut.check_number_range(name="n", val=n, min_val=1, accept_none=True, just_int=True)
    ut.check_str_options(name="non_canonical_aa", val=non_canonical_aa,
                         list_str_options=LIST_NON_CANONICAL_OPTIONS)
    check_min_max_val(min_len=min_len, max_len=max_len)
    is_cs_dataset = _is_cleavage_site_dataset(name=name)
    check_aa_window_size(aa_window_size=aa_window_size, is_cs_dataset=is_cs_dataset)

    # Load overview table
    if name == "Overview":
        return ut.read_csv_cached(FOLDER_BENCHMARKS + f"Overview.{ut.STR_FILE_TYPE}").copy()
    df = ut.read_csv_cached(FOLDER_BENCHMARKS + name + f".{ut.STR_FILE_TYPE}")
    # Filter data
    if min_len is not None:
        mask = [len(x) >= min_len for x in df[ut.COL_SEQ]]
        df = df[mask]
        if len(df) == 0:
            raise ValueError(f"'min_len' ({min_len}) is too high and removed all sequences.")
    if max_len is not None:
        mask = [len(x) <= max_len for x in df[ut.COL_SEQ]]
        df = df[mask]
        if len(df) == 0:
            raise ValueError(f"'max_len' ({max_len}) is too low and removed all sequences.")
    # Adjust non-canonical amino acid (keep, remove, or replace by gap)
    df_seq = _adjust_non_canonical_aa(df=df, non_canonical_aa=non_canonical_aa)
    # Adjust amino acid windows for 'AA' datasets
    if _is_aa_level(name=name):
        # Special case that unfiltered df_seq is returned
        if aa_window_size is None:
            return df_seq.reset_index(drop=True).copy()
        if aa_window_size % 2 != 0:
            df_seq = _get_aa_window_odd(df_seq=df_seq, aa_window_size=aa_window_size)
        else:
            df_seq = _get_aa_window_even(df_seq=df_seq, aa_window_size=aa_window_size)
    # Select balanced groups
    if n is not None:
        labels = set(df_seq[ut.COL_LABEL])
        if random:
            df_seq = pd.concat([df_seq[df_seq[ut.COL_LABEL] == l].sample(n) for l in labels])
        else:
            df_seq = pd.concat([df_seq[df_seq[ut.COL_LABEL] == l].head(n) for l in labels])
    post_check_df_seq(df_seq=df_seq, n=n, name=name)
    # Adjust index and column values
    df_seq = df_seq.reset_index(drop=True)
    df_seq[ut.COL_LABEL] = df_seq[ut.COL_LABEL].astype(int)
    return df_seq.copy()
