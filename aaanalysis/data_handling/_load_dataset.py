"""
This is a script for protein benchmark loading function. Please define new loading functions by their loaded data
by introducing a new data table in docs/source/index/tables_templates.rst.
"""
import os
import pandas as pd
import numpy as np
import re
from pandas import DataFrame
from typing import Optional, Literal
import aaanalysis.utils as ut

# Constants
STR_AA_GAP = "-"
LIST_CANONICAL_AA = ['N', 'A', 'I', 'V', 'K', 'Q', 'R', 'M', 'H', 'F', 'E', 'D', 'C', 'G', 'L', 'T', 'S', 'Y', 'W', 'P']
FOLDER_BENCHMARKS = ut.FOLDER_DATA + "benchmarks" + ut.SEP
LIST_NON_CANONICAL_OPTIONS = ["remove", "keep", "gap"]


# I Helper Functions
# Check functions
def check_name_of_dataset(name="Overview", folder_in=None):
    """"""
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


def check_non_canonical_aa(non_canonical_aa="remove"):
    """Check if non_canonical_aa is valid"""
    if non_canonical_aa not in LIST_NON_CANONICAL_OPTIONS:
        raise ValueError(f"'non_canonical_aa' ({non_canonical_aa}) should be on of following:"
                         f" {LIST_NON_CANONICAL_OPTIONS }")


def check_aa_window_size(aa_window_size=None):
    """Check if aa_window size is a positive odd integer"""
    if aa_window_size is None:
        return
    ut.check_number_range(name="aa_window_size", val=aa_window_size, min_val=1, just_int=True)
    if aa_window_size % 2 == 0:
        raise ValueError(f"'aa_window_size' ({aa_window_size}) must be an odd number.")


def post_check_df_seq(df_seq=None, n=None, name=None, n_match=True):
    """Check if length of df_seq is valid"""
    max_n = df_seq["label"].value_counts().min()
    error_message = f"'n' ({n}) is too high since smaller class for '{name}' contains {max_n} samples." \
                    f" This maximum value depends on filtering settings."
    # Validation of sequence and domain datasets
    if n is not None:
        if n_match and len(df_seq) != n*2:
            raise ValueError(error_message)
        elif len(df_seq) < n*2:
            raise ValueError(error_message)


# Helper functions
def _is_aa_level(name=None):
    return name.split("_")[0] == "AA"


def _adjust_non_canonical_aa(df=None, non_canonical_aa="remove"):
    """"""
    if non_canonical_aa == "keep":
        return df
    # Get all non-canonical amino acids
    f = lambda x: set(str(x))
    vf = np.vectorize(f)
    char_seq = set().union(*vf(df.values).flatten())
    list_non_canonical_aa = [x for x in char_seq if x not in LIST_CANONICAL_AA]
    if non_canonical_aa == "remove":
        pattern = '|'.join(list_non_canonical_aa)  # Joining list into a single regex pattern
        df = df[~df[ut.COL_SEQ].str.contains(pattern, regex=True)]
    else:
        df[ut.COL_SEQ] = [re.sub(f'[{"".join(list_non_canonical_aa)}]', STR_AA_GAP, x) for x in df[ut.COL_SEQ]]
    return df


def _get_aa_window(df_seq=None, aa_window_size=9):
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
            list_aa.append(aa_window)
            entry_pos = f"{entry}_pos{i}"
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
                 aa_window_size: Optional[int] = 9,
                 ) -> DataFrame:
    """
    Loads protein benchmarking datasets.

    The benchmarks are categorized into amino acid ('AA'), domain ('DOM'), and sequence ('SEQ') level datasets.
    By default, an overview table is provided (``name='Overview'``). For in-depth details, refer to [Breimann23a]_.

    Parameters
    ----------
    name
        The name of the loaded dataset, from the 'Dataset' column in the overview table.
    n
        Number of proteins per class, selected by index. If None, the whole dataset will be returned.
    random
        If True, ``n`` randomly selected proteins per class will be chosen.
    non_canonical_aa
        Options for handling non-canonical amino acids:

        - ``remove``: Remove sequences containing non-canonical amino acids.
        - ``keep``: Don't remove sequences containing non-canonical amino acids.
        - ``gap``: Non-canonical amino acids are replaced by the gap symbol ('X').

    min_len
        Minimum length of sequences for filtering, disabled by default.
    max_len
        Maximum length of sequences for filtering, disabled by default.
    aa_window_size
        Length of amino acid window, only used for the amino acid dataset level (``name='AA_'``) and if ``n`` is given.

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

    Examples
    --------
    >>> import aaanalysis as aa
    >>> df_seq = aa.load_dataset(name="SEQ_AMYLO", n=100)

    See Also
    --------
    * Overview of all benchmarks in :ref:`t1_overview_benchmarks`.
    * Step-by-step guide in the `Data Loading Tutorial <tutorial2a_data_loader.html>`_.
    """

    check_name_of_dataset(name=name, folder_in=FOLDER_BENCHMARKS)
    ut.check_number_range(name="n", val=n, min_val=1, accept_none=True, just_int=True)
    check_non_canonical_aa(non_canonical_aa=non_canonical_aa)
    check_min_max_val(min_len=min_len, max_len=max_len)
    check_aa_window_size(aa_window_size=aa_window_size)
    # Load overview table
    if name == "Overview":
        return ut.read_excel_cached(FOLDER_BENCHMARKS + "Overview.xlsx")
    df = ut.read_csv_cached(FOLDER_BENCHMARKS + name + ".tsv", sep="\t")
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
            return df_seq.reset_index(drop=True)
        df_seq = _get_aa_window(df_seq=df_seq, aa_window_size=aa_window_size)
    # Select balanced groups
    post_check_df_seq(df_seq=df_seq, n=n, name=name, n_match=False)
    if n is not None:
        labels = set(df_seq[ut.COL_LABEL])
        if random:
            df_seq = pd.concat([df_seq[df_seq[ut.COL_LABEL] == l].sample(n) for l in labels])
        else:
            df_seq = pd.concat([df_seq[df_seq[ut.COL_LABEL] == l].head(n) for l in labels])
    # Adjust index
    df_seq = df_seq.reset_index(drop=True)
    post_check_df_seq(df_seq=df_seq, n=n, name=name)
    return df_seq
