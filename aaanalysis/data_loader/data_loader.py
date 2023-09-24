"""
This is a script for general data loading functions, such as
a) Protein sequence benchmarking datasets
b) Amino acid scales datasets or their two-level classification (AAontology)
Please define new loading functions by their loaded data by introducing a new data table in
docs/source/index/tables_templates.rst.
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
FOLDER_BENCHMARKS = folder_in = ut.FOLDER_DATA + "benchmarks" + ut.SEP
LIST_NON_CANONICAL_OPTIONS = ["remove", "keep", "gap"]


# I Helper Functions
# Helper functions for load_datasets
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

# Check functions for load_dataset
def check_name_of_dataset(name="INFO", folder_in=None):
    """"""
    if name == "INFO":
        return
    list_datasets = [x.split(".")[0] for x in os.listdir(folder_in) if "." in x]
    if name not in list_datasets:
        list_aa = [x for x in list_datasets if 'AA' in x]
        list_seq = [x for x in list_datasets if 'SEQ' in x]
        list_dom = [x for x in list_datasets if 'DOM' in x]
        raise ValueError(f"'name' ({name}) is not valid."
                         f"\n Amino acid datasets: {list_aa}"
                         f"\n Sequence datasets: {list_seq}"
                         f"\n Domain datasets: {list_dom}")


def check_min_max_val(min_len=None, max_len=None):
    """Check if min_val and max_val are valid and match"""
    ut.check_non_negative_number(name="min_len", val=min_len, min_val=1, accept_none=True, just_int=True)
    ut.check_non_negative_number(name="max_len", val=max_len, min_val=1, accept_none=True, just_int=True)
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
    ut.check_non_negative_number(name="aa_window_size", val=aa_window_size, min_val=1, just_int=True)
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


# Helper functions for load_scales
def _filter_scales(df_cat=None, unclassified_in=False, just_aaindex=False):
    """Filter scales for unclassified and aaindex scales"""
    list_ids_not_in_aaindex = [x for x in df_cat[ut.COL_SCALE_ID] if "LINS" in x or "KOEH" in x]
    list_ids_unclassified = [x for x, cat, sub_cat in zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_CAT], df_cat[ut.COL_SUBCAT])
                             if "Unclassified" in sub_cat or cat == "Others"]
    list_ids_to_exclude = []
    if not unclassified_in:
        list_ids_to_exclude.extend(list_ids_unclassified)
    if just_aaindex:
        list_ids_to_exclude.extend(list_ids_not_in_aaindex)
    df_cat = df_cat[~df_cat[ut.COL_SCALE_ID].isin(list_ids_to_exclude)]
    return df_cat


def _get_selected_scales(top60_n=None):
    """"""
    df_eval = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_TOP60}.xlsx").drop("top60_id", axis=1)
    # Find the names of the index where the value is not 0
    _df = df_eval.iloc[top60_n - 1]
    selected_scales = _df[_df != 0].index.tolist()
    return selected_scales


# Check functions for load_scales
def check_name_of_scale(name=None):
    # Check if the provided scale name is valid
    if name not in ut.NAMES_SCALE_SETS:
        raise ValueError(f"'name' ('{name}') is not valid. Choose one of following: {ut.NAMES_SCALE_SETS}")


def check_top60_n(name=None, top60_n=None):
    """Check if name is valid and top60_n is between 1 and 60"""
    if top60_n is None:
        return
    ut.check_non_negative_number(name="top60_n", val=top60_n, min_val=1, max_val=60, just_int=True)
    matching_scale_sets = [ut.STR_SCALES, ut.STR_SCALE_CAT, ut.STR_SCALES_RAW]
    if name not in matching_scale_sets:
        raise ValueError(f"'name' ('{name}') is not valid for 'top60_n' ({top60_n})."
                         f" Choose one of following: {ut.NAMES_SCALE_SETS}")

# II Main Functions
def load_dataset(name: str = "INFO",
                 n: Optional[int] = None,
                 random: bool = False,
                 non_canonical_aa: Literal["remove", "keep", "gap"] = "remove",
                 min_len: Optional[int] = None,
                 max_len: Optional[int] = None,
                 aa_window_size: Optional[int] = 9,
                 ) -> DataFrame:
    """
    Load protein benchmarking datasets.

    The benchmarks are categorized into amino acid ('AA'), domain ('DOM'), and sequence ('SEQ') level datasets.
    By default, an overview table is provided (``name='INFO'``). For in-depth details, refer to [Breimann23a]_.

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
        general info on all benchmark datasets (``df_info``).

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
    ut.check_non_negative_number(name="n", val=n, min_val=1, accept_none=True, just_int=True)
    check_non_canonical_aa(non_canonical_aa=non_canonical_aa)
    check_min_max_val(min_len=min_len, max_len=max_len)
    check_aa_window_size(aa_window_size=aa_window_size)
    # Load overview table
    if name == "INFO":
        return ut.read_excel_cached(FOLDER_BENCHMARKS + "INFO_benchmarks.xlsx")
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

# Load scales
def load_scales(name: str = "scales",
                just_aaindex: bool = False,
                unclassified_in: bool = True,
                top60_n: Optional[int] = None
                ) -> DataFrame:
    """
    Load amino acid scales, scale classifications (AAontology), or scale evaluations.

    The amino acid scales (``name='scales_raw'``) encompass all scales from AAindex ([Kawashima08]_) along with two
    additional data sources. These scales were min-max normalized (``name='scales'``) and organized in a two-level
    classification called AAontology (``name='scales_cat'``), as detailed in [Breimann23b]_. The first 20 principal
    components (PCs) of all compressed scales are provided (``name='scales_pc'``) and were used for an in-depth analysis
    of redundancy-reduced scale subsets obtained by :class:`AAclust` ([Breimann23a]_). The top 60 scale sets from
    this analysis are available either collectively (``name='top60'``) or individually (``top60_n='1-60'``),
    accompanied by their evaluations in the ``'top60_eval'`` dataset.

    Parameters
    ----------
    name
        Name of the loaded dataset: 'scales', 'scales_raw', 'scales_cat', 'scales_pc', 'top60', 'top60_eval'.
    just_aaindex
        If True, returns only scales from AAindex. Relevant only for 'scales', 'scales_raw', or 'scales_cat'.
    unclassified_in
        Determines inclusion of unclassified scales. Relevant only for 'scales', 'scales_raw', or 'scales_cat'.
    top60_n
         Select the n-th scale set from top60 sets and return it for 'scales', 'scales_raw', or 'scales_cat'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the chosen dataset, recommended to be named by their name suffix (``df_scales``, ``df_cat``).

    Notes
    -----
    - ``df_cat`` includes the following columns:

        * 'scale_id': ID of scale (from AAindex or following the same naming convention).
        * 'category': Category of scale (defined in AAontology).
        * 'subcategory': Subcategory of scale (AAontology).
        * 'scale_name': Name of scale derived from scale description.
        * 'scale_description': Description of scale (derived from AAindex).

    - Scales under the 'Others' category are considered unclassified.

    Examples
    --------
    >>> import aaanalysis as aa
    >>> df_scales = aa.load_scales()
    >>> df_cat = aa.load_scales(name="scales_cat")

    See Also
    --------
    * Overview of all loading options: :ref:`t2_overview_scales`.
    * AAontology: :ref:`t3a_aaontology_categories` and :ref:`t3b_aaontology_subcategories` tables.
    * Step-by-step guide in the `Scale Loading Tutorial <tutorial2b_scales_loader.html>`_.
    * :class:`AAclust` for customizing redundancy-reduced scale sets.
    """
    check_name_of_scale(name=name)
    ut.check_bool(name="just_aaindex", val=just_aaindex)
    ut.check_bool(name="unclassified_in", val=unclassified_in)
    check_top60_n(name=name, top60_n=top60_n)

    # Load and filter top60 scales
    if top60_n is not None:
        selected_scales = _get_selected_scales(top60_n=top60_n)
        if name == ut.STR_SCALE_CAT:
            df_cat = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
            df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(selected_scales)].reset_index(drop=True)
            return df_cat
        elif name in [ut.STR_SCALES, ut.STR_SCALES_RAW]:
            df = ut.read_excel_cached(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
            df = df[selected_scales]
            return df
        else:
            raise ValueError(f"Wrong 'name' ('{name}') for 'top60_n")

    # Load unfiltered data
    if unclassified_in and not just_aaindex:
        if name == ut.STR_SCALE_CAT:
            df_cat = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
            return df_cat
        df = ut.read_excel_cached(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
        return df

    # Load and filter scale categories
    df_cat = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
    df_cat = _filter_scales(df_cat=df_cat, unclassified_in=unclassified_in, just_aaindex=just_aaindex)
    if name == ut.STR_SCALE_CAT:
        return df_cat.reset_index(drop=True)

    # Load and filter scales
    df = ut.read_excel_cached(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
    if name in [ut.STR_SCALES, ut.STR_SCALES_RAW]:
        selected_scales = [x for x in list(df) if x in list(df_cat[ut.COL_SCALE_ID])]
        df = df[selected_scales]
    return df
