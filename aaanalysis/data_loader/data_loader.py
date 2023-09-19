"""
This is a script for loading protein sequence benchmarking datasets and amino acid scales and
their two-level classification (AAontology).
"""
import os
import pandas as pd
import numpy as np
import re
from typing import Optional, Literal
import aaanalysis.utils as ut


# I Helper Functions
STR_AA_GAP = "-"
LIST_CANONICAL_AA = ['N', 'A', 'I', 'V', 'K', 'Q', 'R', 'M', 'H', 'F', 'E', 'D', 'C', 'G', 'L', 'T', 'S', 'Y', 'W', 'P']
NAME_SCALE_SETS_BASE = [ut.STR_SCALES, ut.STR_SCALES_RAW]
NAMES_SCALE_SETS = NAME_SCALE_SETS_BASE + [ut.STR_SCALE_CAT, ut.STR_SCALES_PC, ut.STR_TOP60, ut.STR_TOP60_EVAL]


# II Main Functions
def _adjust_non_canonical_aa(df=None, non_canonical_aa="remove"):
    """"""
    list_options = ["remove", "keep", "gap"]
    if non_canonical_aa not in list_options:
        raise ValueError(f"'non_canonical_aa' ({non_canonical_aa}) should be on of following: {list_options}")
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


def load_dataset(name: str = "INFO",
                 n: Optional[int] = None,
                 non_canonical_aa: Literal["remove", "keep", "gap"] = "remove",
                 min_len: Optional[int] = None,
                 max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Load protein benchmarking datasets.

    The benchmarks are distinguished into residue/amino acid ('AA'), domain ('DOM'), and sequence ('SEQ') level
    datasets. An overview table can be retrieved by using default setting (name='INFO'). A through analysis of
    the residue and sequence datasets can be found in [Breimann23a].

    Parameters
    ----------
    name
        Name of the dataset. See 'Dataset' column in overview table.
    n
        Number of proteins per class. If None, the whole dataset will be returned.
    non_canonical_aa
        Options for modifying non-canonical amino acids:
        - 'remove': Sequences containing non-canonical amino acids are removed.
        - 'keep': Sequences containing non-canonical amino acids are not removed.
        - 'gap': Sequences are kept and modified by replacing non-canonical amino acids by gap symbol ('X').
    min_len
        Minimum length of sequences for filtering. None to disable
    max_len
        Maximum length of sequences for filtering. None to disable

    Returns
    -------
    df_seq
        Dataframe with the selected sequence dataset.

    Notes
    -----
    See further information on the benchmark datasets in

    """
    ut.check_non_negative_number(name="n", val=n, accept_none=True)
    ut.check_non_negative_number(name="min_len", val=min_len, accept_none=True)
    folder_in = ut.FOLDER_DATA + "benchmarks" + ut.SEP
    check_name_of_dataset(name=name, folder_in=folder_in)
    # Load overview table
    if name == "INFO":
        return pd.read_excel(folder_in + "INFO_benchmarks.xlsx")
    df = pd.read_csv(folder_in + name + ".tsv", sep="\t")
    # Filter Rdata
    if min_len is not None:
        mask = [len(x) >= min_len for x in df[ut.COL_SEQ]]
        df = df[mask]
    if max_len is not None:
        mask = [len(x) <= max_len for x in df[ut.COL_SEQ]]
        df = df[mask]
    # Adjust non-canonical amino acid (keep, remove, or replace by gap)
    df_seq = _adjust_non_canonical_aa(df=df, non_canonical_aa=non_canonical_aa)
    # Select balanced groups
    if n is not None:
        labels = set(df_seq[ut.COL_LABEL])
        df_seq = pd.concat([df_seq[df_seq[ut.COL_LABEL] == l].head(n) for l in labels])
    return df_seq


# Load scales
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


# Extend for AAclustTop60
def load_scales(name="scales", just_aaindex=False, unclassified_in=True):
    """
    Load amino acid scales, scale classification (AAontology), or scale evaluation.

    A through analysis of the residue and sequence datasets can be found in TODO[Breimann23a].

    Parameters
    ----------
    name : str, default = 'scales'
        Name of the dataset to load. Options are 'scales', 'scales_raw', 'scale_cat',
        'scales_pc', 'top60', and 'top60_eval'.
    unclassified_in : bool, optional
        Whether unclassified scales should be included. The 'Others' category counts as unclassified.
        Only relevant if `name` is 'scales', 'scales_raw', or 'scale_classification'.
    just_aaindex : bool, optional
        Whether only scales provided from AAindex should be given.
        Only relevant if `name` is 'scales', 'scales_raw', or 'scale_classification'.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Dataframe for the selected scale dataset.
    """
    if name not in NAMES_SCALE_SETS:
        raise ValueError(f"'name' ({name}) is not valid. Choose one of following: {NAMES_SCALE_SETS}")
    # Load _data
    df_cat = pd.read_excel(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
    df_cat = _filter_scales(df_cat=df_cat, unclassified_in=unclassified_in, just_aaindex=just_aaindex)
    if name == ut.STR_SCALE_CAT:
        return df_cat
    df = pd.read_excel(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
    # Filter scales
    if name in NAME_SCALE_SETS_BASE:
        df = df[[x for x in list(df) if x in list(df_cat[ut.COL_SCALE_ID])]]
    return df
