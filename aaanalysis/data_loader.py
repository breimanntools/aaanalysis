"""
This is a script for loading protein sequence benchmarking datasets and amino acid scales including classification
"""
import os
import pandas as pd
import numpy as np
import re

import aaanalysis._utils as ut


# I Helper Functions
STR_AA_GAP = "-"
LIST_CANONICAL_AA = ['N', 'A', 'I', 'V', 'K', 'Q', 'R', 'M', 'H', 'F', 'E', 'D', 'C', 'G', 'L', 'T', 'S', 'Y', 'W', 'P']


def check_non_negative_number(name=None, val=None, min_val=0, max_val=None, accept_none=False, just_int=True):
    """Check if value of given name variable is non-negative integer"""
    check_types = [int] if just_int else [float, int]
    str_check = "non-negative int" if just_int else "non-negative float or int"
    add_str = f"n>{min_val}" if max_val is None else f"{min_val}<=n<={max_val}"
    if accept_none:
        add_str += " or None"
    error = f"'{name}' ({val}) should be {str_check} n, where " + add_str
    if accept_none and val is None:
        return None
    if type(val) not in check_types:
        raise ValueError(error)
    if val < min_val:
        raise ValueError(error)
    if max_val is not None and val > max_val:
        raise ValueError(error)


# II Main Functions
def load_dataset(name="INFO", n=None, non_canonical_aa_as_gaps=False, min_len=None, max_len=None):
    """Load one of following protein sequence benchmarking datasets:
    Load general information about datasets by 'name'='INFO
    :arg name: name of dataset
    :arg n: number of proteins per class (if None, whole dataset will be returned)
    :arg non_canonical_aa_as_gaps: boolean whether non canonical amino acid should be replaced by gap symbol
    :arg min_len: minimum length of sequences used for filtering
    :arg max_len: maximum length of sequences used for filtering
    :return df: dataframe with selected dataset
    """
    check_non_negative_number(name="n", val=n, accept_none=True)
    check_non_negative_number(name="min_len", val=min_len, accept_none=True)
    folder_in = ut.FOLDER_DATA + "benchmarks" + ut.SEP
    if name == "INFO":
        return pd.read_excel(folder_in + "INFO_benchmarks.xlsx")
    list_datasets = [x.split(".")[0] for x in os.listdir(folder_in) if "." in x]
    if name not in list_datasets:
        list_aa = [x for x in list_datasets if 'AA' in x]
        list_seq = [x for x in list_datasets if 'SEQ' in x]
        raise ValueError(f"'name' ({name}) is not valid.\n Amino acid datasets: {list_aa}\n Sequence datasets: {list_seq}")
    df = pd.read_csv(folder_in + name + ".tsv", sep="\t")
    # Filter data
    if min_len is not None:
        mask = [len(x) >= min_len for x in df["sequence"]]
        df = df[mask]
    if max_len is not None:
        mask = [len(x) <= max_len for x in df["sequence"]]
        df = df[mask]
    if n is not None:
        labels = set(df["label"])
        df = pd.concat([df[df["label"] == l].head(n) for l in labels])
    # Replace non canonical amino acid by gap
    if non_canonical_aa_as_gaps:
        f = lambda x: set(str(x))
        vf = np.vectorize(f)
        char_seq = set().union(*vf(df.values).flatten())
        list_non_canonical_aa = [x for x in char_seq if x not in LIST_CANONICAL_AA]
        df["sequence"] = [re.sub(f'[{"".join(list_non_canonical_aa)}]', STR_AA_GAP, x) for x in df["sequence"]]
    return df


# Load scales
def _filter_scales(df_cat=None, unclassified_in=False, just_aaindex=False):
    """Filter scales for unclassified and aaindex scales"""
    list_ids_not_in_aaindex = [x for x in df_cat["scale_id"] if "LINS" in x or "KOEH" in x]
    list_ids_unclassified = [x for x, cat, sub_cat in zip(df_cat["scale_id"], df_cat["category"], df_cat["subcategory"])
                             if "Unclassified" in sub_cat or cat == "Others"]
    list_ids_to_exclude = []
    if not unclassified_in:
        list_ids_to_exclude.extend(list_ids_unclassified)
    if just_aaindex:
        list_ids_to_exclude.extend(list_ids_not_in_aaindex)
    df_cat = df_cat[~df_cat["scale_id"].isin(list_ids_to_exclude)]
    return df_cat


def load_scales(name="scales", unclassified_in=False, just_aaindex=False, missing_values_in=False):
    """Load amino acid scales or scale classification.
    :arg name: name of dataset to load {'scales', 'scales_raw', 'scale_classification'}
    :arg unclassified_in: boolean weather unclassified scales should be included (Others category counts as unclassified)
    :arg just_aaindex: boolean weather just scales provided from AAindex should be given
    :arg missing_values_in: boolean weather scales from AAindex with missing values should be included
        (onl if just_aaindex==True)
    :return df: dataframe for selected dataset
    """

    list_datasets = [ut.STR_SCALES, ut.STR_SCALES_RAW, ut.STR_SCALE_CAT]
    if name not in list_datasets:
        raise ValueError(f"'name' ({name}) is not valid. Choose one of following: {list_datasets}")
    df_cat = pd.read_excel(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
    df_cat = _filter_scales(df_cat=df_cat, unclassified_in=unclassified_in, just_aaindex=just_aaindex)
    if name == ut.STR_SCALE_CAT:
        return df_cat
    elif missing_values_in:
        if name == ut.STR_SCALES_RAW:
            return pd.read_excel(ut.FOLDER_DATA + name + ".xlsx", sheet_name="raw", index_col=0)
        else:
            raise ValueError(f"'missing_values_in' works just if '{ut.STR_SCALES_RAW}' is selected for 'name' ({name})")
    df = pd.read_excel(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
    # Filter scales
    df = df[[x for x in list(df) if x in list(df_cat["scale_id"])]]
    return df

