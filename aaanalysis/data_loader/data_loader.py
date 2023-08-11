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
LIST_SCALES = [ut.STR_SCALES, ut.STR_SCALES_RAW]
LIST_DATASETS = LIST_SCALES + [ut.STR_SCALE_CAT, ut.STR_SCALES_PC, ut.STR_TOP60, ut.STR_TOP60_EVAL]


# II Main Functions
def load_dataset(name="INFO", n=None, non_canonical_aa_as_gaps=False, min_len=None, max_len=None):
    """Load one of following protein sequence benchmarking datasets:
    Load general information about datasets by 'name'='INFO
    :arg name: name of dataset ('Dataset' column)
    :arg n: number of proteins per class (if None, whole dataset will be returned)
    :arg non_canonical_aa_as_gaps: boolean whether non-canonical amino acid should be replaced by gap symbol
    :arg min_len: minimum length of sequences used for filtering
    :arg max_len: maximum length of sequences used for filtering
    :return df: dataframe with selected dataset
    """
    ut.check_non_negative_number(name="n", val=n, accept_none=True)
    ut.check_non_negative_number(name="min_len", val=min_len, accept_none=True)
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
        mask = [len(x) >= min_len for x in df[ut.COL_SEQ]]
        df = df[mask]
    if max_len is not None:
        mask = [len(x) <= max_len for x in df[ut.COL_SEQ]]
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
        df[ut.COL_SEQ] = [re.sub(f'[{"".join(list_non_canonical_aa)}]', STR_AA_GAP, x) for x in df[ut.COL_SEQ]]
    return df


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
    """Load amino acid scales or scale classification.
    :arg name: name of dataset to load {'scales', 'scales_raw', 'scale_classification',
        'scales_pc', 'top60', 'top60_eval'}
    :arg unclassified_in: boolean weather unclassified scales should be included ('Others' category counts as unclassified)
        (only if name='scales', 'scales_raw', or 'scale_classification')
    :arg just_aaindex: boolean weather just scales provided from AAindex should be given
        (only if name='scales', 'scales_raw', or 'scale_classification')
    :return df: dataframe for selected dataset
    """
    if name not in LIST_DATASETS:
        raise ValueError(f"'name' ({name}) is not valid. Choose one of following: {LIST_DATASETS}")
    # Load data
    df_cat = pd.read_excel(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
    df_cat = _filter_scales(df_cat=df_cat, unclassified_in=unclassified_in, just_aaindex=just_aaindex)
    if name == ut.STR_SCALE_CAT:
        return df_cat
    df = pd.read_excel(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
    # Filter scales
    if name in LIST_SCALES:
        df = df[[x for x in list(df) if x in list(df_cat[ut.COL_SCALE_ID])]]
    return df
