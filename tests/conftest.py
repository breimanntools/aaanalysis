"""
This file contains shared fixtures (preloaded data) that can be used by tests.
"""
import pandas as pd
import numpy as np
import pytest

import aaanalysis as aa
from aaanalysis import SequenceFeature
import tests._utils as ut


# Valid functions
@pytest.fixture(scope="module")
def df_seq():
    return aa.load_dataset(name="DOM_GSEC_PU", n=10)


@pytest.fixture(scope="module")
def labels(df_seq):
    labels = [1 if x == "SUBEXPERT" else 0 for x in df_seq["label"]]
    return labels


@pytest.fixture(scope="module")
def df_cat():
    df_cat = aa.load_scales(name="scales_cat").head(100)
    return df_cat


@pytest.fixture(scope="module")
def df_scales():
    df_scales = aa.load_scales()
    return df_scales


@pytest.fixture(scope="module")
def df_parts(df_seq):
    sf = SequenceFeature()
    return sf.get_df_parts(df_seq=df_seq)


@pytest.fixture(scope="module")
def split_kws():
    sf = SequenceFeature()
    return sf.get_split_kws()


@pytest.fixture(scope="function")
def df_feat():
    return pd.read_excel(ut.FOLDER_DATA + ut.FILE_FEAT)


@pytest.fixture(scope="module")
def df_feat_module_scope():
    return pd.read_excel(ut.FOLDER_DATA + ut.FILE_FEAT)


@pytest.fixture(scope="module")
def list_parts():
    list_parts = [["tmd_jmd"], ["tmd"], ["tmd_e"], ["tmd_e", "tmd_c_jmd_c", "jmd_n_tmd_n"],
                  ["tmd", "tmd_e", "tmd_c_jmd_c", "jmd_n_tmd_n"]]
    return list_parts


@pytest.fixture(scope="module")
def list_splits():
    list_splits = ["Segment(5,7)", "Segment(1,1)", "Pattern(C,1,2)", "Pattern(N,1)", "Pattern(N,1,4,10)",
                   "PeriodicPattern(N,i+2/3,1)", "PeriodicPattern(N,i+4/2,5)", "PeriodicPattern(C,i+1/5,1)"]
    return list_splits


# Wrong
@pytest.fixture(params=[pd.DataFrame(), 2, "s", dict])
def wrong_df(request):
    return request.param


# Corrupted input using parametrized fixtures
def _corrupted_list_parts():
    list_parts = [["tmd_md"], ["TMD"], ["tmd_E"], ["md_e", "tmd_c_jmd_n", "jmd_n_tmd_a"],
                  ["tmd", "tmd_e", "tmd_c_jmd_c", "jmd_c_tmd_n"]]
    return list_parts


@pytest.fixture(params=_corrupted_list_parts())
def corrupted_list_parts(request):
    return request.param


def _corrupted_list_splits():
    list_splits = ["Segment(5,2)", "segment(1,1)", "Pttern(C,1,2)", "Pattern(A,1)", "Pattern(N,25,4,10)",
                   "PeriodicPattern(N,i2/3,1)", "PeriodicPattern(N,i+4/2)", "Periodicattern(C,i+1/5,1)"]
    return list_splits


@pytest.fixture(params=_corrupted_list_splits())
def corrupted_list_splits(request):
    return request.param


def _corrupted_df_seq():
    df_seq = aa.load_dataset(name="DOM_GSEC_PU", n=10)
    dfa = df_seq.drop(["sequence"], axis=1)
    df1 = dfa.drop(["tmd"], axis=1)
    df2 = dfa.copy()
    df2.iloc[:1, df2.columns.get_loc("tmd")] = np.nan
    df3 = df2.copy()
    df3["tmd"] = 4
    df4 = dfa.copy()
    df4["tmd"] = np.nan
    dfb = df_seq.drop(["tmd"], axis=1)
    df5 = dfb.copy()
    df5["sequence"] = 4
    df6 = dfb.copy()
    df6["sequence"] = np.nan
    return [df1, df2, df3, df4, df5, df6]


@pytest.fixture(params=_corrupted_df_seq())
def corrupted_df_seq(request):
    return request.param


def _corrupted_df_scales():
    df_scales = aa.load_scales()
    scales = list(df_scales)
    df1 = df_scales.copy()
    df1[scales[0]] = "a"
    df2 = pd.concat([df_scales, df_scales], axis=0)
    df3 = pd.concat([df_scales, df_scales], axis=1)
    df4 = df_scales.copy()
    df4[scales[1]] = [np.NaN] + [0.5] * 19
    df5 = df_scales.copy()
    df5.reset_index(inplace=True)
    df6 = df_scales.copy()
    df6.index = ["A"] * 20
    return [df1, df2, df3, df4, df5, df6]


@pytest.fixture(params=_corrupted_df_scales())
def corrupted_df_scales(request):
    return request.param


def _corrupted_split_kws():
    sf = SequenceFeature()
    split_kws = sf.get_split_kws()
    kws1 = split_kws.copy()
    kws1["test"] = 1
    kws2 = split_kws.copy()
    kws2["segment"] = kws2["Segment"]
    kws2.pop("Segment")
    kws3 = split_kws.copy()
    kws3["Pattern"]["steps"] = [-1, 3]
    kws4 = split_kws.copy()
    kws4["PeriodicPattern"]["steps"] = [0, 0, None]
    kws5 = split_kws.copy()
    kws5["Segment"]["n_split_min"] = 10
    kws5["Segment"]["n_split_max"] = 5
    return [kws1, kws2, kws3, kws4, kws5]


@pytest.fixture(params=_corrupted_split_kws())
def corrupted_split_kws(request):
    return request.param


def _corrupted_df_parts():
    df_seq = aa.load_dataset(name="DOM_GSEC")
    sf = SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq, all_parts=True)
    df1 = pd.concat([df_parts, df_parts], axis=0)
    df2 = pd.concat([df_parts, df_parts], axis=1)
    df3 = df_parts.copy()
    df3["test"] = "AAAAAAAAAAAAAAAAAAAA"
    df4 = df_parts.copy()
    df4["tmd"] = "AAAAAAAAAAAAAAAAa"
    df5 = df_parts.copy()
    df5.columns = [x.upper() for x in list(df_parts)]
    return [df1, df2, df3, df4, df5, df5]


@pytest.fixture(params=_corrupted_df_parts())
def corrupted_df_parts(request):
    return request.param


def _corrupted_labels():
    df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
    labels = df_seq["label"].to_list()
    labels_a = [str(x) for x in labels]
    labels_b = [x + 1 for x in labels]
    labels_c = labels.copy()
    labels_c[0] = np.NaN
    labels_d = labels.copy()
    labels_d[5] = "a"
    labels_e = labels.copy()
    labels_e.extend([0, 1, 0])
    labels_f = labels.copy()
    labels_f.remove(1)
    labels_g = [0] * len(labels)
    labels_h = [1] * len(labels)
    return [labels_a, labels_b, labels_c, labels_d, labels_e, labels_f, labels_g, labels_h]


@pytest.fixture(params=_corrupted_labels())
def corrupted_labels(request):
    return request.param
