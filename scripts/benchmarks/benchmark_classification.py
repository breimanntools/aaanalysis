"""
This is a script for ...
"""
import time
import pandas as pd
import numpy as np
import os
from itertools import repeat
import multiprocessing as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.cluster import AgglomerativeClustering, KMeans

import scripts._utils as ut
from aaclust import AAclust

# Settings
pd.set_option('expand_frame_repr', False)  # Single line print for pd.Dataframe


DICT_MODELS = dict(rf=RandomForestClassifier(), log_reg=LogisticRegressionCV(solver="liblinear"), svm=SVC())
LIST_MODELS = ["rf", "log_reg", "svm"]
DICT_MODELS = {x: DICT_MODELS[x] for x in LIST_MODELS}
ROUND = 3

# TODO adjust benchmarking datasets (see TODO in benchmarks) AIM: 10 proper benchmarking sets
# TODO use smaller datasets for AA tasks: use LDR validation, RBP129 for RNA binding
# TODO add datasets: TAIL, 7DisPro for Accessible surface prediction
# TODO split location into two with most positive class 'Nucleus', 'Cytoplasm', Plasma-Membrane (replace soluble with)

# I Helper Functions
def _feat_matrix(list_seq, dict_scale_vals=None):
    """"""
    if type(list_seq) is not list:
        raise ValueError("'list_seq' should be list with sequences")
    vf_scales = np.vectorize(lambda seq: np.mean([ds[aa] for aa in seq]))
    feat_matrix = np.empty([len(list_seq), len(dict_scale_vals)])
    for i, scale in enumerate(dict_scale_vals):
        ds = dict_scale_vals[scale]
        feat_matrix[:, i] = [vf_scales(seq) for seq in list_seq]
    return feat_matrix


def _get_feat_matrix(dict_scale_vals=None, list_seq=None):
    """Multi-processing to create feature matrix if more than 10 scales are used"""
    if len(dict_scale_vals) <= 10:
        return _feat_matrix(list_seq, dict_scale_vals=dict_scale_vals)
    n_processes = min([os.cpu_count(), len(dict_scale_vals)])
    feat_chunks = [list(x) for x in np.array_split(list_seq, n_processes)]
    args = zip(feat_chunks, repeat(dict_scale_vals))
    with mp.get_context("spawn").Pool(processes=n_processes) as pool:
        result = pool.starmap(_feat_matrix, args)
    feat_matrix = np.concatenate(result, axis=0)
    return feat_matrix


def _get_list_seq(df_seq=None):
    """Filter gaps, missing values and X from sequences"""
    LIST_EXCLUDE = ["X", "_", "B", "U", "!"]
    if "sequence" not in df_seq:
        raise ValueError("'sequence' must be column in 'df_seq'")
    list_seq = ["".join([aa.upper() for aa in seq if aa not in LIST_EXCLUDE]) for seq in df_seq["sequence"].to_list()]
    return list_seq

# Get scales
def _get_random_scales(df_scales=None, list_n=None, n_max=100, rounds=5):
    """"""
    n_all_scales = len(list(df_scales))
    if max(list_n) >= n_all_scales:
        raise ValueError(f"Max n ({max(list_n)}) in 'list_n' should not be higher than number of scales ({n_all_scales}")
    dict_scales = {}
    for n in list_n:
        name = f"RANDOM{n}"
        if n > n_max:
            dict_scales[name] = list(df_scales.sample(n=n, axis=1))
        else:
            for i in range(rounds):
                name_ = name + f"_ROUND{i+1}"
                dict_scales[name_] = list(df_scales.sample(n=n, axis=1))
    return dict_scales


def _get_aaclust_scales(df_scales=None, list_n=None):
    """"""
    list_models = [(AgglomerativeClustering, dict(linkage="ward"), "Agglomerative_ward"),
                   (AgglomerativeClustering, dict(linkage="average"), "Agglomerative_average")]
                   #(KMeans, dict(random_state=42), "KMeans")]
    dict_scales = {}
    n = 0
    for on_center in [True, False]:
        for model_ in list_models:
            n += 1
            #if not n > n_max:
            model, model_kwargs, model_name = model_
            aac = AAclust(model=model, model_kwargs=model_kwargs)
            args = dict(on_center=on_center, min_th=0.3, merge=True, merge_metric="euclidean")
            # TODO check if consistent with sklarn
            aac.fit(np.array(df_scales).T,  **args)
            scales = [list(df_scales)[i] for i in aac.medoid_ind_]
            name = f"AACLUST{n}"
            dict_scales[name] = scales
    for n in list_n:
        name = f"AACLUST_N{n}"
        #aac = AAclust(model=KMeans, model_kwargs=dict(random_state=42))
        if n <= 50:
            model = KMeans
            model_kwargs = dict(random_state=42)
        else:
            model = AgglomerativeClustering
            model_kwargs = dict(linkage="average")
        aac = AAclust(model=model, model_kwargs=model_kwargs)
        aac.fit(np.array(df_scales).T, n_clusters=n)
        scales = [list(df_scales)[i] for i in aac.medoid_ind_]
        dict_scales[name] = scales
    return dict_scales


def _get_goldstandard_scales(df_scales=None):
    """"""
    # Meiler et al., 2001 (Tang et al. 2020),
    # Steric parameter, Hydrophobicity, Volume, Polarizability,
    # Isoelectric point, Alpha-helix probability, Beta-sheet probability
    top7_scales = ["FAUJ880101", "FAUJ830101",  "FAUJ880103",  "CHAM820101",
                   "ZIMJ680104", "CHOP780201", "CHOP780202"]
    # Pazos et al., 2021 (Gasteiger et al., 2005; ExPasy: https://web.expasy.org/protscale/)
    # Beta-sheet (Chou & Fasman), Alpha-helix (Deleage & Roux; Chou & Fasman),
    # Coil (Deleage & Roux; Chou-Fasman parameter), Beta-turn (Levitt),
    # Polarity (Grantham), Hydrophobicity (Kyte & Doolittle), Hydrophobicity (Eisenberg), Hydrophobicity (Chothia)
    # Relative mutability, Average flexibility,  Molecular weight, Bulkiness,
    top12_scales = ["CHOP780202", "LEVM780103", "CHOP780201", "CHAM830101",
                    "GRAR740102", "KYTJ820101", "EISD840101", "CHOC760103",
                    "DAYM780201", "BHAR880101", "FASG760101", "ZIMJ680102"]
    top_all = list(df_scales)
    dict_scales = {"STANDARD7": top7_scales, "STANDARD12": top12_scales, f"STANDARD{len(top_all)}": top_all}
    return dict_scales


def _get_pca_scales(df_scales=None):
    """"""
    X = np.array(df_scales).T
    list_aa = df_scales.index.to_list()
    dict_scales = {}
    for n in [5, 10]:
        pca = PCA(n_components=n)
        pca.fit(X)
        pca_scales = pca.components_
        name = f"PC{n}"
        scales = {f"PC_{i}": dict(zip(list_aa, np.round(s, 5))) for i, s in enumerate(pca_scales)}
        dict_scales[name] = scales
    return dict_scales

# Evaluation
def _evaluation(df=None, dict_scale_vals=None, dict_models=None):
    """"""
    list_seq = _get_list_seq(df_seq=df)
    X = _get_feat_matrix(list_seq=list_seq, dict_scale_vals=dict_scale_vals)
    y = df["label"].to_list()
    print(X.shape)
    list_results = []
    for model_name in dict_models:
        model = dict_models[model_name]
        cv = cross_val_score(model, X, y, scoring="accuracy", cv=5, n_jobs=8)
        list_results.append(round(np.mean(cv), ROUND))
    list_results.append(len(dict_scale_vals))
    return list_results


# Ranking
def _score_ranking(df=None, cols_scores=None):
    """Obtain average ranking for given list of scores"""
    mean_rank = df[cols_scores].round(ROUND).rank(ascending=False).mean(axis=1).rank(method="dense")
    return mean_rank


# II Main Functions
def get_scale_sets(df_scales=None):
    """"""
    dict_scale_names = _get_goldstandard_scales(df_scales=df_scales.copy())
    dict_scale_names.update(_get_random_scales(df_scales=df_scales.copy(),
                                               list_n=[1, 3, 5, 7, 10, 20, 30, 50, 100],
                                               rounds=5))
    dict_scale_names.update(_get_aaclust_scales(df_scales=df_scales,
                                                list_n=[5, 7, 10, 12, 15, 20, 50, 75, 100]))
    dict_scale_names.update(_get_pca_scales(df_scales=df_scales))
    return dict_scale_names


def get_scales(df_scales=None):
    """"""
    df_scales = df_scales[[x for x in df_scales if "KOEH" not in x and "LINS" not in x]]
    dict_all_scales = {col: {i:v for i, v in zip(df_scales.index, df_scales[col])} for col in list(df_scales)}
    dict_scale_sets = get_scale_sets(df_scales=df_scales)
    return dict_all_scales, dict_scale_sets


def load_data(name=None, show_datasets=False):
    """"""
    LIST_DATA = ["AMYLO_SEQ", "CAPSID_SEQ", "DISULFIDE_SEQ", "SOLUBLE_SEQ",
                 "LOCATION_SEQ_MULTI", "LDR_AA", "RNABIND_AA"]
    if show_datasets:
        print(LIST_DATA)
    if name not in LIST_DATA:
        raise ValueError(f"'name' ({name}) should be one of following: {LIST_DATA}")
    FOLDER_IN = ut.FOLDER_DATA + "benchmarks" + ut.SEP
    df = pd.read_csv(FOLDER_IN + name + ".tsv", sep="\t")
    return df


def get_aa_windows(df=None, size=9):
    """"""
    list_aa = []
    list_labels = []
    n_pre = n_post = int((size-1)/2)
    for seq, labels in zip(df["sequence"], df["label"]):
        for i, aa in enumerate(seq):
            seq_pre = seq[max(i-n_pre, 0):i]
            seq_post = seq[i+1:i+n_post+1]
            aa_window = seq_pre + aa + seq_post
            list_aa.append(aa_window)
        list_labels.extend(labels.split(","))
    df = pd.DataFrame({"sequence": list_aa, "label": list_labels})
    return df


def evaluation(df=None, dict_all_scales=None, dict_scale_sets=None, dict_models=None):
    """"""
    results = []
    for scale_set_name in dict_scale_sets:
        if "PC" not in scale_set_name:
            scale_set = dict_scale_sets[scale_set_name]
            dict_scale_vals = {x: dict_all_scales[x] for x in scale_set}
        else:
            dict_scale_vals = dict_scale_sets[scale_set_name]
        list_results = _evaluation(df=df, dict_scale_vals=dict_scale_vals, dict_models=dict_models)
        results.append(list_results)
    return results


def get_ranking(results=None, dict_scale_sets=None, merge_rounds=True):
    """"""
    cols = list(DICT_MODELS.keys()) + ["n_scales"]
    df_bench = pd.DataFrame(results, columns=cols, index=dict_scale_sets.keys())
    # Merge results of rounds of randomly sampled scales
    if merge_rounds:
        dict_set_names = {x:x.split("_")[0] if "ROUND" in x else x for x in df_bench.index}
        df_bench["set_name"] = [dict_set_names[x] for x in df_bench.index.to_list()]
        df_bench = df_bench.groupby("set_name", sort=False).mean().round(ROUND)
        df_bench["n_scales"] = [int(i) for i in df_bench["n_scales"]]
    df_bench["rank"] = _score_ranking(df=df_bench, cols_scores=DICT_MODELS.keys())
    df_bench.index.name = "datasets"
    return df_bench


# III Test/Caller Functions
# Benchmark aaanalysis classification
def benchmark_seq_classification():
    """"""
    # Get scales
    df_scales = pd.read_excel(ut.FOLDER_DATA + "scales.xlsx", index_col=0)
    dict_all_scales, dict_scale_sets = get_scales(df_scales=df_scales.copy())
    # Test for different sequence based single-label protein prediction problems datasets
    list_datasets = ['AMYLO_SEQ', 'CAPSID_SEQ', 'DISULFIDE_SEQ', 'SOLUBLE_SEQ']
    for name in list_datasets:
        print(name)
        df = load_data(name=name)
        results = evaluation(df=df,
                             dict_all_scales=dict_all_scales,
                             dict_models=DICT_MODELS,
                             dict_scale_sets=dict_scale_sets)
        df_bench = get_ranking(results=results, dict_scale_sets=dict_scale_sets)
        file_out = f"AAclust_benchmark_classification_{name}.xlsx"
        df_bench.to_excel(ut.FOLDER_RESULTS + file_out)



def benchmark_seq_multi_classification():
    """"""
    #  Get scales
    df_scales = pd.read_excel(ut.FOLDER_DATA + "scales.xlsx", index_col=0)
    dict_all_scales, dict_scale_sets = get_scales(df_scales=df_scales.copy())
    # Create predictions for 3 subcellular locations with highest occurrence ['Nucleus', 'Cytoplasm', 'Plasma Membrane']
    name = 'LOCATION_SEQ_MULTI'
    df_multi = load_data(name=name)
    list_loc = [l for label in df_multi["label"] for l in label.split(",")]
    list_pos_class = pd.Series(list_loc).value_counts().head(3).index.to_list()
    array = np.empty(shape=(len(dict_scale_sets), len(DICT_MODELS) + 1, len(list_pos_class)))
    for i, pos_class in enumerate(list_pos_class):
        print(name, pos_class)
        df = df_multi.copy()
        df["label"] = [1 if pos_class in l else 0 for l in df_multi["label"]]
        results = evaluation(df=df,
                             dict_all_scales=dict_all_scales,
                             dict_models=DICT_MODELS,
                             dict_scale_sets=dict_scale_sets)

        array[:,:, i] = np.array(results)
    results = array.mean(axis=2).round(ROUND)   # Get average over all datasets
    df_bench = get_ranking(results=results, dict_scale_sets=dict_scale_sets)
    file_out = f"AAclust_benchmark_classification_{name}.xlsx"
    df_bench.to_excel(ut.FOLDER_RESULTS + file_out)


def benchmark_aa_classification():
    """"""
    # Get scales
    df_scales = pd.read_excel(ut.FOLDER_DATA + "scales.xlsx", index_col=0)
    dict_all_scales, dict_scale_sets = get_scales(df_scales=df_scales.copy())
    # Create predictions for 3 subcellular locations with highest occurrence ['Nucleus', 'Cytoplasm', 'Plasma Membrane']
    # Test for different sequence based single-label protein prediction problems datasets
    list_datasets = ['RNABIND_AA', 'LDR_AA']
    for name in list_datasets:
        print(name)
        df = load_data(name=name)
        df = get_aa_windows(df=df, size=13)
        results = evaluation(df=df,
                             dict_all_scales=dict_all_scales,
                             dict_models=DICT_MODELS,
                             dict_scale_sets=dict_scale_sets)
        file_out = f"AAclust_benchmark_classification_{name}.xlsx"
        df_bench = get_ranking(results=results, dict_scale_sets=dict_scale_sets)
        df_bench.to_excel(ut.FOLDER_RESULTS + file_out)

# IV Main
def main():
    t0 = time.time()
    benchmark_seq_classification()
    benchmark_seq_multi_classification()
    benchmark_aa_classification()
    t1 = time.time()
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
