"""Gallery thumbnails for the Tutorials landing page (docs/source/tutorials.rst).

Renders ONE clean, square headline figure per wired tutorial, using the real
AAanalysis pipeline (small bundled fixtures), and writes ``tut<key>.png`` into
``docs/source/_static/img/thumbs/``. Each tile mirrors the tutorial's own
headline figure so the gallery shows real output, not mock-ups.

Run all:      python docs/source/_artwork/thumb_scripts/make_tutorial_thumbs.py
Run one:      python .../make_tutorial_thumbs.py t3c

The two data-loader tutorials (t2b scales, t3b sequence-feature) are
table-based and have no notebook figure; for those we render a representative
real figure from the same tool (a scales heatmap / a feature matrix heatmap).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import aaanalysis as aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thumb_utils import save_square as _save_square  # noqa: E402

# thumb_scripts -> _artwork -> source ; thumbs live under source/_static/img/thumbs
THUMBS = Path(__file__).resolve().parents[2] / "_static" / "img" / "thumbs"


def save_square(out_name):
    """Save the current figure as a uniform white square tile named ``out_name``."""
    _save_square(THUMBS / out_name)


def _cpp_feat(n=50, n_filter=50):
    """Shared: a CPP signature (df_feat) + parts for DOM_GSEC."""
    df_seq = aa.load_dataset(name="DOM_GSEC", n=n)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_parts=df_parts)
    df_feat = cpp.run(labels=labels, n_filter=n_filter, n_jobs=1)
    # Effect-size-based importance (fast surrogate for TreeModel importance) so
    # the profile / feature-map importance bars have a column to draw.
    df_feat = df_feat.copy()
    df_feat["feat_importance"] = 100 * df_feat["abs_auc"] / df_feat["abs_auc"].sum()
    return df_seq, labels, sf, df_parts, df_feat


# --- t2a: Data loader -----------------------------------------------------
def t2a():
    df_seq = aa.load_dataset(name="SEQ_CAPSID", n=100, min_len=200, max_len=800)
    lens = df_seq["sequence"].apply(len)
    aa.plot_settings(font_scale=1.2)
    plt.figure(figsize=(6, 6))
    sns.histplot(lens, binwidth=50, color="tab:gray")
    sns.despine()
    plt.xlim(0, 1500)
    plt.xlabel("Sequence length")
    plt.title("Load a benchmark dataset")
    save_square("tut2a.png")


# --- t2b: Scales loader (table tutorial -> scales heatmap) -----------------
def t2b():
    df_scales = aa.load_scales()
    sub = df_scales.iloc[:, :28]          # 20 AA x 28 scales
    aa.plot_settings(font_scale=0.9)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(sub.T, cmap="viridis", cbar_kws={"shrink": 0.6, "label": "scale value"},
                     yticklabels=False)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Amino acid scales")
    ax.set_title("Load amino acid scales")
    save_square("tut2b.png")


# --- t3a: AAclust (cluster centers) ---------------------------------------
def t3a():
    df_scales = aa.load_scales()
    X = df_scales.T
    aac = aa.AAclust()
    labels = aac.fit(X, n_clusters=5).labels_
    aa.plot_settings()
    aa.AAclustPlot().centers(df_scales=df_scales, labels=labels)
    plt.title("Cluster redundant scales (AAclust)")
    save_square("tut3a.png")


# --- t3b: SequenceFeature (table tutorial -> feature matrix heatmap) -------
def t3b():
    _, _, sf, df_parts, df_feat = _cpp_feat(n=25, n_filter=40)
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    aa.plot_settings(font_scale=1.0)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(np.asarray(X, dtype=float), cmap="viridis",
                     cbar_kws={"shrink": 0.6, "label": "feature value"},
                     xticklabels=False, yticklabels=False)
    ax.set_xlabel("CPP features")
    ax.set_ylabel("Proteins")
    ax.set_title("Sequences to a feature matrix")
    save_square("tut3b.png")


# --- t3c: CPP (feature profile) -------------------------------------------
def t3c():
    _, _, _, _, df_feat = _cpp_feat(n=50, n_filter=50)
    aa.plot_settings(font_scale=0.9)
    aa.CPPPlot().profile(df_feat=df_feat)
    plt.title("CPP signature profile")
    save_square("tut3c.png")


# --- t3d: Data representations (compact feature map) -----------------------
def t3d():
    *_, df_feat = _cpp_feat(n=25, n_filter=25)
    fs = aa.plot_gcfs()
    aa.plot_settings(font_scale=0.62, weight_bold=False)
    aa.CPPPlot().feature_map(df_feat=df_feat, figsize=(7, 7),
                             name_test="substrate", name_ref="non-substrate")
    plt.suptitle("Features from any representation", fontsize=fs + 5, weight="bold")
    save_square("tut3d.png")


# --- t4a: dPULearn (PCA of identified negatives) --------------------------
def t4a():
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    df_feat = aa.load_features(name="DOM_GSEC")
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    labels = [1 if v == 1 else 2 for v in df_seq["label"]]
    dpul = aa.dPULearn(verbose=False, random_state=42).fit(X, labels=labels, n_unl_to_neg=15)
    aa.plot_settings(font_scale=0.8)
    aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=dpul.labels_)
    plt.title("Identify reliable negatives (dPULearn)")
    save_square("tut4a.png")


# --- t5a: ShapModel (CPP-SHAP profile for APP) ----------------------------
def t5a():
    aa.options["random_state"] = 42
    df_seq = aa.load_dataset(name="DOM_GSEC")
    labels = list(df_seq["label"])
    df_feat = aa.load_features(name="DOM_GSEC")
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    sm = aa.ShapModel(verbose=False, random_state=42).fit(X, labels=labels)
    df_feat = sm.add_sample_mean_dif(X, labels=labels, df_feat=df_feat,
                                     sample_positions=0, names="APP")
    df_feat = sm.add_feat_impact(df_feat=df_feat, sample_positions=0, names="APP")
    seq_kws = sf.get_seq_kws(df_seq=df_seq, df_parts=df_parts, sample="P05067")
    aa.plot_settings(font_scale=0.9)
    aa.CPPPlot().profile(df_feat=df_feat, shap_plot=True, col_imp="feat_impact_APP", **seq_kws)
    plt.title("Explain a prediction (CPP-SHAP)")
    save_square("tut5a.png")


# --- t6: Evaluation & comparison (per-protein rank scatter) ---------------
def t6():
    rng = np.random.default_rng(1)
    list_scores = []
    for _ in range(15):
        L = int(rng.integers(50, 90))
        s = rng.random(L) * 0.7
        for p in rng.choice(L, size=int(rng.integers(1, 4)), replace=False):
            s[p] = min(1.0, s[p] + rng.uniform(0.05, 0.35))
        list_scores.append(s)
    df_rank = pd.DataFrame({
        "score": [float(s.max()) for s in list_scores],
        "group": ["substrate"] * 8 + ["non-substrate"] * 7,
    })
    aa.plot_settings()
    aa.AAPredPlot().predict_group(df_rank, kind="rank_scatter", col_group="group", thresholds=0.7)
    plt.title("Rank & evaluate under CV")
    save_square("tut6.png")


# --- t7: Protein engineering (SeqOpt Pareto front) ------------------------
def t7():
    from sklearn.ensemble import RandomForestClassifier
    df_feat = aa.load_features(name="DOM_GSEC")
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    X = np.asarray(sf.feature_matrix(features=df_feat["feature"],
                                     df_parts=sf.get_df_parts(df_seq=df_seq),
                                     df_scales=aa.load_scales()), dtype=float)
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, labels)
    wt = df_seq[df_seq["label"] == 0].iloc[[0]].reset_index(drop=True)
    objectives = [("substrate", "max", "delta_pred"), ("parsimony", "min", "n_mut")]
    seqo = aa.SeqOpt(mode="importance", model=model, target_class=1, random_state=42)
    df_pareto = seqo.run(df_seq=wt, df_feat=df_feat, objectives=objectives,
                         pop_size=40, n_gen=20, n_mut_max=6, region="tmd")
    aa.plot_settings()
    aa.SeqOptPlot().pareto_front(df_pareto=df_pareto, x="substrate", y="parsimony")
    plt.title("Optimize a sequence (SeqOpt)")
    save_square("tut7.png")


TUTS = {"t2a": t2a, "t2b": t2b, "t3a": t3a, "t3b": t3b, "t3c": t3c,
        "t3d": t3d, "t4a": t4a, "t5a": t5a, "t6": t6, "t7": t7}


def main(argv):
    aa.options["verbose"] = False
    keys = argv[1:] if len(argv) > 1 else list(TUTS)
    for k in keys:
        if k not in TUTS:
            print("unknown key:", k, "| choose from", list(TUTS)); continue
        print(f"=== rendering {k} ===")
        try:
            TUTS[k]()
        except Exception as exc:  # keep the batch going; surface the failure
            plt.close("all")
            print(f"!!! {k} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main(sys.argv)
