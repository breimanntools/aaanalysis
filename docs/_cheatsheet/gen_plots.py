"""This is a script for generating the example plot images used in the cheat sheet.

Run with the project's dev environment (it needs the full ``aaanalysis`` stack +
matplotlib, which the lightweight ``.buildenv`` does not have)::

    COVERAGE_CORE=sysmon .venv/bin/python docs/_cheatsheet/gen_plots.py

It writes PNGs into ``docs/source/_static/cs_plots/`` (committed); the cheat-sheet
build then embeds them in the "Example Outputs" gallery. Regenerate only when the
figures should change — the build itself does not call this.
"""
import os
import warnings
from pathlib import Path

import numpy as np  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import aaanalysis as aa  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "source" / "_static" / "cs_plots"
OUT.mkdir(parents=True, exist_ok=True)

aa.options["verbose"] = False
aa.options["random_state"] = 42
# γ-secretase TMD model: ~20-aa transmembrane domain with short JMD flanks.
# JMD=6 is the smallest flank for which the jmd_n_tmd_n / tmd_c_jmd_c parts stay
# >= 15 aa (the shortest DOM_GSEC TMD is 18), so DEFAULT CPP split settings apply.
JMD_LEN = 6
aa.options["jmd_n_len"] = JMD_LEN
aa.options["jmd_c_len"] = JMD_LEN
TMD_LEN = 20


def _save(name):
    plt.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close("all")
    print(f"  wrote {name}.png")


def main():
    # full DOM_GSEC so the APP substrate (P05067) is present for the SHAP example
    df_seq = aa.load_dataset(name="DOM_GSEC")
    labels = df_seq["label"].to_list()
    sf = aa.SequenceFeature()
    # extended parts for CPP -> long enough for the DEFAULT split grid
    df_parts = sf.get_df_parts(df_seq=df_seq,
                               list_parts=["tmd", "jmd_n_tmd_n", "tmd_c_jmd_c"])
    # core parts for the logo (full TMD) + per-protein sequence overlay
    df_parts_core = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd", "jmd_n", "jmd_c"])

    cpp = aa.CPP(df_parts=df_parts, verbose=False)
    df_feat = cpp.run(labels=labels, n_filter=40, n_jobs=1,
                      tmd_len=TMD_LEN, jmd_n_len=JMD_LEN, jmd_c_len=JMD_LEN)
    # swap scales for more interpretable correlated ones (CPP.simplify demo)
    df_feat = cpp.simplify(df_feat=df_feat, labels=labels)
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    tm = aa.TreeModel(verbose=False)
    tm.fit(X, labels=labels)
    df_feat = tm.add_feat_importance(df_feat=df_feat)

    # AAlogo: substrate (label_test=1, the CPP "test set") sequence logo with a
    # bits information bar on top — makes the dataset clear
    n_test = int(sum(v == 1 for v in labels))
    aal = aa.AAlogo()
    df_logo = aal.get_df_logo(df_parts=df_parts_core, labels=labels, label_test=1, tmd_len=TMD_LEN)
    df_info = aal.get_df_logo_info(df_parts=df_parts_core, labels=labels, label_test=1, tmd_len=TMD_LEN)
    aa.plot_settings(font_scale=0.7)
    aa.AAlogoPlot().single_logo(df_logo=df_logo, df_logo_info=df_info,
                                name_data=f"Test set: substrates (n={n_test})")
    _save("logo")

    # group-level CPP feature map (tutorial conventions)
    cpp_plot = aa.CPPPlot()
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot.feature_map(df_feat=df_feat); _save("feature_map")

    # AAclust: cluster the scale set, plot cluster centers in PCA space
    X_scales = np.array(df_scales := aa.load_scales()).T
    aac = aa.AAclust()
    aac.fit(X_scales, names=list(df_scales), n_clusters=10)
    aa.plot_settings()
    aa.AAclustPlot().centers(X_scales, labels=aac.labels_); _save("centers")

    # dPULearn PCA: positives (1) + unlabelled (2) -> reliable negatives (0).
    # Convert only part of the unlabelled pool so unlabelled (grey) remain visible.
    labels_pu = [1 if v == 1 else 2 for v in labels]
    n_pos = sum(v == 1 for v in labels)
    dpul = aa.dPULearn(verbose=False)
    dpul.fit(X, labels=labels_pu, n_unl_to_neg=n_pos // 2)
    aa.plot_settings(font_scale=0.8)
    # match the cheat-sheet PU-label chips: 0 rel-neg=magenta, 1 pos=green, 2 unl=grey
    aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=dpul.labels_,
                          colors=["#B0398E", "#4FA24B", "#8A8A8A"],
                          names=["Reliable negatives", "Positives", "Unlabeled"])
    _save("pca")

    # per-protein CPP-SHAP feature map for a BORDERLINE prediction — LRP6 (O75581),
    # an honest out-of-fold substrate score of ~0.60 (interesting to explain).
    entry = "O75581"
    se = aa.ShapModel(verbose=False)
    se.fit(X, labels=labels, fuzzy_labeling=True)
    pos = list(df_seq["entry"]).index(entry)
    df_feat = se.add_sample_mean_dif(X, labels=labels, df_feat=df_feat,
                                     sample_positions=pos, names="LRP6")
    df_feat = se.add_feat_impact(df_feat=df_feat, sample_positions=pos, names="LRP6")
    args_seq = {k + "_seq": v for k, v in df_parts_core.loc[entry].to_dict().items()}
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot.feature_map(df_feat=df_feat, col_val="mean_dif_LRP6",
                         col_imp="feat_impact_LRP6", shap_plot=True,
                         name_test="LRP6", **args_seq)
    _save("feature_map_shap")


if __name__ == "__main__":
    print("[cheat-sheet] generating example plots ->", OUT)
    main()
