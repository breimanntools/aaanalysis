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
# Tutorial settings (tutorial3c_cpp): DEFAULT parts + DEFAULT JMD flanks
# (jmd_n_len = jmd_c_len = 10) so the feature map spans 40 positions, and a
# redundancy-reduced set of 100 scales (AAclust medoids) — matching the tutorial.
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
    # DEFAULT parts (tmd · jmd_n · jmd_c, jmd=10) — tutorial settings
    df_parts = sf.get_df_parts(df_seq=df_seq)
    df_parts_core = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd", "jmd_n", "jmd_c"])
    # redundancy-reduced set of 100 scales (AAclust medoids) — tutorial
    df_scales = aa.load_scales()
    _sel = aa.AAclust().fit(np.array(df_scales).T, names=list(df_scales),
                            n_clusters=100).medoid_names_
    df_scales = df_scales[_sel]

    def _augment(dff):
        Xm = sf.feature_matrix(features=dff["feature"], df_parts=df_parts)
        tmm = aa.TreeModel(verbose=False); tmm.fit(Xm, labels=labels)
        return tmm.add_feat_importance(df_feat=dff, sort=True), Xm

    cpp = aa.CPP(df_parts=df_parts, df_scales=df_scales, verbose=False)
    df_feat0 = cpp.run(labels=labels, n_filter=100, n_jobs=1)
    # FULL = tutorial (un-simplified); SIMPLIFIED = swapped to interpretable scales
    df_feat, X = _augment(df_feat0)
    df_feat_s, X_s = _augment(cpp.simplify(df_feat=df_feat0, labels=labels))

    # AAlogo: substrate (label_test=1, the CPP "test set") sequence logo with a
    # bits information bar on top — makes the dataset clear
    n_test = int(sum(v == 1 for v in labels))
    aa.plot_settings(font_scale=0.7)
    # aal_kws computes df_logo + the bits bar internally (skips the manual AAlogo step)
    aa.AAlogoPlot().single_logo(
        aal_kws=dict(df_parts=df_parts_core, labels=labels,
                     label_test=1, tmd_len=TMD_LEN),
        name_data=f"Test set: substrates (n={n_test})")
    _save("logo")

    # group-level CPP feature maps: FULL (tutorial) and SIMPLIFIED
    cpp_plot = aa.CPPPlot()
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot.feature_map(df_feat=df_feat); _save("feature_map")
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot.feature_map(df_feat=df_feat_s); _save("feature_map_simplified")

    # CPP ranking: top discriminative features by effect size (mean_dif) + importance
    aa.plot_settings(font_scale=0.6, weight_bold=False)
    cpp_plot.ranking(df_feat=df_feat, n_top=15, rank=True, tmd_len=TMD_LEN,
                     name_test="substrates", name_ref="non-subs.")
    _save("ranking")

    # CPP feature: REF vs TEST value distribution of the single top feature
    # (df_feat is already ranked most-important-first via add_feat_importance(sort=True),
    #  so feat_rank=1 selects the top feature straight from df_feat)
    aa.plot_settings(font_scale=0.7)
    cpp_plot.feature(feature=df_feat, feat_rank=1, df_seq=df_seq, labels=labels,
                     name_test="substrates", name_ref="non-subs.")
    plt.title(f"{df_feat['feature'].iloc[0]}\n({df_feat['subcategory'].iloc[0]})", fontsize=8)
    _save("feature")

    # AAclust: reduce the full scale set via select_scales, plot cluster centers.
    # select_scales fits internally, so aac.labels_ stays available for the plot.
    df_scales_full = aa.load_scales()
    aac = aa.AAclust()
    aac.select_scales(df_scales_full, n_clusters=10)
    aa.plot_settings()
    aa.AAclustPlot().centers(np.array(df_scales_full).T, labels=aac.labels_); _save("centers")

    # dPULearn PCA: DOM_GSEC ships 1/0; label_unl=0 treats 0 as the unlabeled pool.
    # Mine only n_pos//2 reliable negatives so the remaining unlabelled (grey) stay
    # visible. Output labels_ are always 1 (pos) / 0 (rel-neg) / 2 (unl).
    n_pos = sum(v == 1 for v in labels)
    dpul = aa.dPULearn(verbose=False)
    dpul.fit(X, labels=labels, label_unl=0, n_neg=n_pos // 2)
    aa.plot_settings(font_scale=0.8)
    # use the package's canonical sample colours (COLOR_REL_NEG gold, COLOR_POS
    # green, COLOR_UNL grey) — the dPULearnPlot.pca default — so the figure and
    # the cheat-sheet "Groups" chips agree with what the library actually plots.
    aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=dpul.labels_,
                          names=["Reliable negatives", "Positives", "Unlabeled"])
    _save("pca")

    # SHAP analysis on the SIMPLIFIED features — fuzzy-labeling demo: APP (P05067)
    # gets a SOFT prediction-score label of 0.6 (not a hard 1), and the cpp.profile
    # + per-protein SHAP feature map are derived from it.
    entry = "P05067"
    # accession-based interface (#129/#158): entry-keyed fuzzy_labels (no manual
    # fuzzy vector), select the sample by name via samples=+df_seq, and slice the
    # per-protein parts with SequenceFeature.get_args_seq — same output as before.
    args_seq = sf.get_args_seq(df_seq=df_seq, sample=entry)
    sm = aa.ShapModel(verbose=False)
    sm.fit(X_s, labels=labels, df_seq=df_seq, fuzzy_labels={entry: 0.6})
    df_feat_sh = sm.add_sample_mean_dif(X_s, labels=labels, df_feat=df_feat_s,
                                        df_seq=df_seq, samples=entry, names="APP")
    df_feat_sh = sm.add_feat_impact(df_feat=df_feat_sh, df_seq=df_seq,
                                    samples=entry, names="APP")
    aa.plot_settings(font_scale=0.6, weight_bold=False)
    cpp_plot.profile(df_feat=df_feat_sh, col_imp="feat_impact_APP", shap_plot=True,
                     tmd_len=TMD_LEN, **args_seq)
    _save("shap_profile")
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot.feature_map(df_feat=df_feat_sh, col_val="mean_dif_APP",
                         col_imp="feat_impact_APP", shap_plot=True,
                         name_test="APP", **args_seq)
    _save("feature_map_shap")


if __name__ == "__main__":
    print("[cheat-sheet] generating example plots ->", OUT)
    main()
