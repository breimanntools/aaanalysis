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
        return tmm.add_feat_importance(df_feat=dff), Xm

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
    df_top = df_feat.sort_values("feat_importance", ascending=False).reset_index(drop=True)
    top_feature = df_top["feature"][0]
    aa.plot_settings(font_scale=0.7)
    cpp_plot.feature(feature=top_feature, df_seq=df_seq, labels=labels,
                     name_test="substrates", name_ref="non-subs.")
    plt.title(f"{top_feature}\n({df_top['subcategory'][0]})", fontsize=8)
    _save("feature")

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

    # SHAP analysis on the SIMPLIFIED features — fuzzy-labeling demo: APP (P05067)
    # gets a SOFT prediction-score label of 0.6 (not a hard 1), and the cpp.profile
    # + per-protein SHAP feature map are derived from it.
    entry = "P05067"
    pos = list(df_seq["entry"]).index(entry)
    args_seq = {k + "_seq": v for k, v in df_parts_core.loc[entry].to_dict().items()}
    labels_fuzzy = [float(v) for v in labels]
    labels_fuzzy[pos] = 0.6
    sm = aa.ShapModel(verbose=False)
    sm.fit(X_s, labels=labels_fuzzy, fuzzy_labeling=True)
    df_feat_sh = sm.add_sample_mean_dif(X_s, labels=labels_fuzzy, df_feat=df_feat_s,
                                        sample_positions=pos, names="APP")
    df_feat_sh = sm.add_feat_impact(df_feat=df_feat_sh, sample_positions=pos, names="APP")
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
