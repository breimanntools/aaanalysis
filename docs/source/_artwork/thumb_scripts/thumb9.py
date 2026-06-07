"""Protocol 9 gallery thumbnail: canonical CPP-SHAP heatmap.

Sample-level SHAP explanation for APP (P05067): each cell is the signed SHAP
impact of one CPP feature on APP's gamma-secretase-substrate score, anchored to
its real residue position and scale subcategory (red = pushes toward substrate,
blue = toward non-substrate). Mirrors the headline figure of protocol 9 and the
CPP-SHAP heatmap of tutorial5a.
"""
import matplotlib.pyplot as plt
import aaanalysis as aa

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol9.png"


def main():
    aa.options["verbose"] = False
    aa.options["random_state"] = 42

    # --- Signature + feature matrix -----------------------------------
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC")
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    pos_app = list(df_seq["entry"]).index("P05067")

    # --- Fit ShapModel + per-sample impact for APP --------------------
    sm = aa.ShapModel(verbose=False, random_state=42)
    sm = sm.fit(X, labels=labels, n_rounds=3)
    df_feat = sm.add_sample_mean_dif(X, labels=labels, df_feat=df_feat,
                                     sample_positions=pos_app, names="APP")
    df_feat = sm.add_feat_impact(df_feat=df_feat, sample_positions=pos_app, names="APP")

    # APP's sequence parts anchor the heatmap to real residues
    _df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=["tmd", "jmd_c", "jmd_n"])
    _args = _df_parts.loc["P05067"].to_dict()
    args_seq = {key + "_seq": _args[key] for key in _args}

    # --- Canonical CPP-SHAP heatmap -----------------------------------
    fs = aa.plot_gcfs()
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot = aa.CPPPlot()
    fig, ax = cpp_plot.heatmap(df_feat=df_feat, shap_plot=True,
                               col_val="feat_impact_APP", name_test="APP",
                               figsize=(7, 7), **args_seq)
    plt.title("CPP-SHAP heatmap for APP", fontsize=fs + 5, weight="bold")

    fig.set_size_inches(7, 7)
    # Leave headroom so the title is not clipped (saved without bbox_inches="tight").
    fig.subplots_adjust(top=0.92)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
