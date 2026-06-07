"""Thumbnail for Protocol 1: the canonical CPP feature map (DOM_GSEC).

Renders the SAME headline figure as the protocol notebook
(protocol1_cpp_signature.ipynb) and the CPP tutorial (tutorial3c_cpp):
the CPP feature map. Pipeline: SequenceFeature.get_df_parts ->
CPP(df_parts).run -> TreeModel.fit + add_feat_importance ->
CPPPlot().feature_map. Saved as a clean square tile.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aaanalysis as aa

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol1.png"


def main():
    aa.options["verbose"] = False
    aa.options["random_state"] = 42

    # Small fixture
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()

    # Features
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    cpp = aa.CPP(df_parts=df_parts)
    df_feat = cpp.run(labels=labels, n_filter=50, n_jobs=1)

    # Feature importance via TreeModel
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)
    tm = aa.TreeModel(random_state=42)
    tm = tm.fit(X, labels=labels)
    df_feat = tm.add_feat_importance(df_feat=df_feat)

    # Canonical CPP feature map, sized as a square thumbnail with fonts
    # bumped a touch for gallery legibility.
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot = aa.CPPPlot()
    fig, ax = cpp_plot.feature_map(
        df_feat=df_feat,
        name_test="substrate",
        name_ref="non-substrate",
        figsize=(7, 7),
        fontsize_titles=12,
        fontsize_labels=13,
        fontsize_annotations=12,
        fontsize_imp_bar=10,
        fontsize_tmd_jmd=13,
    )

    fig.set_size_inches(7, 7)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
