"""Thumbnail for Protocol 1: a clean CPP signature heatmap (DOM_GSEC).

Pipeline: SequenceFeature.get_df_parts -> CPP(df_parts).run -> TreeModel.fit +
add_feat_importance, then CPPPlot().heatmap. We collapse to scale *category*
(few readable rows) for a clean, legible gallery tile. The scale-category
legend is removed because the colored row labels on the left already carry
that information, leaving a clean colorbar centered at the bottom.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aaanalysis as aa

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol1.png"


def main():
    aa.options["verbose"] = False

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

    # Plot: collapse to category -> few large, readable rows
    aa.plot_settings(font_scale=1.2, weight_bold=False)
    cpp_plot = aa.CPPPlot()
    fig, ax = cpp_plot.heatmap(
        df_feat=df_feat,
        col_cat="category",
        figsize=(7, 7),
        fontsize_labels=15,
        fontsize_tmd_jmd=16,
        grid_linewidth=0.01,
        grid_linecolor="lightgrey",
        cbar_xywh=(0.4, 0.085, 0.22, 0.026),
    )

    # Drop the redundant scale-category legend (rows are already labeled+colored)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    fig.set_size_inches(7, 7)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
