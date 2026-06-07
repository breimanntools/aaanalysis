"""Protocol 9 gallery thumbnail: SHAP-style top feature contributions.

Horizontal bar of the top-importance CPP features, coloured by direction
(red = higher in test, blue = higher in reference).
"""
import matplotlib.pyplot as plt
import aaanalysis as aa

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol9.png"

COLOR_POS = "#9D2B39"  # higher in test
COLOR_NEG = "#326599"  # higher in reference


def main():
    aa.options["verbose"] = False
    aa.options["random_state"] = 42

    # --- Build signature + group-level feature importance --------------
    df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC")
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    X = sf.feature_matrix(df_parts=df_parts, features=df_feat["feature"])
    tm = aa.TreeModel(verbose=False, random_state=42)
    tm = tm.fit(X, labels=labels)
    df_feat = tm.add_feat_importance(df_feat=df_feat, drop=True)

    # --- Aggregate importance per subcategory (unique, readable labels) -
    agg = df_feat.groupby("subcategory", as_index=False).agg(
        imp=("feat_importance", "sum"),
        md=("mean_dif", "mean"),
    )
    top = agg.sort_values("imp", ascending=False).head(10)
    top = top.iloc[::-1]  # so largest plots at the top of a barh
    labels_y = [str(s) for s in top["subcategory"]]
    vals = top["imp"].to_numpy()
    colors = [COLOR_POS if d > 0 else COLOR_NEG for d in top["md"]]

    # --- Plot ----------------------------------------------------------
    aa.plot_settings(font_scale=1.2, weight_bold=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    y = range(len(vals))
    ax.barh(list(y), vals, color=colors, edgecolor="white", height=0.78)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels_y)
    ax.set_xlabel("Feature importance (%)")
    ax.set_title("Top feature contributions", weight="bold", pad=14)
    ax.set_xlim(0, vals.max() * 1.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Clean two-entry legend in the free lower-right corner (smallest bars)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_POS),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NEG),
    ]
    ax.legend(handles, ["higher in test", "higher in reference"],
              loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="lightgrey", fontsize=12.5, handlelength=1.1,
              handleheight=1.1, borderpad=0.6)

    fig.set_size_inches(7, 7)
    plt.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
