"""Thumbnail for Protocol 1: the canonical CPP feature map (DOM_GSEC).

Renders the SAME headline figure as the protocol notebook
(protocol1_cpp_signature.ipynb) and the CPP tutorial (tutorial3c_cpp):
the CPP feature map. Pipeline: SequenceFeature.get_df_parts ->
CPP(df_parts).run -> TreeModel.fit + add_feat_importance ->
CPPPlot().feature_map. Saved as a clean square tile.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aaanalysis as aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thumb_utils import save_square  # noqa: E402

# Write into the docs tree of whichever checkout this script lives in (repo root
# or a git worktree), so a worktree render never clobbers the main checkout's
# tracked thumbnail. thumb1.py sits at docs/source/_artwork/thumb_scripts/, so
# parents[2] is docs/source/.
OUT = str(Path(__file__).resolve().parents[2] / "_static" / "img" / "thumbs" / "protocol1.png")


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

    # Canonical CPP feature map with fonts bumped a touch for gallery legibility.
    #
    # The three bottom legends (Scale-category | "Feature value" colorbar |
    # feature-importance squares) must land in ONE horizontal row. feature_map
    # composes that bottom furniture itself and, on the fixed-figsize path, lays
    # the three items out as an aligned row whose fit is decided by the FIGURE
    # WIDTH, not by legend_imp_xy / cbar_xywh (those are re-anchored by the
    # auto-alignment and so have no effect here). At the old square 7x7 the wide
    # centered "Feature value / substrate - non-substrate" colorbar label left no
    # room on the right, so the importance-squares legend dropped to a second row
    # BELOW the colorbar. Widening to 8 inches gives the right edge enough room to
    # keep all three on one line (8 is the minimum width that clears it here);
    # height stays 7 so the tile stays close to square.
    aa.plot_settings(font_scale=0.65, weight_bold=False)
    cpp_plot = aa.CPPPlot()
    fig, ax = cpp_plot.feature_map(
        df_feat=df_feat,
        name_test="substrate",
        name_ref="non-substrate",
        figsize=(8, 7),
        fontsize_titles=12,
        fontsize_labels=13,
        fontsize_annotations=12,
        fontsize_imp_bar=10,
        fontsize_tmd_jmd=13,
    )

    fig.set_size_inches(8, 7)
    save_square(OUT)


if __name__ == "__main__":
    main()
