"""Thumbnail for Protocol 6: Compositional vs positional CPP signatures.

Grouped bar chart of the number of selected features per split type for the
Compositional vs the Positional feature-engineering recipe.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thumb_utils import save_square  # noqa: E402

# Write into the docs tree of whichever checkout this script lives in (repo root
# or a git worktree), so a worktree render never clobbers the main checkout's
# tracked thumbnail. thumb6.py sits at docs/source/_artwork/thumb_scripts/, so
# parents[2] is docs/source/.
OUT = str(Path(__file__).resolve().parents[2] / "_static" / "img" / "thumbs" / "protocol6.png")

aa.options["verbose"] = False
aa.options["random_state"] = 42

# --- Small fixture: 50 per class -> 100 sequences ---
df_seq = aa.load_dataset(name="DOM_GSEC", n=50)
labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)

# Recipe A: COMPOSITIONAL (one whole-part Segment per part)
split_kws_comp = sf.get_split_kws(split_types="Segment", n_split_min=1, n_split_max=1)
df_feat_comp = aa.CPP(df_parts=df_parts, split_kws=split_kws_comp).run(
    labels=labels, n_filter=20, n_jobs=1)

# Recipe B: POSITIONAL (sub-segments + Pattern + PeriodicPattern)
split_kws_pos = sf.get_split_kws(
    split_types=["Segment", "Pattern", "PeriodicPattern"],
    n_split_min=1, n_split_max=5, steps_pattern=[3, 4], steps_periodicpattern=[3, 4])
df_feat_pos = aa.CPP(df_parts=df_parts, split_kws=split_kws_pos).run(
    labels=labels, n_filter=30, n_jobs=1)


def split_type_counts(df_feat):
    return df_feat["feature"].str.split("-").str[1].str.split("(").str[0].value_counts()


order = ["Segment", "Pattern", "PeriodicPattern"]
counts_comp = split_type_counts(df_feat_comp).reindex(order, fill_value=0).values
counts_pos = split_type_counts(df_feat_pos).reindex(order, fill_value=0).values

# --- Plot ---
aa.plot_settings(font_scale=1.45, weight_bold=False)
colors = aa.plot_get_clist(n_colors=2)
fig, ax = plt.subplots()
x = range(len(order))
w = 0.43
ax.bar([i - w / 2 for i in x], counts_comp, width=w, label="Compositional",
       color=colors[0], edgecolor="white", zorder=3)
ax.bar([i + w / 2 for i in x], counts_pos, width=w, label="Positional",
       color=colors[1], edgecolor="white", zorder=3)

ax.set_xticks(list(x))
ax.set_xticklabels(["Segment", "Pattern", "Periodic\nPattern"])
ax.set_ylabel("# selected features")
ax.set_title("Selected features by split type")
ax.set_ylim(0, max(counts_comp.max(), counts_pos.max()) * 1.28)
ax.grid(axis="y", color="0.85", zorder=0)
ax.tick_params(axis="x", length=0)
sns.despine()
aa.plot_legend(ax=ax, dict_color={"Compositional": colors[0], "Positional": colors[1]},
               n_cols=1, x=0.33, y=1.0, marker="s", marker_size=14)

fig.set_size_inches(7, 7)
plt.tight_layout()
save_square(OUT)
