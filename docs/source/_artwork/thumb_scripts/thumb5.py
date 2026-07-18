"""Thumbnail for Protocol 5: Engineer features.

Stacked bar of the engineered Arm A feature space: counts per Part (x-axis)
stacked by Split type (Segment / Pattern / PeriodicPattern).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thumb_utils import save_square  # noqa: E402

# Write into the docs tree of whichever checkout this script lives in (repo root
# or a git worktree), so a worktree render never clobbers the main checkout's
# tracked thumbnail. thumb5.py sits at docs/source/_artwork/thumb_scripts/, so
# parents[2] is docs/source/.
OUT = str(Path(__file__).resolve().parents[2] / "_static" / "img" / "thumbs" / "protocol5.png")

aa.options["verbose"] = False
aa.options["random_state"] = 42

# --- Build the Arm A feature enumeration (small, offline) --------------------
df_seq = aa.load_dataset(name="DOM_GSEC", n=10)
sf = aa.SequenceFeature()
split_kws = sf.get_split_kws()
df_scales = aa.load_scales()
small_scales = list(df_scales.columns[:3])
features = sf.get_features(split_kws=split_kws, list_scales=small_scales)

# --- Tally features by Part (x) and Split type (stack) -----------------------
parts_order = ["TMD", "JMD_N_TMD_N", "TMD_C_JMD_C"]
part_labels = ["TMD", "JMD_N\nTMD_N", "TMD_C\nJMD_C"]
split_types = list(split_kws)  # ['Segment', 'Pattern', 'PeriodicPattern']
counts = {st: [0] * len(parts_order) for st in split_types}
for feat in features:
    tokens = feat.split("-")
    part, split = tokens[0], tokens[1]
    st = split.split("(")[0]
    if part in parts_order and st in counts:
        counts[st][parts_order.index(part)] += 1

# --- Plot --------------------------------------------------------------------
aa.plot_settings(font_scale=1.35, weight_bold=False)
colors = aa.plot_get_clist(n_colors=len(split_types))

fig, ax = plt.subplots()
x = range(len(parts_order))
bottom = [0] * len(parts_order)
for st, color in zip(split_types, colors):
    ax.bar(x, counts[st], bottom=bottom, label=st, color=color,
           width=0.62, edgecolor="white", linewidth=0.8)
    bottom = [b + c for b, c in zip(bottom, counts[st])]

total = max(bottom)
ax.set_xticks(list(x))
ax.set_xticklabels(part_labels)
ax.set_ylim(0, total * 1.30)
ax.set_xlabel("Part")
ax.set_ylabel("Number of features")
ax.set_title("Engineered feature space", weight="bold", pad=10)
sns.despine()

# Horizontal legend in the headroom band above the bars (no occlusion).
leg = ax.legend(title="Split type", loc="upper center",
                bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                handlelength=1.0, handletextpad=0.4, columnspacing=1.1,
                borderaxespad=0.3, fontsize=15, title_fontsize=16)
leg._legend_box.align = "center"

fig.set_size_inches(7, 7)
plt.tight_layout()
save_square(OUT)
