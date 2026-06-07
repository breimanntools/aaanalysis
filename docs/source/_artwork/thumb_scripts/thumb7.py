"""Thumbnail for Protocol 7 (Select & reduce features).

The model-free CPP-style recipe in one picture: rank features by effect size
(|AUC*|), then de-correlate. Survivors are highlighted (green), redundant
echoes greyed out. The survivors are spread across the ranking rather than a
prefix, which is the whole point.

Rendered as a 7x7 inch figure at 150 dpi -> 1050x1050 px square.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

import aaanalysis as aa

aa.options["verbose"] = False
aa.options["random_state"] = 42

# canonical kept/dropped palette (matches the protocol notebook figure)
GREEN = "tab:green"
GREY = "lightgray"
OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol7.png"

# --- data pipeline (P7 model-free recipe) ----------------------------------
df_seq = aa.load_dataset(name="DOM_GSEC", n=10)      # 20 sequences (10/10)
labels = df_seq["label"].to_list()

sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
cpp = aa.CPP(df_parts=df_parts)
df_feat = cpp.run(labels=labels, n_filter=50, n_jobs=1)
X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)

# 1) effect size, 2) effect-sorted order, 3) correlation filter (sort-first)
auc = aa.comp_auc_adjusted(X=X, labels=labels, n_jobs=1)
order = np.argsort(np.abs(auc), kind="stable")[::-1]
nf = aa.NumericalFeature()
mask = nf.filter_correlation(X=X[:, order], max_cor=0.7)

abs_auc_sorted = np.abs(auc)[order]
n_feat = X.shape[1]
n_keep = int(mask.sum())

# --- plot ------------------------------------------------------------------
aa.plot_settings(font_scale=1.3, weight_bold=False, short_ticks=True)
fig, ax = plt.subplots(figsize=(7, 7))

# zoom y to the data band so the ranked staircase fills the square; baseline
# tick (0.40) kept visible so the zoom is transparent
y_lo, y_hi = 0.39, 0.55
colors = [GREEN if m else GREY for m in mask]
ax.bar(range(len(abs_auc_sorted)), abs_auc_sorted - y_lo, bottom=y_lo,
       color=colors, width=1.0, edgecolor="white", linewidth=0.3, zorder=3)

ax.set_xlabel("Feature rank (by effect size)")
ax.set_ylabel("Effect size  |AUC*|")
ax.set_xlim(-0.7, len(abs_auc_sorted) - 0.3)
ax.set_ylim(y_lo, y_hi)
ax.set_yticks([0.40, 0.45, 0.50])
ax.set_xticks([0, 20, 40])

ax.set_title("Keep the strong, drop the redundant",
             size=aa.plot_gcfs(), pad=12)

aa.plot_legend(
    ax=ax,
    dict_color={f"kept ({n_keep})": GREEN, f"dropped ({n_feat - n_keep})": GREY},
    n_cols=1, loc="upper right", marker="s", marker_size=15,
)

import seaborn as sns
sns.despine()
plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT, "| features", n_feat, "-> kept", n_keep)
