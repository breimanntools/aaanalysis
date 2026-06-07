"""Thumbnail for Protocol 10 (Validation): shuffled-label control.

Real cross-validation MCC fold scores (high) vs the shuffled-label null
(collapses toward 0). The wide vertical gap is the protocol's headline
evidence that the signature tracks the labels, not noise.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef

import aaanalysis as aa

aa.options["verbose"] = False
RS = 42

OUT = ("/Users/stephanbreimann/Programming/1Packages/aaanalysis/"
       "docs/source/_static/img/thumbs/protocol10.png")

# --- Data + signature (same fixture as the protocol) --------------------
df_seq = aa.load_dataset(name="DOM_GSEC", n=20)        # 20 per class -> 40 rows
labels = np.array(df_seq["label"].to_list())
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
cpp = aa.CPP(df_parts=df_parts)
df_feat = cpp.run(labels=labels, n_filter=25, n_jobs=1)
X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)

# --- Real repeated-CV MCC + shuffled-label null -------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=RS)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)  # 15 fits
mcc = make_scorer(matthews_corrcoef)
scores = cross_val_score(clf, X, labels, cv=cv, scoring=mcc)

rng = np.random.default_rng(RS)
shuffled_means = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(10):
        labels_shuffled = rng.permutation(labels)
        s = cross_val_score(clf, X, labels_shuffled, cv=cv, scoring=mcc)
        shuffled_means.append(s.mean())
shuffled_means = np.array(shuffled_means)

# --- Plot ---------------------------------------------------------------
aa.plot_settings(font_scale=1.55, weight_bold=False)
c_null, c_real = aa.plot_get_clist(n_colors=2)

fig, ax = plt.subplots()
jit = np.random.default_rng(0)
x_shuf = 0 + jit.normal(0.0, 0.05, size=len(shuffled_means))
x_real = 1 + jit.normal(0.0, 0.05, size=len(scores))

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6, zorder=0)
ax.scatter(x_shuf, shuffled_means, color=c_null, s=150, alpha=0.85,
           edgecolor="white", linewidth=0.8, zorder=3)
ax.scatter(x_real, scores, color=c_real, s=150, alpha=0.85,
           edgecolor="white", linewidth=0.8, zorder=3)

ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.4, 1.12)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Shuffled\nnull", "Real\nlabels"])
ax.set_ylabel("MCC")
ax.set_title("Shuffled-label control")

fig.set_size_inches(7, 7)
plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
