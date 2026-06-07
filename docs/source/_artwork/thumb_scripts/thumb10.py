"""Thumbnail for Protocol 10 (Validation): per-protein rank plot.

The deployment-view headline figure. Each TMD gets one out-of-fold substrate
probability (scored only by folds that never trained on it), proteins are
ranked high-to-low and colored by their true group. Substrate TMDs ranking
above non-substrates, with the deployment threshold drawn in, is the
one-glance evidence that the CPP signature tracks the labels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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

# --- Out-of-fold substrate probability per TMD --------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=RS)
cv_pred = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
proba = cross_val_predict(clf, X, labels, cv=cv_pred, method="predict_proba")[:, 1]
df_rank = pd.DataFrame({
    "score": proba,
    "group": np.where(labels == 1, "substrate", "non-substrate"),
})

# --- Plot (clean square thumbnail) --------------------------------------
aa.plot_settings(font_scale=1.5, weight_bold=False)
fig, ax = aa.plot_rank(df_rank=df_rank, threshold=0.7, marker_size=90,
                       ylabel="Out-of-fold substrate probability")
ax.set_title("Per-protein rank", size=aa.plot_gcfs() + 2)

fig.set_size_inches(7, 7)
plt.tight_layout()
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
