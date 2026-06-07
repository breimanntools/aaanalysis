"""Thumbnail for Protocol 8 (Classifier): the canonical dPULearn PCA.

Reproduces the protocol's headline figure: the PU proteins projected into the
first two principal components of CPP feature space, with positives, the
carved reliable negatives, and the still-unlabelled pool colour-coded.
Dashed lines mark the positive-class mean per PC.

Rendered as a 7x7 inch figure at 150 dpi -> 1050x1050 px square.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aaanalysis as aa

aa.options["verbose"] = False
aa.options["random_state"] = 42

OUT = "/Users/stephanbreimann/Programming/1Packages/aaanalysis/docs/source/_static/img/thumbs/protocol8.png"

# Mine the CPP signature on the labelled DOM_GSEC set (same as the protocol).
df_seq = aa.load_dataset(name="DOM_GSEC", n=25)
labels = df_seq["label"].to_list()
sf = aa.SequenceFeature()
df_parts = sf.get_df_parts(df_seq=df_seq)
cpp = aa.CPP(df_parts=df_parts)
df_feat = cpp.run(labels=labels, n_filter=50, n_jobs=1)

# Carve reliable negatives from the positives-plus-unlabelled PU dataset.
df_seq_pu = aa.load_dataset(name="DOM_GSEC_PU", n=20)
labels_pu = df_seq_pu["label"].to_list()
df_parts_pu = sf.get_df_parts(df_seq=df_seq_pu)
X_pu = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts_pu, n_jobs=1)
dpul = aa.dPULearn(verbose=False, random_state=42)
dpul = dpul.fit(X=X_pu, labels=labels_pu, n_unl_to_neg=10, n_components=5)

# Canonical dPULearn PCA (square thumbnail).
aa.plot_settings(font_scale=1.05, weight_bold=False)
ax = aa.dPULearnPlot().pca(df_pu=dpul.df_pu_, labels=dpul.labels_,
                           figsize=(7, 7), legend_y=-0.20)
fig = ax.get_figure()
fig.set_size_inches(7, 7)
# Leave generous headroom at the bottom for the multi-row legend
# (saved without bbox_inches="tight").
fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.40)
fig.savefig(OUT, dpi=150, facecolor="white")
print("saved", OUT)
