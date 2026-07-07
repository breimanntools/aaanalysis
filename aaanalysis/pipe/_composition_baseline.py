"""
This is a script for the composition-baseline helpers behind ``find_features(baselines=...)``.

Composition baselines quantify how much the positional ``CPP`` Part-Split-Scale features add over a
plain k-mer frequency encoding. Two representations are produced, both keyed off
:meth:`SequenceFeature.kmer_composition`:

* **AAC (k=1) as a first-class CPP ``df_feat``.** Amino-acid composition *is* CPP over a one-hot
  identity scale set restricted to the whole-part ``Segment(1,1)`` split, so it yields genuine
  ``PART-Segment(1,1)-<AA>`` features with the usual CPP statistics and renders in the CPP feature map.
  The one-hot ``df_scales`` + ``df_cat`` come from ``kmer_composition(k=1, return_scales=True)``.
* **DPC / higher k-mers cannot be Part-Split-Scale features** (a k-mer is a property of an adjacent
  tuple, not of a single residue). But CPP's *discriminative* filtering still applies to the
  composition matrix: ``comp_kmer_df_feat`` scores every k-mer with the same statistics CPP ranks on
  (adjusted AUC, mean difference, test-group std), optionally drops correlated k-mers, and keeps the
  top ``n_filter`` — a ``df_feat``-shaped ranked table. ``plot_composition_map`` shows the per-k-mer
  ``test − ref`` signal (1x20 strip / 20x20 heatmap / top-N bars) and marks / ranks the filtered set.
"""
import itertools
import warnings
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import aaanalysis.utils as ut
from aaanalysis.metrics import comp_auc_adjusted
from aaanalysis.feature_engineering import CPP, NumericalFeature

# Fixed colors for the five AA physicochemical classes (Set2-style), used for the AAC feature map's
# ``dict_color`` (the classes are assigned by the backend ``get_composition_scales_`` df_cat).
AAC_CAT_COLORS = {"Nonpolar": "#8DA0CB", "Aromatic": "#66C2A5", "Polar": "#FFD92F",
                  "Positive": "#E78AC3", "Negative": "#FC8D62"}


# I Helper functions
def _kmer_labels(k):
    """All ``20 ** k`` canonical k-mers in ``itertools.product(LIST_CANONICAL_AA, repeat=k)`` order."""
    return ["".join(p) for p in itertools.product(ut.LIST_CANONICAL_AA, repeat=k)]


# II Main functions
def build_aac_df_feat(sf=None, df_seq=None, labels=None, list_parts=None, jmd_n_len=10, jmd_c_len=10,
                      label_test=1, label_ref=0, n_filter=20, random_state=None, n_jobs=None,
                      verbose=False):
    """Amino-acid composition as a first-class CPP ``df_feat`` (one-hot scales, ``Segment(1,1)``).

    Runs :class:`CPP` over the one-hot identity scale set (from
    ``kmer_composition(k=1, return_scales=True)``) with the whole-part ``Segment(1,1)`` split, so each
    feature ``<PART>-Segment(1,1)-<AA>`` carries the standard CPP statistics and the result can be
    reconstructed by :meth:`SequenceFeature.feature_matrix` and drawn by :meth:`CPPPlot.feature_map`.

    Returns ``(df_feat, df_parts, df_scales, df_cat)``.
    """
    list_parts_used = ["tmd_jmd"] if list_parts is None else list_parts
    _, df_scales, df_cat = sf.kmer_composition(df_seq=df_seq, k=1, list_parts=list_parts_used,
                                               return_scales=True)
    df_parts = sf.get_df_parts(df_seq=df_seq, list_parts=list_parts_used,
                               jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    split_kws = sf.get_split_kws(split_types="Segment", n_split_min=1, n_split_max=1)
    cpp = CPP(df_parts=df_parts, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat,
              random_state=random_state, verbose=verbose)
    df_feat = cpp.run(labels=labels, label_test=label_test, label_ref=label_ref,
                      n_filter=n_filter, n_jobs=n_jobs)
    return df_feat, df_parts, df_scales, df_cat


def comp_kmer_signal(sf=None, df_seq=None, labels=None, k=2, list_parts=None,
                     label_test=1, label_ref=0):
    """Per-k-mer discriminative signal: mean composition in the test class minus in the reference class.

    Returns ``(signal, kmers)`` where ``signal[i] = mean(comp | test) - mean(comp | ref)`` for k-mer
    ``kmers[i]`` (``itertools.product`` order); positive = enriched in the test class.
    """
    X = sf.kmer_composition(df_seq=df_seq, k=k,
                            list_parts=list_parts if list_parts is not None else "tmd_jmd")
    y = np.asarray(labels)
    signal = np.nanmean(X[y == label_test], axis=0) - np.nanmean(X[y == label_ref], axis=0)
    return signal, _kmer_labels(k)


def comp_kmer_df_feat(sf=None, df_seq=None, labels=None, k=2, list_parts=None, label_test=1,
                      label_ref=0, n_filter=100, max_cor=None, min_count=1):
    """CPP-style discriminative filtering of k-mer composition into a ranked ``df_feat``-like table.

    Scores every k-mer with the statistics CPP ranks on — adjusted AUC (``abs_auc`` via
    :func:`comp_auc_adjusted`), mean difference (``mean_dif`` / ``abs_mean_dif``) and test-group std
    (``std_test``) — orders by ``abs_auc``, optionally drops correlated k-mers
    (:meth:`NumericalFeature.filter_correlation`, keeping the higher-AUC one), and keeps the top
    ``n_filter``. This is the composition-space analog of CPP's feature filtering; the positional
    (``max_overlap``) and scale-category redundancy filters do not apply (a k-mer has no residue
    position and is not a per-residue scale). Returns a ``df_feat``-shaped table (``feature`` = k-mer,
    ``category`` / ``subcategory`` = residue class, plus the stat columns), ranked best-first.

    ``min_count`` is a min-occurrence guard: a k-mer must be **present (non-zero) in at least
    ``min_count`` sequences** to be eligible. Higher ``k`` is dominated by sparse presence/absence
    noise (at ``k=3`` most k-mers are zero in almost every sequence), so this keeps noise-only k-mers
    out of the ranking; a ``UserWarning`` is emitted when the eligible set cannot fill ``n_filter``.
    """
    X, _df_scales, df_cat = sf.kmer_composition(df_seq=df_seq, k=k, list_parts=list_parts,
                                                return_scales=True)
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels)
    keep_rows = np.isfinite(X).all(axis=1)                     # drop spans shorter than k (all-NaN)
    Xk, yk = X[keep_rows], y[keep_rows]
    eligible = (Xk > 0).sum(axis=0) >= max(1, int(min_count))  # min-occurrence guard
    auc = np.abs(np.asarray(comp_auc_adjusted(Xk, yk, label_test=label_test, label_ref=label_ref)))
    mean_dif = Xk[yk == label_test].mean(axis=0) - Xk[yk == label_ref].mean(axis=0)
    std_test = Xk[yk == label_test].std(axis=0)
    df = pd.DataFrame({
        ut.COL_FEATURE: df_cat[ut.COL_SCALE_ID].to_numpy(),
        ut.COL_CAT: df_cat[ut.COL_CAT].to_numpy(),
        ut.COL_SUBCAT: df_cat[ut.COL_SUBCAT].to_numpy(),
        ut.COL_ABS_AUC: auc,
        ut.COL_MEAN_DIF: mean_dif,
        ut.COL_ABS_MEAN_DIF: np.abs(mean_dif),
        ut.COL_STD_TEST: std_test,
    })
    df, Xk = df[eligible].reset_index(drop=True), Xk[:, eligible]
    if len(df) < n_filter:
        warnings.warn(f"only {len(df)} of {20 ** k} {k}-mer(s) occur in >= {min_count} sequence(s); "
                      f"the k={k} composition is too sparse to fill n_filter={n_filter} (higher k is "
                      f"mostly presence/absence noise — raise 'min_count' or lower 'k').")
    order = np.argsort(-df[ut.COL_ABS_AUC].to_numpy())         # best AUC first (scale-free across k)
    df, Xk = df.iloc[order].reset_index(drop=True), Xk[:, order]
    if max_cor is not None:
        keep = np.asarray(NumericalFeature.filter_correlation(Xk, max_cor=max_cor), dtype=bool)
        df = df[keep].reset_index(drop=True)
    return df.head(int(n_filter)).reset_index(drop=True)


def plot_composition_map(signal=None, k=None, df_feat=None, ax=None, top_n=25, name_test="TEST",
                         name_ref="REF", cmap="RdBu_r"):
    """Draw the composition signal: 1x20 strip (k=1), 20x20 heatmap (k=2), or top-N bars (k>=3).

    All three share the diverging, zero-centered ``TEST - REF`` colormap so AAC / DPC / k-mer read as
    one family, consistent with the CPP feature map. When ``df_feat`` (a
    :func:`comp_kmer_df_feat` table) is given, the **filtered / ranked** k-mers drive the plot: the
    heatmap outlines the selected cells and the bars show the ``df_feat`` rows (ranked by ``abs_auc``).
    """
    import matplotlib.pyplot as plt
    aa = list(ut.LIST_CANONICAL_AA)
    aa_idx = {a: i for i, a in enumerate(aa)}
    signal = np.asarray(signal, dtype=float)
    vmax = float(np.nanmax(np.abs(signal))) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cbar_label = f"Composition   {name_test} − {name_ref}"
    sel_kmers = list(df_feat[ut.COL_FEATURE]) if df_feat is not None else None
    if k == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 1.5))
        im = ax.imshow(signal.reshape(1, 20), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(20)); ax.set_xticklabels(aa)
        ax.set_yticks([])
        if sel_kmers is not None:
            for km in sel_kmers:
                ax.scatter(aa_idx[km], 0, s=16, facecolors="none", edgecolors="black", linewidths=0.9)
        ax.figure.colorbar(im, ax=ax, fraction=0.08, pad=0.02, label=cbar_label)
    elif k == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(signal.reshape(20, 20), cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(20)); ax.set_xticklabels(aa, fontsize=8)
        ax.set_yticks(range(20)); ax.set_yticklabels(aa, fontsize=8)
        ax.set_xlabel("2nd residue"); ax.set_ylabel("1st residue")
        if sel_kmers is not None:                              # mark the CPP-filtered dipeptides
            for km in sel_kmers:
                ax.scatter(aa_idx[km[1]], aa_idx[km[0]], s=14, facecolors="none",
                           edgecolors="black", linewidths=0.7)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
    else:
        if df_feat is not None:                                # ranked filtered k-mers
            rows = df_feat.head(top_n)
            labs = list(rows[ut.COL_FEATURE])[::-1]
            vals = rows[ut.COL_MEAN_DIF].to_numpy()[::-1]
        else:
            kmers = _kmer_labels(k)
            order = np.argsort(-np.abs(signal))[:top_n][::-1]
            labs = [kmers[i] for i in order]
            vals = signal[order]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, max(3.0, 0.28 * len(vals))))
        ax.barh(range(len(vals)), vals, color=plt.get_cmap(cmap)(norm(vals)))
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labs, fontsize=7, family="monospace")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel(cbar_label)
    return ax
