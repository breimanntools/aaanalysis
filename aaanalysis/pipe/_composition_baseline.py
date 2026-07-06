"""
This is a script for the composition-baseline helpers behind ``find_features(baselines=...)``.

Composition baselines quantify how much the positional ``CPP`` Part-Split-Scale features add over a
plain k-mer frequency encoding. Two representations are produced, both keyed off
:meth:`SequenceFeature.kmer_composition`:

* **AAC (k=1) as a first-class CPP ``df_feat``.** Amino-acid composition *is* CPP over a one-hot
  identity scale set restricted to the whole-part ``Segment(1,1)`` split, so it yields genuine
  ``PART-Segment(1,1)-<AA>`` features with the usual CPP statistics and renders in the CPP feature map
  (``build_onehot_scales`` + ``build_aac_df_feat``).
* **DPC / higher k-mers as a composition signal map.** A dipeptide / k-mer is a property of an
  adjacent residue *tuple*, not of a single residue, so it cannot be a Part-Split-Scale feature; it is
  instead visualized by ``plot_composition_map`` as the per-k-mer discriminative signal (mean
  composition in the test class minus in the reference class): a 1x20 strip (k=1), a 20x20 heatmap
  (k=2), or top-N ranked bars (k>=3), all on the same diverging, feature-map-consistent palette.
"""
import itertools
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import aaanalysis.utils as ut

# AAontology-independent physicochemical grouping of the 20 canonical amino acids, used only to give
# the one-hot AAC scales a small, colorable category structure in the feature map.
_AA_GROUPS = {"Nonpolar": "GAVLIPM", "Aromatic": "FWY", "Polar": "STCNQ",
              "Positive": "KRH", "Negative": "DE"}
_AA_TO_CAT = {aa: cat for cat, aas in _AA_GROUPS.items() for aa in aas}
# Fixed colors for the five AA classes (Set2-style), used for the AAC feature map's ``dict_color``.
AAC_CAT_COLORS = {"Nonpolar": "#8DA0CB", "Aromatic": "#66C2A5", "Polar": "#FFD92F",
                  "Positive": "#E78AC3", "Negative": "#FC8D62"}


# I Helper functions
def _kmer_labels(k):
    """All ``20 ** k`` canonical k-mers in ``itertools.product(LIST_CANONICAL_AA, repeat=k)`` order."""
    return ["".join(p) for p in itertools.product(ut.LIST_CANONICAL_AA, repeat=k)]


# II Main functions
def build_onehot_scales():
    """Build the one-hot identity ``df_scales`` (20x20) and its AA-category ``df_cat`` for AAC-as-CPP.

    Returns
    -------
    df_scales : pd.DataFrame, shape (20, 20)
        Identity matrix indexed by :data:`ut.LIST_CANONICAL_AA`; column ``<AA>`` is 1 on residue ``AA``
        and 0 elsewhere, so ``Segment(1,1)`` averaging over a span gives that amino acid's fraction.
    df_cat : pd.DataFrame, shape (20, 5)
        Category table for the 20 one-hot scales (``category`` / ``subcategory`` = physicochemical
        class), matching the ``df_cat`` contract consumed by :class:`CPP` / :class:`CPPPlot`.
    """
    aa = list(ut.LIST_CANONICAL_AA)
    df_scales = pd.DataFrame(np.eye(len(aa)), index=aa, columns=aa)
    df_cat = pd.DataFrame({
        ut.COL_SCALE_ID: aa,
        ut.COL_CAT: [_AA_TO_CAT[a] for a in aa],
        ut.COL_SUBCAT: [_AA_TO_CAT[a] for a in aa],
        ut.COL_SCALE_NAME: aa,
        ut.COL_SCALE_DES: [f"Amino acid {a} indicator" for a in aa],
    })
    return df_scales, df_cat


def build_aac_df_feat(sf=None, df_seq=None, labels=None, list_parts=None, jmd_n_len=10, jmd_c_len=10,
                      label_test=1, label_ref=0, n_filter=20, random_state=None, n_jobs=None,
                      verbose=False):
    """Amino-acid composition as a first-class CPP ``df_feat`` (one-hot scales, ``Segment(1,1)``).

    Runs :class:`CPP` over the one-hot identity scale set with the whole-part ``Segment(1,1)`` split, so
    each feature ``<PART>-Segment(1,1)-<AA>`` carries the standard CPP statistics and the result can be
    reconstructed by :meth:`SequenceFeature.feature_matrix` and drawn by :meth:`CPPPlot.feature_map`.

    Returns
    -------
    df_feat : pd.DataFrame
        Up to 20 amino-acid-composition features with CPP statistics.
    df_parts : pd.DataFrame
        The parts used (so the caller can rebuild ``X`` / draw the map on the same geometry).
    df_scales : pd.DataFrame
    df_cat : pd.DataFrame
        The one-hot scale set and its category table (needed to reconstruct ``X`` and color the map).
    """
    from aaanalysis.feature_engineering import CPP
    df_scales, df_cat = build_onehot_scales()
    list_parts_used = ["tmd_jmd"] if list_parts is None else list_parts
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

    Returns
    -------
    signal : np.ndarray, shape (20 ** k,)
        ``mean(kmer_composition | label_test) - mean(kmer_composition | label_ref)`` per k-mer
        (``itertools.product(LIST_CANONICAL_AA, repeat=k)`` order); positive = enriched in the test class.
    kmers : list of str
        The ``20 ** k`` k-mer column labels aligned with ``signal``.
    """
    X = sf.kmer_composition(df_seq=df_seq, k=k,
                            list_parts=list_parts if list_parts is not None else "tmd_jmd")
    y = np.asarray(labels)
    signal = np.nanmean(X[y == label_test], axis=0) - np.nanmean(X[y == label_ref], axis=0)
    return signal, _kmer_labels(k)


def plot_composition_map(signal=None, k=None, ax=None, top_n=25, name_test="TEST", name_ref="REF",
                         cmap="RdBu_r"):
    """Draw the composition signal: 1x20 strip (k=1), 20x20 heatmap (k=2), or top-N bars (k>=3).

    All three share the diverging, zero-centered ``TEST - REF`` colormap so AAC / DPC / k-mer read as
    one family ("which composition units separate the classes"), consistent with the CPP feature map.
    """
    import matplotlib.pyplot as plt
    aa = list(ut.LIST_CANONICAL_AA)
    signal = np.asarray(signal, dtype=float)
    vmax = float(np.nanmax(np.abs(signal))) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cbar_label = f"Composition   {name_test} − {name_ref}"
    if k == 1:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 1.5))
        im = ax.imshow(signal.reshape(1, 20), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(20)); ax.set_xticklabels(aa)
        ax.set_yticks([])
        ax.figure.colorbar(im, ax=ax, fraction=0.08, pad=0.02, label=cbar_label)
    elif k == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(signal.reshape(20, 20), cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(20)); ax.set_xticklabels(aa, fontsize=8)
        ax.set_yticks(range(20)); ax.set_yticklabels(aa, fontsize=8)
        ax.set_xlabel("2nd residue"); ax.set_ylabel("1st residue")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)
    else:
        kmers = _kmer_labels(k)
        order = np.argsort(-np.abs(signal))[:top_n][::-1]
        vals = signal[order]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, max(3.0, 0.28 * len(vals))))
        ax.barh(range(len(vals)), vals, color=plt.get_cmap(cmap)(norm(vals)))
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([kmers[i] for i in order], fontsize=7, family="monospace")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel(cbar_label)
    return ax
