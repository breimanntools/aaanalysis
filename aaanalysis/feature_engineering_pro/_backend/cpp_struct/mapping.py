"""
This is a script for the backend feature->residue mapping of the CPPStructurePlot
class. It reuses the shared CPP feature backend (``get_positions_`` +
``get_df_pos_`` with the normalized-sum semantics) so the per-residue impact is
identical to what the CPP profile / feature map show, never a re-implemented
per-position loop.
"""
import numpy as np

import aaanalysis.utils as ut
# Shared CPP feature backend (the deliberately shared ``_backend/cpp/`` package,
# registered as SHARED_BACKEND_SUBPKGS for feature_engineering).
from aaanalysis.feature_engineering._backend.cpp.utils_feature import (
    get_positions_, get_df_pos_)


# I Helper Functions
def _positions_union(feat_positions):
    """Flatten the comma-separated position strings into a sorted list of ints."""
    positions = set()
    for pos_str in feat_positions:
        if not pos_str:
            continue
        for p in str(pos_str).split(","):
            if p != "":
                positions.add(int(p))
    return sorted(positions)


# II Main Functions
def compute_residue_impact(df_feat=None, col_imp=None, start=1, tmd_len=20,
                           jmd_n_len=10, jmd_c_len=10, col_cat=None,
                           normalize_by_span=False):
    """Map per-feature impact onto absolute residue numbers.

    The feature positions are derived with the shared ``get_positions_`` helper
    (so ``start`` shifts them to absolute residue numbers). Two per-residue
    aggregations are supported, selected by ``normalize_by_span``:

    * ``normalize_by_span=False`` (default, app-fidelity): each feature's **full**
      signed impact is added to **every** residue it spans, exactly like the
      deployed app's per-position cumulative loop. A multi-residue feature
      therefore colours all its residues at full strength.
    * ``normalize_by_span=True``: aggregated with ``get_df_pos_(value_type="sum")``,
      which divides each feature's value by the number of positions it spans before
      summing — the same span-normalized sum the :meth:`CPPPlot.profile` and the
      :meth:`CPPPlot.feature_map` top per-position bar use.

    Returns
    -------
    dict_impact : dict
        ``{resi: impact}`` for every residue in ``[start, stop]``.
    max_abs : float
        Maximum absolute per-residue impact (0.0 if none finite); used to
        normalise the colour ramp.
    positions_union : list of int
        Sorted residue numbers actually spanned by ``df_feat`` (the auto window).
    """
    col_cat = ut.COL_CAT if col_cat is None else col_cat
    df_feat = df_feat.copy()
    features = df_feat[ut.COL_FEATURE].to_list()
    feat_positions = get_positions_(features=features, start=start, tmd_len=tmd_len,
                                    jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    df_feat[ut.COL_POSITION] = feat_positions
    stop = start + jmd_n_len + tmd_len + jmd_c_len - 1
    if normalize_by_span:
        if col_cat not in df_feat.columns:
            df_feat[col_cat] = "feature"
        df_pos = get_df_pos_(df_feat=df_feat, col_cat=col_cat, col_val=col_imp,
                             value_type="sum", start=start, stop=stop)
        # Rows = scale categories, columns = positions; sum to one value per residue.
        series = df_pos.sum(axis=0)
        dict_impact = {int(p): float(v) for p, v in series.items()}
    else:
        # App-fidelity: add each feature's full signed impact to every residue it
        # spans (no division by span count). Reproduces the deployed app's
        # per-position cumulative colouring; differs from the span-normalized
        # feature_map top bar for any multi-residue feature.
        dict_impact = {p: 0.0 for p in range(start, stop + 1)}
        values = df_feat[col_imp].astype(float).to_list()
        for pos_str, imp in zip(feat_positions, values):
            # Skip NaN impact (e.g. a partially failed SHAP run) so it does not poison
            # the per-residue sum — mirrors the skipna sum of the normalized branch.
            if not np.isfinite(imp):
                continue
            for p in str(pos_str).split(","):
                if p == "":
                    continue
                resi = int(p)
                if resi in dict_impact:
                    dict_impact[resi] += imp
    finite = [abs(v) for v in dict_impact.values() if np.isfinite(v)]
    max_abs = max(finite) if finite else 0.0
    positions_union = _positions_union(feat_positions)
    return dict_impact, max_abs, positions_union
