"""
This is a script for the backend of the AAMut class (per-scale amino-acid substitution impact).
"""
import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
def _get_df_cat_lookup(df_cat=None):
    """Return a scale_id -> (category, subcategory) lookup from a df_cat, or empty dicts."""
    cat, subcat = {}, {}
    if df_cat is not None and ut.COL_SCALE_ID in df_cat.columns:
        cat = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_CAT]))
        subcat = dict(zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_SUBCAT]))
    return cat, subcat


# II Main Functions
def comp_substitution_impact(df_scales=None, df_cat=None, list_from=None, list_to=None,
                             list_scales=None):
    """Compute the signed per-scale delta for every ``from_aa`` -> ``to_aa`` pair.

    Returns a tidy long DataFrame with one row per (from_aa, to_aa, scale_id);
    ``delta = df_scales.loc[to_aa, scale] - df_scales.loc[from_aa, scale]``.
    """
    sub = df_scales[list_scales]
    cat_map, subcat_map = _get_df_cat_lookup(df_cat=df_cat)
    n_scales = len(list_scales)
    arr_scale_id = np.asarray(list_scales)
    arr_cat = np.asarray([cat_map.get(s, np.nan) for s in list_scales], dtype=object)
    arr_subcat = np.asarray([subcat_map.get(s, np.nan) for s in list_scales], dtype=object)
    # Vectorized over scales: accumulate per-(from, to) pair into column lists and build
    # ONE DataFrame at the end (avoids constructing/concatenating hundreds of small frames).
    M = sub.to_numpy(dtype=float)  # rows aligned to sub.index
    idx_of = {a: i for i, a in enumerate(sub.index)}
    from_list = []
    to_list = []
    delta_rows = []
    for from_aa in list_from:
        row_from = M[idx_of[from_aa]]
        for to_aa in list_to:
            if to_aa == from_aa:
                continue
            from_list.append(from_aa)
            to_list.append(to_aa)
            delta_rows.append(M[idx_of[to_aa]] - row_from)
    if len(delta_rows) == 0:
        return pd.DataFrame(columns=ut.COLS_AAMUT)
    n_pairs = len(delta_rows)
    delta = np.concatenate(delta_rows)  # (n_pairs * n_scales,)
    df_impact = pd.DataFrame({
        ut.COL_FROM_AA: np.repeat(from_list, n_scales),
        ut.COL_TO_AA: np.repeat(to_list, n_scales),
        ut.COL_SCALE_ID: np.tile(arr_scale_id, n_pairs),
        ut.COL_CAT: np.tile(arr_cat, n_pairs),
        ut.COL_SUBCAT: np.tile(arr_subcat, n_pairs),
        ut.COL_DELTA: delta,
        ut.COL_ABS_DELTA: np.abs(delta),
    })
    return df_impact[ut.COLS_AAMUT].reset_index(drop=True)


def eval_substitution_impact(df_impact=None):
    """Summarize a substitution-impact table per scale (sensitivity) and per from_aa (mutability)."""
    per_scale = (df_impact.groupby(ut.COL_SCALE_ID, sort=False)[ut.COL_ABS_DELTA]
                 .mean().reset_index().rename(columns={ut.COL_ABS_DELTA: ut.COL_MEAN_DELTA_CPP}))
    per_scale = per_scale.sort_values(ut.COL_MEAN_DELTA_CPP, ascending=False).reset_index(drop=True)
    return per_scale
