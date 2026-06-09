"""
This is a script for scale loading function. Please define new loading functions by their loaded data by introducing
 a new data table in docs/source/index/tables_templates.rst.
"""
from typing import Literal, Optional, Union
import numpy as np
from pandas import DataFrame
from aaanalysis import utils as ut


# Check functions for load_scales
def check_name_of_scale(name=None):
    # Check if the provided scale name is valid
    if name not in ut.NAMES_SCALE_SETS:
        raise ValueError(f"'name' ('{name}') is not valid. Choose one of following: {ut.NAMES_SCALE_SETS}")


def check_top60_n(name=None, top60_n=None):
    """Check if name is valid and top60_n is between 1 and 60"""
    if top60_n is None:
        return
    if isinstance(top60_n, str):
        if "AAC" not in top60_n:
            raise ValueError(f"'top60_n' ('{top60_n}') should be int or 'AAC' id")
        top60_n = int(top60_n.replace("AAC", ""))
    ut.check_number_range(name="top60_n", val=top60_n, min_val=1, max_val=60, just_int=True)
    matching_scale_sets = [ut.STR_SCALES, ut.STR_SCALE_CAT, ut.STR_SCALES_RAW]
    if name not in matching_scale_sets:
        raise ValueError(f"'name' ('{name}') is not valid for 'top60_n' ({top60_n})."
                         f" Choose one of following: {matching_scale_sets}")
    return top60_n


def check_top_explain(name=None, top_explain_n=None, top_explain_min_th=None, top60_n=None):
    """Validate the interpretability-tier selector (top_explain_n / top_explain_min_th)."""
    if top_explain_n is None:
        if top_explain_min_th is not None:
            raise ValueError(f"'top_explain_min_th' ('{top_explain_min_th}') should be None "
                             f"unless 'top_explain_n' is set")
        return
    if top60_n is not None:
        raise ValueError(f"'top_explain_n' ('{top_explain_n}') should not be combined with "
                         f"'top60_n' ('{top60_n}') (only one selector at a time)")
    matching_scale_sets = [ut.STR_SCALES, ut.STR_SCALE_CAT, ut.STR_SCALES_RAW]
    if name not in matching_scale_sets:
        raise ValueError(f"'name' ('{name}') is not valid for 'top_explain_n' ({top_explain_n})."
                         f" Choose one of following: {matching_scale_sets}")
    if top_explain_n not in ut.LIST_TOP_EXPLAIN_N:
        raise ValueError(f"'top_explain_n' ('{top_explain_n}') should be one of {ut.LIST_TOP_EXPLAIN_N}")
    if top_explain_min_th is not None:
        if not any(np.isclose(top_explain_min_th, v) for v in ut.LIST_TOP_EXPLAIN_MIN_TH):
            raise ValueError(f"'top_explain_min_th' ('{top_explain_min_th}') should be None or one of "
                             f"{ut.LIST_TOP_EXPLAIN_MIN_TH}")


# Helper functions for load_scales
def _load_df_cat():
    """Read the per-scale AAontology classification (scales_cat)."""
    return ut.read_csv_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.{ut.STR_FILE_TYPE}")


def _load_df_subcat_raw():
    """Read the curated subcategory table (interpretability, tier, cluster, descriptions)."""
    return ut.read_csv_cached(ut.FOLDER_DATA + f"{ut.STR_SUBCAT}.{ut.STR_FILE_TYPE}")


def _subcat_tier_map():
    """Map each subcategory to its interpretability tier (top_explain); NaN for unclassified."""
    df_sub = _load_df_subcat_raw()
    return dict(zip(df_sub[ut.COL_SUBCAT], df_sub[ut.COL_TOP_EXPLAIN]))


def _load_df_subcat(just_aaindex=False, unclassified_out=False):
    """Assemble df_subcat: curated facts + live n_scales / n_scales_aaindex from scales_cat."""
    df_sub = _load_df_subcat_raw().copy()
    df_cat = _load_df_cat()
    n_all = df_cat.groupby(ut.COL_SUBCAT)[ut.COL_SCALE_ID].size()
    df_aa = _filter_scales(df_cat=df_cat, unclassified_out=False, just_aaindex=True)
    n_aa = df_aa.groupby(ut.COL_SUBCAT)[ut.COL_SCALE_ID].size()
    df_sub[ut.COL_N_SCALES] = df_sub[ut.COL_SUBCAT].map(n_all).fillna(0).astype(int)
    df_sub[ut.COL_N_SCALES_AAINDEX] = df_sub[ut.COL_SUBCAT].map(n_aa).fillna(0).astype(int)
    if unclassified_out:
        df_sub = df_sub[~df_sub[ut.COL_SUBCAT].str.contains("Unclassified") & (df_sub[ut.COL_CAT] != "Others")]
    if just_aaindex:
        df_sub = df_sub[df_sub[ut.COL_N_SCALES_AAINDEX] > 0]
    df_sub = df_sub[ut.COLS_SUBCAT].reset_index(drop=True)
    return df_sub


def _get_explain_scales(top_explain_n=None, top_explain_min_th=None, just_aaindex=False):
    """Resolve scale_ids for an interpretability tier (optionally AAclust-reduced)."""
    if top_explain_min_th is None:
        # Pure subcategory-tier filter (all member scales of the selected subcats).
        # Tier lives on df_subcat (single source); join it onto scales by subcategory.
        df_cat = _load_df_cat()
        df_cat = _filter_scales(df_cat=df_cat, unclassified_out=False, just_aaindex=just_aaindex)
        tier = df_cat[ut.COL_SUBCAT].map(_subcat_tier_map())
        mask = tier.notna() & (tier <= top_explain_n)
        return df_cat.loc[mask, ut.COL_SCALE_ID].tolist()
    # Precomputed AAclust-reduced selection
    df_te = ut.read_csv_cached(ut.FOLDER_DATA + f"{ut.STR_TOP_EXPLAIN}.{ut.STR_FILE_TYPE}")
    row = df_te[(df_te[ut.COL_TOP_EXPLAIN] == top_explain_n)
                & (np.isclose(df_te["min_th"], top_explain_min_th))
                & (df_te["just_aaindex"] == bool(just_aaindex))]
    if len(row) != 1:
        raise RuntimeError(f"No precomputed top_explain selection for (top_explain_n={top_explain_n}, "
                           f"min_th={top_explain_min_th}, just_aaindex={just_aaindex})")
    ids = row.iloc[0]["scale_ids"]
    if ids is None or (isinstance(ids, float) and np.isnan(ids)) or ids == "":
        return []
    return str(ids).split(";")
def _filter_scales(df_cat=None, unclassified_out=False, just_aaindex=False):
    """Filter scales for unclassified and aaindex scales"""
    list_ids_not_in_aaindex = [x for x in df_cat[ut.COL_SCALE_ID] if "LINS" in x or "KOEH" in x]
    list_ids_unclassified = [x for x, cat, sub_cat in zip(df_cat[ut.COL_SCALE_ID], df_cat[ut.COL_CAT], df_cat[ut.COL_SUBCAT])
                             if "Unclassified" in sub_cat or cat == "Others"]
    list_ids_to_exclude = []
    if unclassified_out:
        list_ids_to_exclude.extend(list_ids_unclassified)
    if just_aaindex:
        list_ids_to_exclude.extend(list_ids_not_in_aaindex)
    df_cat = df_cat[~df_cat[ut.COL_SCALE_ID].isin(list_ids_to_exclude)]
    return df_cat


def _get_selected_scales(top60_n=None):
    """Get selected top scale set"""
    df_eval = ut.read_csv_cached(ut.FOLDER_DATA + f"{ut.STR_TOP60}.{ut.STR_FILE_TYPE}").drop("top60_id", axis=1)
    # Find the names of the index where the value is not 0
    _df = df_eval.iloc[top60_n - 1]
    selected_scales = _df[_df != 0].index.tolist()
    return selected_scales


def _adjust_dtypes(df=None, name=None):
    """Set dtypes to avoid loading problems"""
    # Adjust data type of column values
    name_all_float = [ut.STR_SCALES, ut.STR_SCALES_RAW, ut.STR_SCALES_PC]
    name_all_int = [ut.STR_TOP60]
    if name in name_all_float:
        df = df.astype(float)
    elif name in name_all_int:
        df = df.astype(int)
    return df


# II Main Functions
def load_scales(name: Literal["scales", "scales_raw", "scales_cat", "scales_pc", "top60", "top60_eval", "subcat"] = "scales",
                just_aaindex: bool = False,
                unclassified_out: bool = False,
                top60_n: Optional[Union[int, str]] = None,
                top_explain_n: Optional[int] = None,
                top_explain_min_th: Optional[float] = None,
                ) -> DataFrame:
    """
    Load amino acid scales or their classification (AAontology).

    The amino acid scales (``name='scales_raw'``) encompass all scales from AAindex ([Kawashima08]_) along with two
    additional data sources. These scales were min-max normalized (``name='scales'``) and organized in a two-level
    classification called AAontology (``name='scales_cat'``), as detailed in [Breimann24b]_. The first 20 principal
    components (PCs) of all compressed scales are provided (``name='scales_pc'``) and were used for an in-depth analysis
    of redundancy-reduced scale subsets obtained by :class:`AAclust` ([Breimann24a]_). The top 60 scale sets from
    this analysis are available either collectively (``name='top60'``) or individually (``top60_n='1-60'``),
    accompanied by their evaluations in the ``'top60_eval'`` dataset. For more interpretable analyses,
    simplified scale sets restricted to the most interpretable AAontology subcategories are available via
    ``top_explain_n`` (optionally redundancy-reduced with ``top_explain_min_th``).

    .. versionadded:: 0.1.0

    Parameters
    ----------
    name : str, default='scales'
        Name of the loaded dataset:

        - ``scales_raw``: All amino acid scales.
        - ``scales``:  Min-max normalized raw scales.
        - ``scales_cat``: Two-level classification (AAontology).
        - ``scales_pc``: First 20 PCs of compressed scales.
        - ``top60``:  Selection of 60 best performing scale sets.
        - ``top60_eval``: Evaluation of 60 best performing scale sets.
        - ``subcat``: Subcategory overview (one row per subcategory) with its interpretability
          rating (1-10; 1 = most interpretable), interpretability tier, scale counts
          (``n_scales`` and ``n_scales_aaindex``), cluster, and descriptions/references.

        Or Number between 1 and 60 to select the i-th top60 dataset.

    just_aaindex : bool, default=False
        If ``True``, returns only scales from AAindex. Relevant only for 'scales', 'scales_raw', or 'scales_cat'.
    unclassified_out : bool, default=False
        Determines exclusion of unclassified scales. Relevant only for 'scales', 'scales_raw', or 'scales_cat'.
    top60_n : int or str, optional
         Select the n-th scale set from top60 sets and return it for 'scales', 'scales_raw', or 'scales_cat'.
         Allowed strings are AAclust ids (e.g., 'AAC01').
    top_explain_n : int, optional
        Select a simplified, more interpretable scale set by restricting it to the ``n`` most interpretable
        AAontology subcategories (one of 5, 10, ..., 60). Returns *all* member scales of those subcategories
        for 'scales', 'scales_raw', or 'scales_cat'. Mutually exclusive with ``top60_n``. Subcategories were
        ranked by interpretability from unsupervised clustering combined with expert domain knowledge of
        AAontology (no separate publication).

        .. versionadded:: 1.1.0
    top_explain_min_th : float, optional
        Pearson correlation threshold (one of 0.3, 0.4, ..., 0.9) for an additional :class:`AAclust`
        redundancy reduction of the ``top_explain_n`` set, using pre-computed selections with AAclust default
        settings. ``None`` (default) returns the full set without redundancy reduction. Only valid together
        with ``top_explain_n``.

        .. versionadded:: 1.1.0

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the chosen dataset, recommended to be named by their name suffix (``df_scales``, ``df_cat``).

    Notes
    -----
    * ``df_cat`` includes the following columns:

        - 'scale_id': ID of scale (from AAindex or following the same naming convention).
        - 'category': Category of scale (defined in AAontology).
        - 'subcategory': Subcategory of scale (AAontology).
        - 'scale_name': Name of scale derived from scale description.
        - 'scale_description': Description of scale (derived from AAindex).

    * Scales under the 'Others' category are considered unclassified.
    * When ``top_explain_n`` is set, the returned ``df_cat`` additionally includes an 'interpretability'
      column (1-10 rating; 1 = most interpretable) and a 'top_explain' column (the interpretability tier);
      these columns are absent from the default ``df_cat``.
    * Unlike ``top60`` (AAclust redundancy-reduced and performance-ranked), ``top_explain_n`` with
      ``top_explain_min_th=None`` returns all scales of the selected subcategories without redundancy
      reduction. With ``top_explain_min_th`` set, the AAclust reduction (and a ``just_aaindex=True``
      post-filter) may leave a selected subcategory with no representative scale, so the reduced set is
      not guaranteed to cover every subcategory in the tier.

    See Also
    --------
    * Overview of all loading options: :ref:`t2_overview_scales`.
    * AAontology: :ref:`t3a_aaontology_categories` and :ref:`t3b_aaontology_subcategories` tables.
    * Step-by-step guide in the `Scale Loading Tutorial <tutorial2b_scales_loader.html>`_.
    * :class:`AAclust` for customizing redundancy-reduced scale sets.

    Examples
    --------
    .. include:: examples/load_scales.rst
    """
    # DEV: every returned must be copied to avoid in-place mutation for newly loaded dataframes
    # Check input
    check_name_of_scale(name=name)
    ut.check_bool(name="just_aaindex", val=just_aaindex)
    ut.check_bool(name="unclassified_in", val=unclassified_out)
    top60_n = check_top60_n(name=name, top60_n=top60_n)
    check_top_explain(name=name, top_explain_n=top_explain_n,
                      top_explain_min_th=top_explain_min_th, top60_n=top60_n)

    # Subcategory overview (interpretability, tier, scale counts, descriptions)
    if name == ut.STR_SUBCAT:
        return _load_df_subcat(just_aaindex=just_aaindex, unclassified_out=unclassified_out).copy()

    # Load and filter top60 scales
    if top60_n is not None:
        selected_scales = _get_selected_scales(top60_n=top60_n)
        if name == ut.STR_SCALE_CAT:
            df_cat = _load_df_cat()
            df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(selected_scales)].reset_index(drop=True)
            return df_cat.copy()
        elif name in [ut.STR_SCALES, ut.STR_SCALES_RAW]:
            df = ut.read_csv_cached(ut.FOLDER_DATA + name + f".{ut.STR_FILE_TYPE}", index_col=0)
            df = df[selected_scales]
            df = _adjust_dtypes(df=df, name=name)
            return df.copy()
        else:
            raise ValueError(f"Wrong 'name' ('{name}') for 'top60_n")

    # Load and filter interpretability-tiered ("explainable") scales
    if top_explain_n is not None:
        selected_scales = _get_explain_scales(top_explain_n=top_explain_n,
                                               top_explain_min_th=top_explain_min_th,
                                               just_aaindex=just_aaindex)
        if name == ut.STR_SCALE_CAT:
            df_cat = _load_df_cat()
            df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(selected_scales)].reset_index(drop=True)
            return df_cat.copy()
        df = ut.read_csv_cached(ut.FOLDER_DATA + name + f".{ut.STR_FILE_TYPE}", index_col=0)
        selected_in_order = [x for x in list(df) if x in set(selected_scales)]
        df = df[selected_in_order]
        df = _adjust_dtypes(df=df, name=name)
        return df.copy()

    # Load unfiltered data
    if not unclassified_out and not just_aaindex:
        if name == ut.STR_SCALE_CAT:
            return _load_df_cat().copy()
        df = ut.read_csv_cached(ut.FOLDER_DATA + name + f".{ut.STR_FILE_TYPE}", index_col=0)
        df = _adjust_dtypes(df=df, name=name)
        return df.copy()

    # Load and filter scale categories
    df_cat = _load_df_cat()
    df_cat = _filter_scales(df_cat=df_cat, unclassified_out=unclassified_out, just_aaindex=just_aaindex)
    if name == ut.STR_SCALE_CAT:
        return df_cat.reset_index(drop=True).copy()

    # Load and filter scales
    df = ut.read_csv_cached(ut.FOLDER_DATA + name + f".{ut.STR_FILE_TYPE}", index_col=0)
    if name in [ut.STR_SCALES, ut.STR_SCALES_RAW]:
        selected_scales = [x for x in list(df) if x in list(df_cat[ut.COL_SCALE_ID])]
        df = df[selected_scales]
    df = _adjust_dtypes(df=df, name=name)
    return df.copy()
