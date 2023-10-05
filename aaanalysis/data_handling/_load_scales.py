"""
This is a script for scale loading function. Please define new loading functions by their loaded data by introducing
 a new data table in docs/source/index/tables_templates.rst.
"""
from typing import Optional
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
    ut.check_number_range(name="top60_n", val=top60_n, min_val=1, max_val=60, just_int=True)
    matching_scale_sets = [ut.STR_SCALES, ut.STR_SCALE_CAT, ut.STR_SCALES_RAW]
    if name not in matching_scale_sets:
        raise ValueError(f"'name' ('{name}') is not valid for 'top60_n' ({top60_n})."
                         f" Choose one of following: {ut.NAMES_SCALE_SETS}")


# Helper functions for load_scales
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
    """"""
    df_eval = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_TOP60}.xlsx").drop("top60_id", axis=1)
    # Find the names of the index where the value is not 0
    _df = df_eval.iloc[top60_n - 1]
    selected_scales = _df[_df != 0].index.tolist()
    return selected_scales


# II Main Functions
def load_scales(name: str = "scales",
                just_aaindex: bool = False,
                unclassified_out: bool = False,
                top60_n: Optional[int] = None
                ) -> DataFrame:
    """
    Loads amino acid scales or their classification (AAontology).

    The amino acid scales (``name='scales_raw'``) encompass all scales from AAindex ([Kawashima08]_) along with two
    additional data sources. These scales were min-max normalized (``name='scales'``) and organized in a two-level
    classification called AAontology (``name='scales_cat'``), as detailed in [Breimann23b]_. The first 20 principal
    components (PCs) of all compressed scales are provided (``name='scales_pc'``) and were used for an in-depth analysis
    of redundancy-reduced scale subsets obtained by :class:`AAclust` ([Breimann23a]_). The top 60 scale sets from
    this analysis are available either collectively (``name='top60'``) or individually (``top60_n='1-60'``),
    accompanied by their evaluations in the ``'top60_eval'`` dataset.

    Parameters
    ----------
    name
        Name of the loaded dataset:

        - ``scales_raw``: All amino acid scales.
        - ``scales``:  Min-max normlized raw scales.
        - ``scales_cat``: Two-level classification (AAontology).
        - ``scales_pc``: First 20 PC of compressed scales.
        - ``top60``:  Selection of 60 best performing scale sets.
        - ``top60_eval``: Evaluation of 60 best performing scale sets.

        Or Number between 1 and 60 to select the i-th top60 dataset.

    just_aaindex
        If True, returns only scales from AAindex. Relevant only for 'scales', 'scales_raw', or 'scales_cat'.
    unclassified_out
        Determines exclusion of unclassified scales. Relevant only for 'scales', 'scales_raw', or 'scales_cat'.
    top60_n
         Select the n-th scale set from top60 sets and return it for 'scales', 'scales_raw', or 'scales_cat'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the chosen dataset, recommended to be named by their name suffix (``df_scales``, ``df_cat``).

    Notes
    -----
    - ``df_cat`` includes the following columns:

        * 'scale_id': ID of scale (from AAindex or following the same naming convention).
        * 'category': Category of scale (defined in AAontology).
        * 'subcategory': Subcategory of scale (AAontology).
        * 'scale_name': Name of scale derived from scale description.
        * 'scale_description': Description of scale (derived from AAindex).

    - Scales under the 'Others' category are considered unclassified.

    Examples
    --------
    >>> import aaanalysis as aa
    >>> df_scales = aa.load_scales()
    >>> df_cat = aa.load_scales(name="scales_cat")

    See Also
    --------
    * Overview of all loading options: :ref:`t2_overview_scales`.
    * AAontology: :ref:`t3a_aaontology_categories` and :ref:`t3b_aaontology_subcategories` tables.
    * Step-by-step guide in the `Scale Loading Tutorial <tutorial2b_scales_loader.html>`_.
    * :class:`AAclust` for customizing redundancy-reduced scale sets.
    """
    check_name_of_scale(name=name)
    ut.check_bool(name="just_aaindex", val=just_aaindex)
    ut.check_bool(name="unclassified_in", val=unclassified_out)
    check_top60_n(name=name, top60_n=top60_n)

    # Load and filter top60 scales
    if top60_n is not None:
        selected_scales = _get_selected_scales(top60_n=top60_n)
        if name == ut.STR_SCALE_CAT:
            df_cat = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
            df_cat = df_cat[df_cat[ut.COL_SCALE_ID].isin(selected_scales)].reset_index(drop=True)
            return df_cat
        elif name in [ut.STR_SCALES, ut.STR_SCALES_RAW]:
            df = ut.read_excel_cached(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
            df = df[selected_scales]
            return df
        else:
            raise ValueError(f"Wrong 'name' ('{name}') for 'top60_n")

    # Load unfiltered data
    if not unclassified_out and not just_aaindex:
        if name == ut.STR_SCALE_CAT:
            df_cat = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
            return df_cat
        df = ut.read_excel_cached(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
        return df

    # Load and filter scale categories
    df_cat = ut.read_excel_cached(ut.FOLDER_DATA + f"{ut.STR_SCALE_CAT}.xlsx")
    df_cat = _filter_scales(df_cat=df_cat, unclassified_out=unclassified_out, just_aaindex=just_aaindex)
    if name == ut.STR_SCALE_CAT:
        return df_cat.reset_index(drop=True)

    # Load and filter scales
    df = ut.read_excel_cached(ut.FOLDER_DATA + name + ".xlsx", index_col=0)
    if name in [ut.STR_SCALES, ut.STR_SCALES_RAW]:
        selected_scales = [x for x in list(df) if x in list(df_cat[ut.COL_SCALE_ID])]
        df = df[selected_scales]
    return df
