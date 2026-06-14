"""
This is a script for the backend of CPP's redundancy-reduction stage:
``filtering`` performs greedy selection in descending order of absolute AUC,
dropping features that overlap (by position and correlation) with previously
accepted features.

Greedy selection is independent of the upstream value source — both
seq-mode (``cpp.run``) and numerical-mode (``cpp.run_num``) consume the
same implementation.

# DEV: greedy selection is inherently sequential. A precompute-vectorize
# pass (per-pair overlap + scale-correlation matrices) was attempted and
# reverted: materializing (n_pre_filter, n_pre_filter) matrices upfront
# turned out to be ~44x SLOWER than the current "early-exit at n_filter"
# greedy loop at default settings (``n_filter=100``), because the early
# exit cuts pair checks dramatically. See CPP_RUN_NUM_BACKLOG.md for the
# data.
"""
import aaanalysis.utils as ut


# I Helper Functions
def filtering_info_(df=None, df_scales=None, check_cat=True):
    """Get datasets structures for filtering."""
    # DEV: ``check_cat`` controls scale-category-aware redundancy gating.
    # When True, ``dict_c`` maps every feature id to its scale category so the
    # filtering loop only compares features that share a category; when False,
    # ``dict_c`` stays empty and the loop's ``not check_cat`` short-circuit
    # skips the category gate entirely (so the empty dict is never indexed).
    # ``dict_c`` is built from the same ``df`` the loop iterates, so its keys
    # always cover every candidate feature (no KeyError is possible). A null
    # category cell (only reachable via a user-supplied ``df_cat`` with a NaN
    # category) compares unequal to everything (NaN != NaN), so such a feature
    # is treated as its own category and is never dropped as redundant.
    if check_cat:
        dict_c = dict(zip(df[ut.COL_FEATURE], df[ut.COL_CAT]))
    else:
        dict_c = dict()
    dict_p = dict(zip(df[ut.COL_FEATURE], [set(x) for x in df[ut.COL_POSITION]]))
    df_cor = df_scales.corr()
    return dict_c, dict_p, df_cor


# II Main Functions
def filtering(df=None, df_scales=None, max_overlap=0.5, max_cor=0.5, n_filter=100, check_cat=True):
    """CPP filtering algorithm based on redundancy reduction in descending order of absolute AUC."""
    dict_c, dict_p, df_cor = filtering_info_(df=df, df_scales=df_scales, check_cat=check_cat)
    df = df.sort_values(by=[ut.COL_ABS_AUC, ut.COL_ABS_MEAN_DIF], ascending=False).copy().reset_index(drop=True)
    list_feat = list(df[ut.COL_FEATURE])
    list_top_feat = [list_feat.pop(0)]
    for feat in list_feat:
        add_flag = True
        if len(list_top_feat) == n_filter:
            break
        for top_feat in list_top_feat:
            if not check_cat or dict_c[feat] == dict_c[top_feat]:
                pos, top_pos = dict_p[feat], dict_p[top_feat]
                overlap = len(top_pos.intersection(pos)) / len(top_pos.union(pos))
                if overlap >= max_overlap or pos.issubset(top_pos):
                    scale = ut.split_feat_id(feat_id=feat)[2]
                    top_scale = ut.split_feat_id(feat_id=top_feat)[2]
                    cor = df_cor[top_scale][scale]
                    if cor > max_cor:
                        add_flag = False
        if add_flag:
            list_top_feat.append(feat)
    return df[df[ut.COL_FEATURE].isin(list_top_feat)]
