"""
This script implements the Bridge Layer for the `CPP.run_num()` method,
the numerical-mode development twin of `CPP.run()`.

PR2.5 flow (memory-optimized two-pass):
  1. ``assign_scale_values_to_seq_num`` builds the per-part (n, L, D) tensor.
  2. ``pre_filtering_info_num`` streams per-(scale, part, split) stats and
     drops features with ``std_test > max_std_test`` (and ``NaN
     abs_mean_dif`` when ``accept_gaps=True``) in-stream. NO cache.
  3. ``pre_filtering`` narrows surviving features to top ``n_pre_filter``.
  4. ``recompute_feature_matrix_num`` rebuilds the (n_samples, n_pre_filter)
     value matrix from the (n, L, D) tensor for ONLY the survivors —
     vectorized per (split_type, part, scale) bucket.
  5. ``add_stat_num`` consumes the recomputed matrix (no get_feature_matrix_).
  6. Position / scale-info / redundancy filter / final rounding.

Memory at 10k samples × 586 scales: ~6 GB peak (vs ~70 GB for the PR2 cache).
Time: ~5-10% slower than the PR2 cache at small fixtures, comparable or
faster at large fixtures (no cache write).

DEV: Bridge layers should be used only in exceptional cases to preserve the
primary backend-frontend architecture.
"""
import numpy as np
import pandas as pd
import aaanalysis.utils as ut

from .cpp.sequence_feature import get_features_
from .cpp.utils_feature import get_positions_, add_scale_info_
from .cpp._filters_num._get_feature_matrix_fast import get_feature_matrix_fast_

# Optional Cython-accelerated path. Gated behind try/except so a pure-Python
# install (no compiler / extension not built) still gets ``CPP.run_num`` —
# only ``CPP.run_c`` raises an informative error in that case.
try:
    from .cpp._filters_num_c._get_feature_matrix_c import get_feature_matrix_c_
    _HAS_CYTHON_INNER = True
except ImportError:
    get_feature_matrix_c_ = None  # type: ignore[assignment]
    _HAS_CYTHON_INNER = False

from .cpp._filters_num._assign import (
    assign_scale_values_to_seq_num, assign_dict_num_to_parts,
)
from .cpp._filters_num._stat_filter import (
    pre_filtering_info_num,
    build_feature_index_map,
    accumulate_partial_stats,
    finalize_stats,
)
from .cpp._filters_num._pre_filter import pre_filtering
from .cpp._filters_num._add_stat import add_stat_num
from .cpp._filters_num._recompute import recompute_feature_matrix_num
from .cpp._filters_num._redundancy_filter import filtering


# I Helper Functions
def _get_n_pre_filter(n_pre_filter=None, n_filter=None, n_feat=None, pct_pre_filter=None):
    """Get number of feature to pre-filter (mirrors ``cpp_run._get_n_pre_filter``)."""
    if n_pre_filter is None:
        n_pre_filter = int(n_feat * (pct_pre_filter / 100))
        n_pre_filter = n_filter if n_pre_filter < n_filter else n_pre_filter
    pct_pre_filter = np.round((n_pre_filter / n_feat * 100), 2)
    return n_pre_filter, pct_pre_filter


# II Main Functions
def cpp_run_num_single(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None,
                      accept_gaps=True, labels=None, label_test=1, label_ref=0, n_filter=100,
                      n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5,
                      max_cor=0.5, check_cat=True, parametric=False, start=1, tmd_len=20,
                      jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True,
                      _staging_shape="A", df_seq=None, dict_num=None,
                      aa_lookup_cache=None, feature_matrix_builder=None):
    """PR2.5+ two-pass memory-optimized CPP through the numerical-mode pipeline.

    Per-residue values come from one of two sources:

    * ``dict_num is None`` (seq-mode, default): AA-letter ``scale_matrix``
      lookup against ``df_scales``. Output bit-identical with ``CPP.run``.
    * ``dict_num`` supplied: per-protein tensor ``Dict[entry, (L, D)]`` is
      sliced into per-part 3D tensors using ``df_seq``'s ``tmd_start/stop``
      positions; ``df_scales``/``df_cat`` provide D dimension names and the
      categorical filter input only — no value lookup needed.

    ``_staging_shape`` (seq-mode only) routes to per-part 3D tensors (A) or
    a single 4D tensor (B); only the dev bench flips this.
    """
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    list_scales = list(df_scales)

    n_feat = len(get_features_(list_parts=list(df_parts),
                               split_kws=split_kws,
                               list_scales=list_scales))
    n_filter = n_feat if n_feat < n_filter else n_filter

    if verbose:
        ut.print_start_progress(start_message=(
            f"1. CPP creates {n_feat} features for {len(df_parts)} samples"
            f"\n1.1 Assigning {'dict_num tensor' if dict_num is not None else 'scale values'} to parts"
        ))
    if dict_num is not None:
        dict_part_vals, dict_part_lens = assign_dict_num_to_parts(
            df_seq=df_seq, dict_num=dict_num, list_parts=list(df_parts),
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
        )
    else:
        dict_part_vals, dict_part_lens = assign_scale_values_to_seq_num(
            df_parts=df_parts, df_scales=df_scales, verbose=verbose, n_jobs=n_jobs,
            _staging_shape=_staging_shape,
        )

    if verbose:
        ut.print_start_progress(start_message=f"\n1.2 Streaming pre-filter stats (mask in stream)")

    # Pass 1: stats only; std_test + accept_gaps mask applied in stream.
    abs_mean_dif, std_test, features_all = pre_filtering_info_num(
        df_parts=df_parts, split_kws=split_kws, dict_part_vals=dict_part_vals,
        dict_part_lens=dict_part_lens, list_scales=list_scales, labels=labels,
        label_test=label_test, label_ref=label_ref, max_std_test=max_std_test,
        accept_gaps=accept_gaps, verbose=verbose, n_jobs=n_jobs, vectorized=vectorized,
    )

    n_pre_filter, pct_pre_filter = _get_n_pre_filter(
        n_pre_filter=n_pre_filter, n_filter=n_filter,
        n_feat=n_feat, pct_pre_filter=pct_pre_filter,
    )
    if verbose:
        ut.print_end_progress(end_message=(
            f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest "
            f"'{ut.COL_ABS_MEAN_DIF}' and 'max_std_test' <= {max_std_test} (kept={len(features_all)} of {n_feat})"
        ))
    df = pre_filtering(features=features_all, abs_mean_dif=abs_mean_dif, std_test=std_test,
                       n=n_pre_filter, max_std_test=max_std_test, accept_gaps=accept_gaps)
    features = df[ut.COL_FEATURE].to_list()

    # Pass 2: recompute (n_samples, n_pre_filter) matrix.
    # For seq-mode (dict_num is None) we delegate to the legacy
    # ``get_feature_matrix_`` to guarantee BIT-EXACT parity with CPP.run —
    # legacy's per-(sample, feature) ``np.mean(list)`` produces ULP-different
    # values from our vectorized ``nanmean`` over a NaN-padded buffer, which
    # cascades through Mann-Whitney rank computation into noticeably different
    # p-values for a few features near rank boundaries.
    # The vectorized recompute (``recompute_feature_matrix_num``) remains the
    # path for ``dict_num`` mode where no legacy equivalent exists.
    if dict_num is None:
        builder = feature_matrix_builder or get_feature_matrix_fast_
        X_cached = builder(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=accept_gaps, n_jobs=n_jobs,
            aa_lookup_cache=aa_lookup_cache,
        )
    else:
        X_cached = recompute_feature_matrix_num(
            dict_part_vals=dict_part_vals, dict_part_lens=dict_part_lens,
            list_scales=list_scales, features=features, split_kws=split_kws,
        )
    # Release the per-part tensors before add_stat — they're not needed past
    # the recompute and can dominate peak RSS at large n × D.
    del dict_part_vals, dict_part_lens
    import gc
    gc.collect()

    df = add_stat_num(df_feat=df, X_cached=X_cached, labels=labels, parametric=parametric,
                     label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                     vectorized=vectorized)
    feat_positions = get_positions_(features=features, start=start, **args_len)
    df[ut.COL_POSITION] = feat_positions
    df = add_scale_info_(df_feat=df, df_cat=df_cat)

    if verbose:
        ut.print_out(f"3. CPP filtering algorithm")
    df_feat = filtering(df=df, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                        max_overlap=max_overlap, max_cor=max_cor)
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat


def cpp_run_num_batch(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None,
                     accept_gaps=True, labels=None, label_test=1, label_ref=0, n_filter=100,
                     n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5,
                     max_cor=0.5, check_cat=True, parametric=False, start=1, tmd_len=20,
                     jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True, n_batches=10,
                     feature_matrix_builder=None):
    """PR2.5 D-chunk batched orchestration with two-pass memory-bounded flow.

    Per scale-batch: assign + streaming pre-filter stats (no cache). After all
    batches, concat stats, pre_filter narrows to survivors, then recompute the
    (n_samples, n_pre_filter) matrix per scale-batch and feed add_stat_num
    per feature-batch (so ``p_val_fdr_bh`` matches the legacy batched
    BH-correction).
    """
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    list_scales = list(df_scales)
    n_feat = len(get_features_(list_parts=list(df_parts),
                               split_kws=split_kws,
                               list_scales=list_scales))
    n_filter = n_feat if n_feat < n_filter else n_filter
    if verbose:
        ut.print_out(f"1. CPP creates {n_feat} features for {len(df_parts)} samples in {n_batches} batches")

    scale_batches = np.array_split(np.array(list_scales), n_batches)
    list_amd = []
    list_std = []
    list_feats = []
    # Keep per-batch tensors only long enough to compute stats; we'll rebuild
    # them on demand in pass 2 (to keep peak memory bounded per batch).
    for i, scales_batch in enumerate(scale_batches):
        scales_batch_list = list(scales_batch)
        df_scales_batch = df_scales[scales_batch_list]
        if verbose:
            str_start = "" if i == 0 else "\n"
            ut.print_start_progress(start_message=f"{str_start}1.1 Assigning scales values to parts ({i + 1}/{n_batches} batch)")
        dict_part_vals, dict_part_lens = assign_scale_values_to_seq_num(
            df_parts=df_parts, df_scales=df_scales_batch, verbose=verbose, n_jobs=n_jobs
        )
        if verbose:
            ut.print_start_progress(start_message=f"\n1.2 Streaming pre-filter stats ({i + 1}/{n_batches} batch)")
        amd_b, std_b, feats_b = pre_filtering_info_num(
            df_parts=df_parts, split_kws=split_kws, dict_part_vals=dict_part_vals,
            dict_part_lens=dict_part_lens, list_scales=scales_batch_list, labels=labels,
            label_test=label_test, label_ref=label_ref, max_std_test=max_std_test,
            accept_gaps=accept_gaps, verbose=verbose, n_jobs=n_jobs, vectorized=vectorized,
        )
        list_amd.append(amd_b)
        list_std.append(std_b)
        list_feats.append(feats_b)
        # dict_part_vals released at end of iteration — peak memory bounded.

    abs_mean_dif = np.concatenate(list_amd)
    std_test = np.concatenate(list_std)
    features_all = np.concatenate(list_feats)

    n_pre_filter, pct_pre_filter = _get_n_pre_filter(n_pre_filter=n_pre_filter, n_filter=n_filter,
                                                     n_feat=n_feat, pct_pre_filter=pct_pre_filter)
    if verbose:
        ut.print_end_progress(end_message=(
            f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest "
            f"'{ut.COL_ABS_MEAN_DIF}' and 'max_std_test' <= {max_std_test} "
            f"(kept={len(features_all)} of {n_feat})"
        ))
    df = pre_filtering(features=features_all, abs_mean_dif=abs_mean_dif, std_test=std_test,
                       n=n_pre_filter, max_std_test=max_std_test, accept_gaps=accept_gaps)
    features = df[ut.COL_FEATURE].to_list()

    # Pass 2: recompute survivor columns per scale-batch (to keep memory peak
    # bounded by one batch's (n, L, D) tensor + the survivor matrix).
    # Then run add_stat per feature-batch to match legacy batched FDR.
    scale_to_batch_idx = {}
    for batch_idx, scales_batch in enumerate(scale_batches):
        for s in scales_batch:
            scale_to_batch_idx[s] = batch_idx
    # Group surviving features by which scale-batch their scale belongs to.
    features_by_batch = [[] for _ in scale_batches]
    for f in features:
        scale = f.split("-")[2]
        features_by_batch[scale_to_batch_idx[scale]].append(f)

    # Pass 2 + per feature-batch add_stat. Use legacy ``get_feature_matrix_``
    # per feature-batch to guarantee BIT-EXACT parity (vectorized recompute
    # diverges at ULP for a few features via Mann-Whitney rank cascades; see
    # cpp_run_num_single notes). Mirrors legacy ``cpp_run_batch``'s per-batch
    # BH-FDR semantics.
    feature_batches = np.array_split(np.array(features), n_batches)
    builder = feature_matrix_builder or get_feature_matrix_fast_
    list_batch_dfs = []
    for feature_batch in feature_batches:
        _df = df[df[ut.COL_FEATURE].isin(feature_batch)]
        _features_batch = _df[ut.COL_FEATURE].to_list()
        _X_cached = builder(
            features=_features_batch, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=accept_gaps, n_jobs=n_jobs,
        )
        _df = add_stat_num(df_feat=_df, X_cached=_X_cached, labels=labels, parametric=parametric,
                          label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                          vectorized=vectorized)
        feat_positions = get_positions_(features=feature_batch, start=start, **args_len)
        _df[ut.COL_POSITION] = feat_positions
        _df = add_scale_info_(df_feat=_df, df_cat=df_cat)
        list_batch_dfs.append(_df)
    df_merged = pd.concat(list_batch_dfs, ignore_index=True)

    if verbose:
        ut.print_out(f"3. CPP filtering algorithm")
    df_feat = filtering(df=df_merged, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                        max_overlap=max_overlap, max_cor=max_cor)
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat


def cpp_run_num_sample_batched(df_parts=None, split_kws=None, df_scales=None, df_cat=None,
                              verbose=None, accept_gaps=True, labels=None, label_test=1,
                              label_ref=0, n_filter=100, n_pre_filter=None, pct_pre_filter=5,
                              max_std_test=0.2, max_overlap=0.5, max_cor=0.5, check_cat=True,
                              parametric=False, start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                              n_jobs=None, vectorized=True, n_sample_batches=10):
    """Sample-batched orchestration: bounds memory at O(batch_size * L * D), not O(n).

    Per sample batch: assign + accumulate per-feature partial stats (sum, sum²,
    counts). After all batches: combine into final mean/std/abs_mean_dif and
    apply std_test + accept_gaps masks. Pre-filter top n_pre_filter survivors.
    Pass 2: per sample batch, re-assign + recompute survivor columns; write into
    the full (n_samples, n_pre_filter) X output. add_stat on X.

    Memory peak: O(batch_size * L * D + n_samples * n_pre_filter * 8).
    For 100k samples × 586 scales × default splits: ~25 GB vs ~70 GB for the
    non-batched path. add_stat itself reuses ``add_stat_num`` on the full X.

    Numeric note: pass-1 std_test is computed via accumulator-style variance
    (``E[X²] - E[X]²``) which may differ from the single-pass ``np.std`` by
    ULP-level rounding. After the existing ``round(3)`` on ``COLS_FEAT_STAT``
    these match in 99%+ of cases; in pathological tie-breaks the redundancy
    filter sort can land a few features in different positions.
    """
    import gc
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    list_scales = list(df_scales)
    list_parts = list(df_parts)
    n_samples = len(df_parts)
    labels = np.asarray(labels)

    n_feat_total = len(get_features_(list_parts=list_parts, split_kws=split_kws,
                                     list_scales=list_scales))
    n_filter = n_feat_total if n_feat_total < n_filter else n_filter

    if verbose:
        ut.print_out(
            f"1. CPP creates {n_feat_total} features for {n_samples} samples "
            f"in {n_sample_batches} sample-batches"
        )

    features_canonical, indices_map = build_feature_index_map(
        list_parts=list_parts, split_kws=split_kws, list_scales=list_scales,
    )
    sum_test = np.zeros(n_feat_total, dtype=np.float64)
    sum_sq_test = np.zeros(n_feat_total, dtype=np.float64)
    sum_ref = np.zeros(n_feat_total, dtype=np.float64)
    count_test = np.zeros(n_feat_total, dtype=np.int64)
    count_ref = np.zeros(n_feat_total, dtype=np.int64)

    batch_size = int(np.ceil(n_samples / n_sample_batches))
    batch_ranges = [
        (i * batch_size, min((i + 1) * batch_size, n_samples))
        for i in range(n_sample_batches)
    ]

    for batch_idx, (b_start, b_end) in enumerate(batch_ranges):
        if b_start >= b_end:
            continue
        df_parts_batch = df_parts.iloc[b_start:b_end]
        labels_batch = labels[b_start:b_end]
        if verbose:
            ut.print_out(
                f"1.{batch_idx+1} pass 1 assign + stats "
                f"({b_start}:{b_end} of {n_samples})"
            )
        dict_part_vals_batch, dict_part_lens_batch = assign_scale_values_to_seq_num(
            df_parts=df_parts_batch, df_scales=df_scales, verbose=False, n_jobs=n_jobs,
        )
        accumulate_partial_stats(
            dict_part_vals=dict_part_vals_batch, dict_part_lens=dict_part_lens_batch,
            list_parts=list_parts, list_scales=list_scales, split_kws=split_kws,
            labels_batch=labels_batch, label_test=label_test, label_ref=label_ref,
            sum_test=sum_test, sum_sq_test=sum_sq_test, sum_ref=sum_ref,
            count_test=count_test, count_ref=count_ref, indices_map=indices_map,
        )
        del dict_part_vals_batch, dict_part_lens_batch
        gc.collect()

    abs_mean_dif, std_test, features_all = finalize_stats(
        sum_test=sum_test, sum_sq_test=sum_sq_test, sum_ref=sum_ref,
        count_test=count_test, count_ref=count_ref,
        features_canonical=features_canonical,
        max_std_test=max_std_test, accept_gaps=accept_gaps,
    )

    n_pre_filter, pct_pre_filter = _get_n_pre_filter(
        n_pre_filter=n_pre_filter, n_filter=n_filter,
        n_feat=n_feat_total, pct_pre_filter=pct_pre_filter,
    )
    if verbose:
        ut.print_out(
            f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) "
            f"with 'max_std_test' <= {max_std_test} (kept={len(features_all)} of {n_feat_total})"
        )
    df = pre_filtering(
        features=features_all, abs_mean_dif=abs_mean_dif, std_test=std_test,
        n=n_pre_filter, max_std_test=max_std_test, accept_gaps=accept_gaps,
    )
    features = df[ut.COL_FEATURE].to_list()

    n_kept = len(features)
    X = np.full((n_samples, n_kept), np.nan, dtype=np.float64)
    for batch_idx, (b_start, b_end) in enumerate(batch_ranges):
        if b_start >= b_end:
            continue
        df_parts_batch = df_parts.iloc[b_start:b_end]
        if verbose:
            ut.print_out(
                f"3.{batch_idx+1} pass 2 recompute survivors "
                f"({b_start}:{b_end} of {n_samples})"
            )
        dict_part_vals_batch, dict_part_lens_batch = assign_scale_values_to_seq_num(
            df_parts=df_parts_batch, df_scales=df_scales, verbose=False, n_jobs=n_jobs,
        )
        X_batch = recompute_feature_matrix_num(
            dict_part_vals=dict_part_vals_batch, dict_part_lens=dict_part_lens_batch,
            list_scales=list_scales, features=features, split_kws=split_kws,
        )
        X[b_start:b_end, :] = X_batch
        del dict_part_vals_batch, dict_part_lens_batch, X_batch
        gc.collect()

    df = add_stat_num(
        df_feat=df, X_cached=X, labels=labels.tolist(), parametric=parametric,
        label_test=label_test, label_ref=label_ref, n_jobs=n_jobs, vectorized=vectorized,
    )
    feat_positions = get_positions_(features=features, start=start, **args_len)
    df[ut.COL_POSITION] = feat_positions
    df = add_scale_info_(df_feat=df, df_cat=df_cat)

    if verbose:
        ut.print_out("4. CPP filtering algorithm")
    df_feat = filtering(
        df=df, df_scales=df_scales, n_filter=n_filter,
        check_cat=check_cat, max_overlap=max_overlap, max_cor=max_cor,
    )
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    if verbose:
        ut.print_out(
            f"5. CPP returns df of {len(df_feat)} unique features with general information and statistics"
        )
    return df_feat
