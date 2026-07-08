"""
This script implements the Bridge Layer for the `CPP.run_num()` method,
the numerical-mode development twin of `CPP.run()`.

PR2.5 flow (memory-optimized two-pass):
  1. ``assign_scale_values_to_seq`` builds the per-part (n, L, D) tensor.
  2. ``pre_filtering_info`` streams per-(scale, part, split) stats and
     drops features with ``std_test > max_std_test`` (and ``NaN
     abs_mean_dif`` when ``accept_gaps=True``) in-stream. NO cache.
  3. ``pre_filtering`` narrows surviving features to top ``n_pre_filter``.
  4. ``recompute_feature_matrix`` rebuilds the (n_samples, n_pre_filter)
     value matrix from the (n, L, D) tensor for ONLY the survivors —
     vectorized per (split_type, part, scale) bucket.
  5. ``add_stat`` consumes the recomputed matrix (no get_feature_matrix_).
  6. Position / scale-info / redundancy filter / final rounding.

Memory at 10k samples × 586 scales: ~6 GB peak (vs ~70 GB for the PR2 cache).
Time: ~5-10% slower than the PR2 cache at small fixtures, comparable or
faster at large fixtures (no cache write).

DEV: Bridge layers should be used only in exceptional cases to preserve the
primary backend-frontend architecture.
"""
import gc
import warnings

import numpy as np
import pandas as pd
import aaanalysis.utils as ut

from .cpp.sequence_feature import get_features_
from .cpp.utils_feature import get_positions_, add_scale_info_
from .cpp._filters._get_feature_matrix_fast import get_feature_matrix_fast_

# Optional Cython-accelerated path. Gated behind try/except so installs
# without the compiled extension (e.g. unsupported platform with no
# prebuilt wheel) silently fall back to the pure-Python kernel via
# ``_pick_feature_matrix_builder``.
try:
    from .cpp._filters_c._get_feature_matrix_c import get_feature_matrix_c_
    _HAS_CYTHON_INNER = True
except ImportError:
    get_feature_matrix_c_ = None  # type: ignore[assignment]
    _HAS_CYTHON_INNER = False

from .cpp._filters._assign import (
    assign_scale_values_to_seq, assign_dict_num_to_parts,
)
from .cpp._filters._stat_filter import (
    pre_filtering_info,
    build_feature_index_map,
    accumulate_partial_stats,
    finalize_stats,
)
from .cpp._filters._pre_filter import pre_filtering
from .cpp._filters._add_stat import add_stat
from .cpp._filters._recompute import recompute_feature_matrix
from .cpp._filters._redundancy_filter import filtering


# I Helper Functions
# Only force a gc.collect() after freeing the per-part (n, L, D) tensors when
# they are large enough for the reclaimed RAM to matter. A full collect costs
# ~25 ms; on small/medium inputs that is pure overhead (it dominated run_num's
# wall-clock on the tiny benchmark fixture), while on a multi-GB tensor it
# meaningfully lowers peak RSS before the add_stat step. 64 MB is the break-even.
_GC_TENSOR_BYTES = 64 * 1024 ** 2


def _part_tensor_nbytes(*objs):
    """Total nbytes across per-part value/length dicts (or bare arrays)."""
    total = 0
    for obj in objs:
        if isinstance(obj, dict):
            total += sum(getattr(v, "nbytes", 0) for v in obj.values())
        else:
            total += getattr(obj, "nbytes", 0)
    return total


def _get_n_pre_filter(n_pre_filter=None, n_filter=None, n_feat=None, pct_pre_filter=None):
    """Get number of feature to pre-filter (mirrors ``cpp_run._get_n_pre_filter``)."""
    if n_pre_filter is None:
        n_pre_filter = int(n_feat * (pct_pre_filter / 100))
        n_pre_filter = n_filter if n_pre_filter < n_filter else n_pre_filter
    pct_pre_filter = np.round((n_pre_filter / n_feat * 100), 2)
    return n_pre_filter, pct_pre_filter


def _attach_filter_stats(df_feat=None, n_candidates=None, n_after_prefilter=None,
                         n_after_redundancy=None, n_requested=None, verbose=False):
    """Attach the CPP filter-funnel counts to ``df_feat.attrs`` and surface a
    feature shortfall.

    ``n_requested`` is the user's pre-cap ``n_filter`` (before it is clipped to
    ``n_candidates``). Two distinct shortfalls are surfaced, mutually exclusive
    so the run never double-reports:

    * **D5b — sparse config (``UserWarning``):** when ``n_candidates`` itself is
      below ``n_filter``, the ``split_kws`` × parts × scales expansion cannot
      generate enough features *regardless of filtering* (the small-``n_jmd`` /
      narrow-``split_kws`` footgun). This is an input-shaped issue the user can
      fix, so it stays a warning and points at ``split_kws`` / part lengths.
    * **D7 — filter shortfall (info, ``verbose``-gated):** enough candidates
      existed, but the pre-filter / redundancy steps trimmed to fewer than
      requested. This is normal filter behaviour, not a fault, so it is plain
      user-facing information printed via ``print_out`` only when ``verbose``
      — never a warning.

    The stats are a plain dict — no typed record, per the backend house style.
    """
    # Standardize the df_feat column order (issue #18). This is the single common
    # tail of all CPP run/run_num orchestrators, so reordering here covers every
    # path. Reorder BEFORE the attrs write so ``last_filter_stats`` survives.
    df_feat = ut.sort_cols_feat(df_feat=df_feat)
    n_final = len(df_feat)
    df_feat.attrs["last_filter_stats"] = {
        "n_candidates": int(n_candidates),
        "n_after_prefilter": int(n_after_prefilter),
        "n_after_redundancy": int(n_after_redundancy),
        "n_final": int(n_final),
    }
    if n_requested is not None and n_candidates < n_requested:
        warnings.warn(
            f"'n_filter' ({n_requested}) should be <= the number of candidate "
            f"features the configuration can generate ({n_candidates}); the "
            f"'split_kws' × parts × 'df_scales' expansion is too sparse for "
            f"these part lengths (small 'n_jmd' / 'tmd_len' or narrow "
            f"'split_kws'). Adjust 'split_kws' (e.g. 'len_max'/'steps'), enlarge "
            f"the parts, or lower 'n_filter'.",
            UserWarning,
        )
    elif verbose and n_requested is not None and n_final < n_requested:
        ut.print_out(
            f"Note: returning {n_final} features, fewer than the requested "
            f"'n_filter' ({n_requested}); the redundancy filter trimmed the rest. "
            f"Inspect df_feat.attrs['last_filter_stats'], or relax "
            f"'max_overlap'/'max_cor' (or enlarge 'df_scales' / 'n_jmd') to keep "
            f"more."
        )
    return df_feat


# Module-level guard so the "using Python fallback" notice fires at most once
# per process. Users with a properly-built wheel never see it.
_PYTHON_FALLBACK_NOTIFIED = False


def _pick_feature_matrix_builder():
    """Return Cython builder if the compiled `.so` is importable, else Python.

    Public ``CPP.run`` and ``CPP.run_num`` both call this — neither exposes a
    user-facing backend switch (the choice is determined entirely by wheel
    state at import time). One-time ``UserWarning`` on first fallback use so a
    pure-Python install isn't silent.
    """
    global _PYTHON_FALLBACK_NOTIFIED
    if _HAS_CYTHON_INNER:
        return get_feature_matrix_c_
    if not _PYTHON_FALLBACK_NOTIFIED:
        # A real warning (not an INFO print) so it surfaces even with
        # ``aa.options['verbose'] = False`` — a silent INFO let users blame the
        # algorithm for ~2x-slower runs instead of reinstalling for the wheel.
        warnings.warn(
            "CPP is using the Python kernel fallback — the compiled Cython "
            "extension is not available in this install. Output is bit-exact "
            "with the Cython path but ~2x slower. Reinstall via "
            "`pip install --force-reinstall aaanalysis` to fetch a prebuilt "
            "wheel.",
            UserWarning,
        )
        _PYTHON_FALLBACK_NOTIFIED = True
    return get_feature_matrix_fast_


def _resample_row_indices(idx_test=None, idx_ref=None, resample="reference",
                          bootstrap_frac=0.8, rng=None):
    """Draw one bootstrap resample of row indices (sampling with replacement per group).

    ``idx_test`` / ``idx_ref`` are the row positions of the test / reference group in the full
    ``labels`` vector. Per ``resample``, one or both groups are resampled with replacement to
    ``round(bootstrap_frac * n_group)`` rows (at least 1); the other group is passed through
    unchanged. ``resample='reference'`` fixes the test group and resamples only the reference,
    isolating the dominant instability source.
    """
    def _draw(idx):
        size = max(1, int(round(bootstrap_frac * len(idx))))
        return rng.choice(idx, size=size, replace=True)

    if resample == ut.RESAMPLE_BOTH:
        test_s, ref_s = _draw(idx_test), _draw(idx_ref)
    elif resample == ut.RESAMPLE_REFERENCE:
        test_s, ref_s = idx_test, _draw(idx_ref)
    else:  # ut.RESAMPLE_TEST
        test_s, ref_s = _draw(idx_test), idx_ref
    return np.concatenate([test_s, ref_s])


# II Main Functions
def cpp_run_single(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None,
                      accept_gaps=True, labels=None, label_test=1, label_ref=0, n_filter=100,
                      n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5,
                      max_cor=0.5, check_cat=True, redundancy="legacy", parametric=False, start=1, tmd_len=20,
                      jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True,
                      _staging_shape="A", df_seq=None, dict_num=None,
                      dict_part_vals=None, dict_part_lens=None,
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
    n_requested = n_filter
    n_filter = n_feat if n_feat < n_filter else n_filter

    # PR5: numerical-mode input has THREE possible shapes:
    #   1. Pre-sliced  (dict_part_vals + dict_part_lens supplied)        ← CPP.run_num path
    #   2. Raw         (dict_num supplied; sliced via assign_dict_num_to_parts)
    #   3. Seq-mode    (neither supplied; assigned from df_parts + df_scales) ← CPP.run path
    has_pre_sliced = dict_part_vals is not None and dict_part_lens is not None
    has_raw_dict_num = dict_num is not None
    if verbose:
        ut.print_start_progress(start_message=(
            f"1. CPP creates {n_feat} features for {len(df_parts)} samples"
            f"\n1.1 Assigning "
            f"{'pre-sliced dict_num_parts' if has_pre_sliced else 'dict_num tensor' if has_raw_dict_num else 'scale values'}"
            f" to parts"
        ))
    if has_pre_sliced:
        # NumericalFeature.get_parts produced these — no further work needed.
        pass
    elif has_raw_dict_num:
        dict_part_vals, dict_part_lens = assign_dict_num_to_parts(
            df_seq=df_seq, dict_num=dict_num, list_parts=list(df_parts),
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
        )
    else:
        dict_part_vals, dict_part_lens = assign_scale_values_to_seq(
            df_parts=df_parts, df_scales=df_scales, verbose=verbose, n_jobs=n_jobs,
            _staging_shape=_staging_shape,
        )

    if verbose:
        ut.print_start_progress(start_message=f"\n1.2 Streaming pre-filter stats (mask in stream)")

    # Pass 1: stats only; std_test + accept_gaps mask applied in stream.
    abs_mean_dif, std_test, features_all = pre_filtering_info(
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
    # For seq-mode (no numerical input) we delegate to the bit-exact
    # ``get_feature_matrix_fast_`` / ``get_feature_matrix_c_`` builder —
    # the per-(sample, feature) ``np.mean(list)`` summation matches legacy
    # ``CPP.run`` exactly. The vectorized recompute
    # (``recompute_feature_matrix``) handles numerical-mode where the
    # AA→scale lookup doesn't apply (dict_num provides per-residue values
    # directly).
    is_numerical_mode = has_pre_sliced or has_raw_dict_num
    if not is_numerical_mode:
        builder = feature_matrix_builder or get_feature_matrix_fast_
        X_cached = builder(
            features=features, df_parts=df_parts, df_scales=df_scales,
            accept_gaps=accept_gaps, n_jobs=n_jobs,
            aa_lookup_cache=aa_lookup_cache,
        )
    else:
        X_cached = recompute_feature_matrix(
            dict_part_vals=dict_part_vals, dict_part_lens=dict_part_lens,
            list_scales=list_scales, features=features, split_kws=split_kws,
        )
    # Release the per-part tensors before add_stat — they're not needed past
    # the recompute and can dominate peak RSS at large n × D. Only force a
    # collect when they are large (see _GC_TENSOR_BYTES); on small inputs the
    # collect costs more than it reclaims.
    _nbytes = _part_tensor_nbytes(dict_part_vals, dict_part_lens)
    del dict_part_vals, dict_part_lens
    if _nbytes >= _GC_TENSOR_BYTES:
        gc.collect()

    df = add_stat(df_feat=df, X_cached=X_cached, labels=labels, parametric=parametric,
                     label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                     vectorized=vectorized)
    feat_positions = get_positions_(features=features, start=start, **args_len)
    df[ut.COL_POSITION] = feat_positions
    df = add_scale_info_(df_feat=df, df_cat=df_cat)

    if verbose:
        ut.print_out(f"3. CPP filtering algorithm")
    df_feat = filtering(df=df, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                        max_overlap=max_overlap, max_cor=max_cor, redundancy=redundancy)
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    df_feat = _attach_filter_stats(
        df_feat=df_feat, n_candidates=n_feat, n_after_prefilter=len(df),
        n_after_redundancy=len(df_feat), n_requested=n_requested, verbose=verbose,
    )
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat


def cpp_run_batch(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None,
                     accept_gaps=True, labels=None, label_test=1, label_ref=0, n_filter=100,
                     n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5,
                     max_cor=0.5, check_cat=True, redundancy="legacy", parametric=False, start=1, tmd_len=20,
                     jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True, n_batches=10,
                     feature_matrix_builder=None):
    """PR2.5 D-chunk batched orchestration with two-pass memory-bounded flow.

    Per scale-batch: assign + streaming pre-filter stats (no cache). After all
    batches, concat stats, pre_filter narrows to survivors, then recompute the
    (n_samples, n_pre_filter) matrix per scale-batch and feed add_stat
    per feature-batch (so ``p_val_fdr_bh`` matches the legacy batched
    BH-correction).
    """
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    list_scales = list(df_scales)
    n_feat = len(get_features_(list_parts=list(df_parts),
                               split_kws=split_kws,
                               list_scales=list_scales))
    n_requested = n_filter
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
        dict_part_vals, dict_part_lens = assign_scale_values_to_seq(
            df_parts=df_parts, df_scales=df_scales_batch, verbose=verbose, n_jobs=n_jobs
        )
        if verbose:
            ut.print_start_progress(start_message=f"\n1.2 Streaming pre-filter stats ({i + 1}/{n_batches} batch)")
        amd_b, std_b, feats_b = pre_filtering_info(
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
        scale = ut.split_feat_id(feat_id=f)[2]
        features_by_batch[scale_to_batch_idx[scale]].append(f)

    # Pass 2 + per feature-batch add_stat. Use legacy ``get_feature_matrix_``
    # per feature-batch to guarantee BIT-EXACT parity (vectorized recompute
    # diverges at ULP for a few features via Mann-Whitney rank cascades; see
    # cpp_run_single notes). Mirrors legacy ``cpp_run_batch``'s per-batch
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
        _df = add_stat(df_feat=_df, X_cached=_X_cached, labels=labels, parametric=parametric,
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
                        max_overlap=max_overlap, max_cor=max_cor, redundancy=redundancy)
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    df_feat = _attach_filter_stats(
        df_feat=df_feat, n_candidates=n_feat, n_after_prefilter=len(df),
        n_after_redundancy=len(df_feat), n_requested=n_requested, verbose=verbose,
    )
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat


def cpp_run_batch_num(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None,
                         accept_gaps=True, labels=None, label_test=1, label_ref=0, n_filter=100,
                         n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5,
                         max_cor=0.5, check_cat=True, redundancy="legacy", parametric=False, start=1, tmd_len=20,
                         jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True, n_batches=10,
                         dict_part_vals=None, dict_part_lens=None):
    """D-chunk batched orchestration for numerical mode (``CPP.run_num``).

    The seq-mode :func:`cpp_run_batch` bounds memory by *assigning* scale values
    one D-batch at a time. In numerical mode the per-residue values are already
    supplied (``dict_part_vals``: ``{part: (n, L_part, D)}``), so we instead
    *slice* the D axis into ``n_batches`` contiguous chunks (zero-copy views) and
    run the pass-1 streaming pre-filter stats per chunk — bounding the working set
    of the split/stat computation to one chunk's D dimensions.

    Pass 2 (recompute survivor matrix + ``add_stat``) is run **globally** on the
    full tensor, exactly as :func:`cpp_run_single`. Output is therefore bit-exact
    with the single-pass numerical path: per-feature pass-1 stats are independent
    of how D is chunked, ``pre_filtering`` applies a total deterministic sort
    (``abs_mean_dif`` → ``std_test`` → feature name), and pass 2 is identical.
    """
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    list_scales = list(df_scales)
    D = len(list_scales)
    n_feat = len(get_features_(list_parts=list(df_parts),
                               split_kws=split_kws, list_scales=list_scales))
    n_requested = n_filter
    n_filter = n_feat if n_feat < n_filter else n_filter

    # Contiguous D-index chunks -> slicing gives views (no copy of the input).
    idx_chunks = [c for c in np.array_split(np.arange(D), min(n_batches, D)) if len(c)]
    if verbose:
        ut.print_out(f"1. CPP creates {n_feat} features for {len(df_parts)} samples "
                     f"in {len(idx_chunks)} D-batches")
    list_amd, list_std, list_feats = [], [], []
    for i, idx in enumerate(idx_chunks):
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        scales_chunk = list_scales[lo:hi]
        dict_part_vals_chunk = {p: v[:, :, lo:hi] for p, v in dict_part_vals.items()}
        if verbose:
            ut.print_start_progress(start_message=(
                f"\n1.{i + 1} Streaming pre-filter stats (D-batch {i + 1}/{len(idx_chunks)})"))
        amd_b, std_b, feats_b = pre_filtering_info(
            df_parts=df_parts, split_kws=split_kws, dict_part_vals=dict_part_vals_chunk,
            dict_part_lens=dict_part_lens, list_scales=scales_chunk, labels=labels,
            label_test=label_test, label_ref=label_ref, max_std_test=max_std_test,
            accept_gaps=accept_gaps, verbose=verbose, n_jobs=n_jobs, vectorized=vectorized,
        )
        list_amd.append(amd_b)
        list_std.append(std_b)
        list_feats.append(feats_b)
        del dict_part_vals_chunk

    abs_mean_dif = np.concatenate(list_amd)
    std_test = np.concatenate(list_std)
    features_all = np.concatenate(list_feats)

    n_pre_filter, pct_pre_filter = _get_n_pre_filter(
        n_pre_filter=n_pre_filter, n_filter=n_filter, n_feat=n_feat, pct_pre_filter=pct_pre_filter)
    if verbose:
        ut.print_end_progress(end_message=(
            f"2. CPP pre-filters {n_pre_filter} features ({pct_pre_filter}%) with highest "
            f"'{ut.COL_ABS_MEAN_DIF}' and 'max_std_test' <= {max_std_test} "
            f"(kept={len(features_all)} of {n_feat})"))
    df = pre_filtering(features=features_all, abs_mean_dif=abs_mean_dif, std_test=std_test,
                       n=n_pre_filter, max_std_test=max_std_test, accept_gaps=accept_gaps)
    features = df[ut.COL_FEATURE].to_list()

    # Pass 2: global recompute + add_stat (identical to cpp_run_single) -> bit-exact.
    X_cached = recompute_feature_matrix(
        dict_part_vals=dict_part_vals, dict_part_lens=dict_part_lens,
        list_scales=list_scales, features=features, split_kws=split_kws,
    )
    df = add_stat(df_feat=df, X_cached=X_cached, labels=labels, parametric=parametric,
                     label_test=label_test, label_ref=label_ref, n_jobs=n_jobs,
                     vectorized=vectorized)
    feat_positions = get_positions_(features=features, start=start, **args_len)
    df[ut.COL_POSITION] = feat_positions
    df = add_scale_info_(df_feat=df, df_cat=df_cat)

    if verbose:
        ut.print_out(f"3. CPP filtering algorithm")
    df_feat = filtering(df=df, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                        max_overlap=max_overlap, max_cor=max_cor, redundancy=redundancy)
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    df_feat = _attach_filter_stats(
        df_feat=df_feat, n_candidates=n_feat, n_after_prefilter=len(df),
        n_after_redundancy=len(df_feat), n_requested=n_requested, verbose=verbose,
    )
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} unique features with general information and statistics")
    return df_feat


def cpp_run_sample_batched(df_parts=None, split_kws=None, df_scales=None, df_cat=None,
                              verbose=None, accept_gaps=True, labels=None, label_test=1,
                              label_ref=0, n_filter=100, n_pre_filter=None, pct_pre_filter=5,
                              max_std_test=0.2, max_overlap=0.5, max_cor=0.5, check_cat=True,
                              redundancy="legacy",
                              parametric=False, start=1, tmd_len=20, jmd_n_len=10, jmd_c_len=10,
                              n_jobs=None, vectorized=True, n_sample_batches=10,
                              dict_part_vals=None, dict_part_lens=None):
    """Sample-batched orchestration: bounds memory at O(batch_size * L * D), not O(n).

    Per sample batch: get per-part values + accumulate per-feature partial stats
    (sum, sum², counts). After all batches: combine into final mean/std/abs_mean_dif
    and apply std_test + accept_gaps masks. Pre-filter top n_pre_filter survivors.
    Pass 2: per sample batch, re-fetch values + recompute survivor columns; write
    into the full (n_samples, n_pre_filter) X output. add_stat on X.

    Two value sources, one orchestrator:

    * **Sequence mode** (``dict_part_vals`` is None): each batch *assigns* scale
      values from ``df_parts`` strings via ``assign_scale_values_to_seq`` — the
      full tensor is never resident, so peak is O(batch_size * L * D).
    * **Numerical mode** (``CPP.run_num``; ``dict_part_vals`` /
      ``dict_part_lens`` supplied): the full per-part tensor is already resident,
      so each batch *slices* it along the sample axis (zero-copy views). The
      resident input stays O(n * L * D), but the per-batch *working* set (stat
      intermediates + pass-2 recompute) is bounded to O(batch_size), which is the
      dominant term for large numerical runs.

    Numeric note: pass-1 std_test is computed via accumulator-style variance
    (``E[X²] - E[X]²``) which may differ from the single-pass ``np.std`` by
    ULP-level rounding. After the existing ``round(3)`` on ``COLS_FEAT_STAT``
    these match in 99%+ of cases; in pathological tie-breaks the redundancy
    filter sort can land a few features in different positions.
    """
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    list_scales = list(df_scales)
    list_parts = list(df_parts)
    n_samples = len(df_parts)
    labels = np.asarray(labels)

    # Per-batch (vals, lens) provider — slice the resident tensor in numerical
    # mode, else assign from the part strings (the full tensor never materializes).
    is_numerical = dict_part_vals is not None and dict_part_lens is not None

    def _batch_vals(b_start, b_end):
        if is_numerical:
            vals = {p: v[b_start:b_end] for p, v in dict_part_vals.items()}
            lens = {p: l[b_start:b_end] for p, l in dict_part_lens.items()}
            return vals, lens
        return assign_scale_values_to_seq(
            df_parts=df_parts.iloc[b_start:b_end], df_scales=df_scales,
            verbose=False, n_jobs=n_jobs,
        )

    n_feat_total = len(get_features_(list_parts=list_parts, split_kws=split_kws,
                                     list_scales=list_scales))
    n_requested = n_filter
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
        labels_batch = labels[b_start:b_end]
        if verbose:
            ut.print_out(
                f"1.{batch_idx+1} pass 1 assign + stats "
                f"({b_start}:{b_end} of {n_samples})"
            )
        dict_part_vals_batch, dict_part_lens_batch = _batch_vals(b_start, b_end)
        accumulate_partial_stats(
            dict_part_vals=dict_part_vals_batch, dict_part_lens=dict_part_lens_batch,
            list_parts=list_parts, list_scales=list_scales, split_kws=split_kws,
            labels_batch=labels_batch, label_test=label_test, label_ref=label_ref,
            sum_test=sum_test, sum_sq_test=sum_sq_test, sum_ref=sum_ref,
            count_test=count_test, count_ref=count_ref, indices_map=indices_map,
        )
        _nbytes = _part_tensor_nbytes(dict_part_vals_batch, dict_part_lens_batch)
        del dict_part_vals_batch, dict_part_lens_batch
        if _nbytes >= _GC_TENSOR_BYTES:
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
        if verbose:
            ut.print_out(
                f"3.{batch_idx+1} pass 2 recompute survivors "
                f"({b_start}:{b_end} of {n_samples})"
            )
        dict_part_vals_batch, dict_part_lens_batch = _batch_vals(b_start, b_end)
        X_batch = recompute_feature_matrix(
            dict_part_vals=dict_part_vals_batch, dict_part_lens=dict_part_lens_batch,
            list_scales=list_scales, features=features, split_kws=split_kws,
        )
        X[b_start:b_end, :] = X_batch
        _nbytes = _part_tensor_nbytes(dict_part_vals_batch, dict_part_lens_batch, X_batch)
        del dict_part_vals_batch, dict_part_lens_batch, X_batch
        if _nbytes >= _GC_TENSOR_BYTES:
            gc.collect()

    df = add_stat(
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
        redundancy=redundancy,
    )
    df_feat.reset_index(drop=True, inplace=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    df_feat = _attach_filter_stats(
        df_feat=df_feat, n_candidates=n_feat_total, n_after_prefilter=len(df),
        n_after_redundancy=len(df_feat), n_requested=n_requested, verbose=verbose,
    )
    if verbose:
        ut.print_out(
            f"5. CPP returns df of {len(df_feat)} unique features with general information and statistics"
        )
    return df_feat


def cpp_run_bootstrap(df_parts=None, split_kws=None, df_scales=None, df_cat=None, verbose=None,
                      accept_gaps=True, labels=None, label_test=1, label_ref=0, n_filter=100,
                      n_pre_filter=None, pct_pre_filter=5, max_std_test=0.2, max_overlap=0.5,
                      max_cor=0.5, check_cat=True, redundancy="legacy", parametric=False, start=1,
                      tmd_len=20, jmd_n_len=10, jmd_c_len=10, n_jobs=None, vectorized=True,
                      n_bootstrap=20, resample="reference", bootstrap_frac=0.8, min_freq=0.25,
                      random_state=None, aa_lookup_cache=None, feature_matrix_builder=None,
                      dict_part_vals=None, dict_part_lens=None):
    """Bootstrap / stability feature selection wrapped around the single-pass CPP selection.

    Two phases:

    1. **Candidate generation (resampled).** Repeat the single-pass selection
       (:func:`cpp_run_single`) ``n_bootstrap`` times, each on a bootstrap resample of the rows
       (``_resample_row_indices``, with replacement per group per ``resample``). Tally how often
       each feature is selected; ``selection_frequency`` is that count over ``n_bootstrap``. Every
       feature whose ``selection_frequency`` is at least ``min_freq`` (the stability threshold, a
       fraction of the rounds) is a stable candidate.
    2. **Full-data filtering (authoritative).** Statistics for the candidates are (re)computed on
       the **complete** test + reference set, then CPP's own filters decide the output: the
       ``max_std_test`` pre-filter threshold (a feature stable in subsamples but with too-high
       ``std_test`` on the full data is dropped) and the redundancy filter
       (``max_overlap`` / ``max_cor`` / ``n_filter``). The result is ordered by ``abs_auc`` like a
       normal run and carries an extra ``selection_frequency`` column.

    Numerical mode (``dict_part_vals`` / ``dict_part_lens`` supplied) resamples along the sample
    axis of the per-part tensors; sequence mode resamples the ``df_parts`` rows.
    """
    list_scales = list(df_scales)
    args_len = dict(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
    labels_arr = np.asarray(labels)
    is_numerical = dict_part_vals is not None and dict_part_lens is not None

    # Row positions per group. labels are exactly {label_test, label_ref} and both are present
    # (frontend-guaranteed by ut.check_labels), so both index arrays are non-empty.
    idx_test = np.where(labels_arr == label_test)[0]
    idx_ref = np.where(labels_arr == label_ref)[0]
    rng = np.random.default_rng(random_state)

    # Candidate-feature count (parts x split_kws x scales expansion) — stable across rounds; used
    # only for the funnel stats.
    n_feat_total = len(get_features_(list_parts=list(df_parts), split_kws=split_kws,
                                     list_scales=list_scales))

    # --- Phase 1: bootstrap candidate generation ---------------------------------------------
    counts = {}
    for b in range(n_bootstrap):
        if verbose:
            ut.print_out(f"1.{b + 1} CPP bootstrap round {b + 1}/{n_bootstrap} "
                         f"(resample='{resample}', frac={bootstrap_frac})")
        idx_b = _resample_row_indices(idx_test=idx_test, idx_ref=idx_ref, resample=resample,
                                      bootstrap_frac=bootstrap_frac, rng=rng)
        df_parts_b = df_parts.iloc[idx_b].reset_index(drop=True)
        labels_b = labels_arr[idx_b].tolist()
        round_kwargs = dict(
            df_parts=df_parts_b, split_kws=split_kws, df_scales=df_scales, df_cat=df_cat,
            verbose=False, accept_gaps=accept_gaps, labels=labels_b, label_test=label_test,
            label_ref=label_ref, n_filter=n_filter, n_pre_filter=n_pre_filter,
            pct_pre_filter=pct_pre_filter, max_std_test=max_std_test, max_overlap=max_overlap,
            max_cor=max_cor, check_cat=check_cat, redundancy=redundancy, parametric=parametric,
            start=start, tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len, n_jobs=n_jobs,
            vectorized=vectorized, df_seq=None, dict_num=None,
            feature_matrix_builder=feature_matrix_builder,
        )
        if is_numerical:
            df_feat_b = cpp_run_single(
                dict_part_vals={p: v[idx_b] for p, v in dict_part_vals.items()},
                dict_part_lens={p: l[idx_b] for p, l in dict_part_lens.items()},
                aa_lookup_cache=None, **round_kwargs,
            )
        else:
            # aa_lookup_cache is row-aligned to the full df_parts, so it cannot be reused on a
            # resampled subset — the seq-mode builder rebuilds it from df_parts_b.
            df_feat_b = cpp_run_single(aa_lookup_cache=None, **round_kwargs)
        for f in df_feat_b[ut.COL_FEATURE]:
            counts[f] = counts.get(f, 0) + 1

    freq = {f: c / n_bootstrap for f, c in counts.items()}
    # Deterministic rank: selection frequency desc, then feature id asc (reproducible tie-break).
    candidates = sorted((f for f, fr in freq.items() if fr >= min_freq),
                        key=lambda f: (-freq[f], f))
    if verbose:
        ut.print_out(f"2. CPP bootstrap selected {len(freq)} distinct features across "
                     f"{n_bootstrap} rounds; keeping the {len(candidates)} stable ones "
                     f"('selection_frequency' >= 'min_freq'={min_freq}) for full-data filtering")

    # --- Phase 2: full-data statistics + authoritative filtering on the stable candidates ----
    # A high 'min_freq' can leave no feature above the stability threshold; return the
    # schema-correct empty df_feat rather than proceeding into the full-data stat pass.
    if not candidates:
        df_feat = pd.DataFrame(columns=list(ut.LIST_COLS_FEAT) + [ut.COL_SELECTION_FREQUENCY])
        return _attach_filter_stats(
            df_feat=df_feat, n_candidates=n_feat_total, n_after_prefilter=0,
            n_after_redundancy=0, n_requested=None, verbose=verbose,
        )

    df = pd.DataFrame({ut.COL_FEATURE: candidates})
    if is_numerical:
        X_full = recompute_feature_matrix(
            dict_part_vals=dict_part_vals, dict_part_lens=dict_part_lens,
            list_scales=list_scales, features=candidates, split_kws=split_kws,
        )
    else:
        builder = feature_matrix_builder or get_feature_matrix_fast_
        X_full = builder(features=candidates, df_parts=df_parts, df_scales=df_scales,
                         accept_gaps=accept_gaps, n_jobs=n_jobs, aa_lookup_cache=aa_lookup_cache)
    df = add_stat(df_feat=df, X_cached=X_full, labels=labels, parametric=parametric,
                  label_test=label_test, label_ref=label_ref, n_jobs=n_jobs, vectorized=vectorized)
    # Full-data pre-filter threshold: the full data is authoritative, so a candidate whose
    # full-data std_test exceeds max_std_test (or whose abs_mean_dif is NaN under accept_gaps) is
    # dropped even if it was stable across subsamples. .copy() so the position assignment below
    # writes to an owned frame (no SettingWithCopyWarning on the boolean-filtered slice).
    df = df[df[ut.COL_STD_TEST] <= max_std_test].copy()
    if accept_gaps:
        df = df[~df[ut.COL_ABS_MEAN_DIF].isna()].copy()
    features_kept = df[ut.COL_FEATURE].to_list()
    df[ut.COL_POSITION] = get_positions_(features=features_kept, start=start, **args_len)
    df = add_scale_info_(df_feat=df, df_cat=df_cat)

    # The full-data max_std_test filter can drop every candidate (reachable under
    # resample='both'/'test', where a candidate's in-round std passes while its full-data std does
    # not). The greedy redundancy filter assumes >= 1 row (it pops the first feature), so skip it
    # on an empty frame and return the schema-correct empty df_feat.
    if df.empty:
        df_feat = df
    else:
        if verbose:
            ut.print_out("3. CPP full-data redundancy filtering on the stable candidates")
        df_feat = filtering(df=df, df_scales=df_scales, n_filter=n_filter, check_cat=check_cat,
                            max_overlap=max_overlap, max_cor=max_cor, redundancy=redundancy)
    df_feat = df_feat.reset_index(drop=True)
    df_feat[ut.COLS_FEAT_STAT] = df_feat[ut.COLS_FEAT_STAT].round(3)
    df_feat[ut.COL_FEATURE] = df_feat[ut.COL_FEATURE].astype(str)
    df_feat[ut.COL_SELECTION_FREQUENCY] = [round(freq[f], 3) for f in df_feat[ut.COL_FEATURE]]
    df_feat = _attach_filter_stats(
        df_feat=df_feat, n_candidates=n_feat_total, n_after_prefilter=len(candidates),
        n_after_redundancy=len(df_feat), n_requested=None, verbose=verbose,
    )
    if verbose:
        ut.print_out(f"4. CPP returns df of {len(df_feat)} stable features with statistics "
                     f"and 'selection_frequency'")
    return df_feat
