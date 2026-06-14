"""
This is a script for the backend of CPP's bit-exact fast feature-matrix builder.

``get_feature_matrix_fast_`` is a BYTE-IDENTICAL equivalent of legacy
``utils_feature.get_feature_matrix_`` with the per-(sample, feature)
``np.mean`` summation order preserved (so Mann-Whitney p-values land exactly
on the same ranks). The speedup comes from:

1. Pre-building a per-part ``(n_samples, L_max)`` AA-index matrix ONCE per
   part, then fancy-indexing into a per-call float64 ``scale_matrix`` to get
   numerical per-residue values without per-feature ``dict_scale`` lookups.
2. Hoisting the ``Split`` instance and the ``is_dtype_str`` check out of the
   per-feature loop.
3. Replacing ``np.vectorize(lambda x: np.mean([dict_scale[a] for a in x]))``
   (which is itself a Python loop over part_split rows) with an equivalent
   direct loop on numerical slices that yields the same ``np.mean`` value
   in the same summation order.
4. (Phase A.1) Hoisting the gap-symbol check out of the per-feature loop:
   once at the top of ``get_feature_matrix_fast_`` instead of n_features
   times.
5. (Phase A.3) Caching the parsed ``(split_type, split_kwargs, f_split_num,
   f_split_str)`` per unique ``split_str`` so features sharing a split don't
   re-parse + getattr.
6. (Phase A.4) Optionally accepting a precomputed ``aa_lookup_cache`` from
   the caller (CPP instance) so repeat ``run_num`` calls don't rebuild the
   per-part AA-index matrices or the float64 ``scale_matrix``.

The float64 ``scale_matrix`` stores ``dict_scale`` values at full precision,
so per-residue values gathered from ``scale_matrix[aa_idx, scale_idx]`` are
bit-identical to legacy's ``dict_scale[c]`` lookups. Output is rounded to 5
decimals to match legacy ``_feature_value``.
"""
import os
import hashlib
from functools import lru_cache
import numpy as np
from joblib import Parallel, delayed

import aaanalysis.utils as ut
from .._split import Split, SplitVec
from ..utils_feature import _get_split_info, post_check_vf_scale


# I Helper Functions
class _ScalesKey:
    """Content-hash wrapper making ``df_scales`` usable as an ``lru_cache`` key.

    Two distinct ``df_scales`` objects with identical columns / index / values
    hash equal, so sweeps that rebuild the same scale set across thousands of
    configs reuse one cached scale-matrix instead of recomputing it per
    construction. Holds a reference to ``df_scales`` so the cached entry can
    rebuild on a miss.
    """

    __slots__ = ("df_scales", "_hash")

    def __init__(self, df_scales):
        self.df_scales = df_scales
        cols = tuple(map(str, df_scales.columns))
        idx = tuple(map(str, df_scales.index))
        vals = np.ascontiguousarray(df_scales.to_numpy(dtype=np.float64)).tobytes()
        self._hash = hash((cols, idx, hashlib.blake2b(vals).digest()))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _ScalesKey) and self._hash == other._hash


@lru_cache(maxsize=32)
def _build_scale_lookup_cached(key):
    """Build ``(dict_all_scales, scale_to_idx, scale_matrix_f64, n_aa)`` from a
    ``_ScalesKey``; memoized by ``df_scales`` content (see :class:`_ScalesKey`).

    The scale-matrix derives purely from ``df_scales`` (independent of
    ``df_parts``), so it is the one piece worth caching across configs. The cache
    is self-bounding (``maxsize=32``); clear it eagerly via
    :func:`clear_scale_lookup_cache` (internal utility).
    """
    df_scales = key.df_scales
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col]))
                       for col in list(df_scales)}
    scale_to_idx = {s: i for i, s in enumerate(list(df_scales))}
    n_aa = len(df_scales.index)
    scale_matrix_f64 = _build_scale_matrix_f64(dict_all_scales=dict_all_scales, n_aa=n_aa)
    return dict_all_scales, scale_to_idx, scale_matrix_f64, n_aa


def build_scale_lookup(df_scales=None):
    """Public-to-backend entry: content-cached scale lookup for ``df_scales``."""
    return _build_scale_lookup_cached(_ScalesKey(df_scales))


def clear_scale_lookup_cache():
    """Evict the module-level scale-lookup LRU.

    Internal utility (not public API). The cache self-bounds at
    ``maxsize=32``; call this only to release the held scale matrices eagerly in
    a long-running process that cycles through many distinct scale sets.
    """
    _build_scale_lookup_cached.cache_clear()
def _build_aa_idx_per_part(df_parts=None, dict_all_scales=None):
    """Per-part (n_samples, L_max) int32 AA-index matrix; ``n_aa`` is the NaN sentinel."""
    list_aa = list(next(iter(dict_all_scales.values())).keys())
    aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
    n_aa = len(list_aa)
    aa_idx_per_part = {}
    seq_lens_per_part = {}
    for part in list(df_parts):
        seqs = df_parts[part].tolist()
        seq_lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        L_max = int(seq_lens.max()) if len(seq_lens) else 0
        X_chars = np.full((len(seqs), L_max), "_", dtype="<U1")
        for i, s in enumerate(seqs):
            X_chars[i, : len(s)] = list(s)
        aa_idx = np.full(X_chars.shape, n_aa, dtype=np.int32)
        for aa, idx in aa_to_idx.items():
            aa_idx[X_chars == aa] = idx
        aa_idx_per_part[part] = aa_idx
        seq_lens_per_part[part] = seq_lens
    return aa_idx_per_part, seq_lens_per_part, aa_to_idx, n_aa


def _build_scale_matrix_f64(dict_all_scales=None, n_aa=None):
    """``(n_aa + 1, n_scales)`` float64 lookup. Row ``n_aa`` is NaN (padding / unknown).

    Float64 preserves the full precision of ``dict_scale`` floats so the
    gathered per-residue values are bit-identical to legacy's
    ``dict_scale[c]`` lookups.
    """
    list_scales = list(dict_all_scales)
    list_aa = list(next(iter(dict_all_scales.values())).keys())
    aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
    scale_matrix = np.full((n_aa + 1, len(list_scales)), np.nan, dtype=np.float64)
    for s, scale in enumerate(list_scales):
        for aa, idx in aa_to_idx.items():
            scale_matrix[idx, s] = dict_all_scales[scale][aa]
    return scale_matrix


def _hoisted_gap_check(df_parts=None):
    """One-time gap-symbol scan across all parts.

    Equivalent to running ``pre_check_vf_scale`` per feature but ~100×
    cheaper because the check is independent of the (part, split) being
    computed. Raises ``ValueError`` with the legacy error message if any
    part contains the gap symbol ``ut.STR_AA_GAP``.
    """
    for part in list(df_parts):
        if df_parts[part].str.contains(ut.STR_AA_GAP, regex=False).any():
            raise ValueError("Some input sequences contain gaps ('-').")


def _positions_are_uniform(split_type=None, split_kwargs=None, seq_lens=None,
                           all_same_L=None):
    """Whether the per-sample residue positions for this split are identical across samples.

    * ``Pattern`` with ``terminus="N"``: positions are ``list_pos - 1`` — does NOT
      depend on the per-sample sequence length. Always uniform.
    * All other split types (Segment, Pattern-C, PeriodicPattern) compute
      positions from the per-sample length; uniform iff every sample's
      sequence length for this part is identical.
    """
    if split_type == "Pattern" and split_kwargs.get("terminus") == "N":
        return True
    return bool(all_same_L)


def _positions_are_contiguous(positions):
    """True iff ``positions`` is a contiguous ascending run ``[k, k+1, ..., k+n-1]``.

    Why this matters: when ``positions`` is contiguous, ``arr_2d[:, start:end]``
    gives a view, and ``view.mean(axis=1)`` is bit-identical to applying
    ``np.mean`` to each row individually. When ``positions`` is non-contiguous,
    fancy-indexing produces a copy with a different memory layout, and
    ``np.mean(axis=1)`` may use a different SIMD reduction path that drifts at
    ULP — enough to cross the ``np.round(_, 5)`` boundary on some inputs.
    """
    if len(positions) == 0:
        return False
    p_start = int(positions[0])
    p_end = int(positions[-1]) + 1
    return (p_end - p_start) == len(positions) and bool(
        np.array_equal(np.asarray(positions, dtype=np.int64),
                       np.arange(p_start, p_end, dtype=np.int64))
    )


def _compute_uniform_feature_column(arr_2d=None, positions=None, accept_gaps=False):
    """Vectorized per-sample mean when positions are uniform across samples.

    Uses a contiguous SLICE (view, not copy) so the reduction is bit-identical
    to the per-sample ``np.mean`` fallback. Caller must verify positions are
    contiguous (see ``_positions_are_contiguous``).
    """
    p_start = int(positions[0])
    p_end = int(positions[-1]) + 1
    gathered = arr_2d[:, p_start:p_end]  # view, not fancy-index copy
    if accept_gaps:
        col = np.nanmean(gathered, axis=1)
    else:
        col = gathered.mean(axis=1)
    return np.round(col, 5)


def _compute_features_into(X_out=None, features=None,
                           aa_idx_per_part=None, seq_lens_per_part=None,
                           scale_matrix_f64=None, scale_to_idx=None,
                           sp_vec=None, accept_gaps=False):
    """Per-feature ``np.mean`` over residue values.

    Bit-identical with legacy ``_feature_value``: same residue order, same
    ``np.mean`` summation, same ``np.round(_, 5)``.

    Per-feature work is reduced by:
      * Phase A.1 — gap check hoisted out of this loop.
      * Phase A.3 — parsed split info cached per unique ``split_str``.
      * Phase B   — when per-sample positions are uniform across samples
        (Pattern N-terminus always; Segment/Pattern-C/PeriodicPattern only
        when ``seq_lens`` is constant across samples for that part), replace
        the per-sample Python loop with a single vectorized
        ``arr_2d[:, positions].mean(axis=1)``. ``np.mean(axis=1)`` on a
        ``(n, k)`` matrix is element-wise identical to applying
        ``np.mean`` to each row individually, so bit-exact parity holds.
    """
    n_samples = aa_idx_per_part[next(iter(aa_idx_per_part))].shape[0]

    # Phase A.3: cache parsed split info per unique split_str.
    split_info_cache = {}  # split_str → (split_type, split_kwargs, f_split_num)

    # Phase B: per (part) precompute whether every sample shares the same length.
    all_same_L_per_part = {
        part: bool(np.all(seq_lens_per_part[part] == seq_lens_per_part[part][0]))
                if len(seq_lens_per_part[part]) > 0 else True
        for part in seq_lens_per_part
    }
    # Cache reference-sample positions per (part, split_str) once positions
    # are confirmed uniform; reused across all scales sharing this (part, split).
    uniform_positions_cache = {}  # (part, split_str) → np.ndarray of positions

    for feat_idx, feat_name in enumerate(features):
        part_upper, split_str, scale = ut.split_feat_id(feat_id=feat_name)
        part = part_upper.lower()

        cached = split_info_cache.get(split_str)
        if cached is None:
            split_type, split_kwargs = _get_split_info(split=split_str)
            f_split_num = getattr(sp_vec, split_type.lower())
            cached = (split_type, split_kwargs, f_split_num)
            split_info_cache[split_str] = cached
        split_type, split_kwargs, f_split_num = cached

        scale_idx = scale_to_idx[scale]
        aa_idx = aa_idx_per_part[part]  # (n, L_max) int32
        seq_lens = seq_lens_per_part[part]
        arr_2d = scale_matrix_f64[aa_idx, scale_idx]  # (n, L_max) float64

        # Phase B: take the vectorized path when positions are uniform AND contiguous.
        # Non-contiguous positions require fancy indexing, whose mean(axis=1)
        # can drift by 1 ULP from the per-sample np.mean (different SIMD
        # reduction path); contiguous positions allow a zero-copy slice
        # that gives bit-identical means.
        all_same_L = all_same_L_per_part[part]
        is_uniform = _positions_are_uniform(
            split_type=split_type, split_kwargs=split_kwargs,
            seq_lens=seq_lens, all_same_L=all_same_L,
        )
        if is_uniform:
            cache_key = (part, split_str)
            cached_positions = uniform_positions_cache.get(cache_key)
            if cached_positions is None:
                L0 = int(seq_lens[0]) if len(seq_lens) else 0
                sentinel = np.arange(L0, dtype=np.int64)
                ref_segs = f_split_num(seq=sentinel, **split_kwargs)
                positions = np.asarray(ref_segs, dtype=np.int64)
                contiguous = _positions_are_contiguous(positions)
                cached_positions = (positions, contiguous)
                uniform_positions_cache[cache_key] = cached_positions
            positions, contiguous = cached_positions
            if contiguous:
                col = _compute_uniform_feature_column(
                    arr_2d=arr_2d, positions=positions, accept_gaps=accept_gaps,
                )
                if accept_gaps:
                    post_check_vf_scale(feature_values=col)
                X_out[:, feat_idx] = col
                continue
            # Non-contiguous: fall through to per-sample loop.

        # Phase C: inline per-sample loop, dispatching by split_type to skip the
        # ``f_split_num`` Python call. Each branch reproduces the exact slice /
        # index computation that ``SplitVec.<split_type>`` does, then calls
        # ``np.mean`` on the 1-D result — bit-identical because the values + order
        # match the SplitVec implementation.
        seq_lens_int = seq_lens.tolist()  # int conversion once per feature
        col = np.empty(n_samples, dtype=np.float64)
        _np_mean = np.mean
        _np_nanmean = np.nanmean

        if split_type == "Segment":
            i_th = split_kwargs["i_th"]
            n_split = split_kwargs["n_split"]
            for i in range(n_samples):
                L_i = seq_lens_int[i]
                len_segment = L_i / n_split
                start = int(len_segment * (i_th - 1))
                end = int(len_segment * i_th)
                seg = arr_2d[i, start:end]
                if accept_gaps:
                    col[i] = _np_nanmean(seg)
                else:
                    col[i] = _np_mean(seg)
        elif split_type == "Pattern":
            terminus = split_kwargs["terminus"]
            list_pos = split_kwargs["list_pos"]
            if terminus == "N":
                # positions = list_pos - 1 (fixed; independent of L_i)
                pos_arr = np.asarray(list_pos, dtype=np.int64) - 1
                for i in range(n_samples):
                    seg = arr_2d[i, pos_arr]
                    if accept_gaps:
                        col[i] = _np_nanmean(seg)
                    else:
                        col[i] = _np_mean(seg)
            else:  # terminus == "C"
                list_pos_arr = np.asarray(list_pos, dtype=np.int64)
                for i in range(n_samples):
                    L_i = seq_lens_int[i]
                    pos_arr = L_i - list_pos_arr
                    seg = arr_2d[i, pos_arr]
                    if accept_gaps:
                        col[i] = _np_nanmean(seg)
                    else:
                        col[i] = _np_mean(seg)
        else:
            # PeriodicPattern (and any other split_type added later) — fall
            # back to the canonical ``f_split_num`` path. PeriodicPattern's
            # position list is computed by ``SplitVec.get_list_periodic_pattern_pos``
            # which has its own caching; not worth inlining here.
            for i in range(n_samples):
                L_i = seq_lens_int[i]
                seq_vals = arr_2d[i, :L_i]
                segs = f_split_num(seq=seq_vals, **split_kwargs)
                if accept_gaps:
                    vals = np.asarray(segs, dtype=np.float64)
                    col[i] = _np_nanmean(vals)
                else:
                    col[i] = _np_mean(segs)
        col = np.round(col, 5)
        if accept_gaps:
            post_check_vf_scale(feature_values=col)
        X_out[:, feat_idx] = col


# II Main Functions
class AALookupCache:
    """Phase A.4: cache the per-part AA index + float64 scale_matrix for reuse.

    Built once per (df_parts, df_scales) pair — typically once per CPP
    instance. Lets repeat ``cpp.run_num()`` calls skip the per-call
    ``_build_aa_idx_per_part`` (3 parts × n_samples char-matrix construction)
    and ``_build_scale_matrix_f64`` work.

    Construct via ``AALookupCache.from_df(df_parts, df_scales)``; passing the
    resulting instance to ``get_feature_matrix_fast_(..., aa_lookup_cache=...)``
    short-circuits the per-call rebuild. The cache is read-only — changing
    ``df_parts`` or ``df_scales`` requires building a fresh one (CPP keeps the
    cache on the instance and discards it if needed).
    """

    __slots__ = ("aa_idx_per_part", "seq_lens_per_part", "scale_matrix_f64",
                 "scale_to_idx", "df_parts_id", "df_scales_id")

    def __init__(self, aa_idx_per_part, seq_lens_per_part, scale_matrix_f64,
                 scale_to_idx, df_parts_id, df_scales_id):
        self.aa_idx_per_part = aa_idx_per_part
        self.seq_lens_per_part = seq_lens_per_part
        self.scale_matrix_f64 = scale_matrix_f64
        self.scale_to_idx = scale_to_idx
        self.df_parts_id = df_parts_id
        self.df_scales_id = df_scales_id

    @classmethod
    def from_df(cls, df_parts, df_scales):
        # Scale-matrix is content-cached across configs (see build_scale_lookup);
        # the per-part AA-index matrix depends on df_parts so stays per-call.
        dict_all_scales, scale_to_idx, scale_matrix_f64, _ = build_scale_lookup(
            df_scales=df_scales,
        )
        aa_idx_per_part, seq_lens_per_part, _, _ = _build_aa_idx_per_part(
            df_parts=df_parts, dict_all_scales=dict_all_scales,
        )
        return cls(aa_idx_per_part, seq_lens_per_part, scale_matrix_f64,
                   scale_to_idx, id(df_parts), id(df_scales))

    def matches(self, df_parts, df_scales):
        return id(df_parts) == self.df_parts_id and id(df_scales) == self.df_scales_id


def get_feature_matrix_fast_(features=None, df_parts=None, df_scales=None,
                             accept_gaps=False, n_jobs=None,
                             aa_lookup_cache=None):
    """Bit-exact replacement for ``utils_feature.get_feature_matrix_``.

    Parameters
    ----------
    aa_lookup_cache : AALookupCache, optional
        Precomputed AA-index + scale-matrix cache (Phase A.4). When provided,
        skips the per-call ``_build_aa_idx_per_part`` and
        ``_build_scale_matrix_f64`` calls. The cache MUST match the supplied
        ``df_parts`` / ``df_scales`` (caller's responsibility — typically
        managed by ``CPP._aa_cache``).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features) float64
        Per-sample feature value matrix. ``X[:, i] == legacy_get_feature_matrix_[:, i]``
        for every i — byte-identical.

    The independent parity test in
    ``tests/unit/cpp_tests/test_get_feature_matrix_fast_parity.py`` exercises
    this end-to-end against legacy. Use through ``cpp_run`` for the
    full ``CPP.run_num`` pipeline.
    """
    features = [features] if isinstance(features, str) else list(features)

    # Phase A.1: hoist the gap-symbol check OUT of the per-feature loop.
    if not accept_gaps:
        _hoisted_gap_check(df_parts=df_parts)

    # Phase A.4: reuse the lookup cache when available.
    if aa_lookup_cache is not None and aa_lookup_cache.matches(df_parts, df_scales):
        aa_idx_per_part = aa_lookup_cache.aa_idx_per_part
        seq_lens_per_part = aa_lookup_cache.seq_lens_per_part
        scale_matrix_f64 = aa_lookup_cache.scale_matrix_f64
        scale_to_idx = aa_lookup_cache.scale_to_idx
    else:
        dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col]))
                           for col in list(df_scales)}
        scale_to_idx = {s: i for i, s in enumerate(list(df_scales))}
        aa_idx_per_part, seq_lens_per_part, _, n_aa = _build_aa_idx_per_part(
            df_parts=df_parts, dict_all_scales=dict_all_scales,
        )
        scale_matrix_f64 = _build_scale_matrix_f64(
            dict_all_scales=dict_all_scales, n_aa=n_aa,
        )

    sp_vec = SplitRange_sp_vec()  # one shared instance, hoisted out of loop

    n_samples = len(df_parts)
    n_jobs = ut.resolve_n_jobs(n_jobs=n_jobs, n_work=len(features))

    if n_jobs == 1 or len(features) <= 1:
        X = np.empty((n_samples, len(features)), dtype=np.float64)
        _compute_features_into(
            X_out=X, features=features,
            aa_idx_per_part=aa_idx_per_part, seq_lens_per_part=seq_lens_per_part,
            scale_matrix_f64=scale_matrix_f64, scale_to_idx=scale_to_idx,
            sp_vec=sp_vec, accept_gaps=accept_gaps,
        )
        return X

    feature_chunks = np.array_split(features, n_jobs)

    def _mp(chunk):
        X_chunk = np.empty((n_samples, len(chunk)), dtype=np.float64)
        _compute_features_into(
            X_out=X_chunk, features=list(chunk),
            aa_idx_per_part=aa_idx_per_part, seq_lens_per_part=seq_lens_per_part,
            scale_matrix_f64=scale_matrix_f64, scale_to_idx=scale_to_idx,
            sp_vec=sp_vec, accept_gaps=accept_gaps,
        )
        return X_chunk

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(_mp)(chunk) for chunk in feature_chunks)
    return np.concatenate(results, axis=1)


def SplitRange_sp_vec():
    """Return a ``SplitVec`` instance — module-level helper to avoid the per-feature alloc."""
    return SplitVec(type_str=False)
