"""
This is a script for the backend of CPP's numerical-mode survivor-recompute stage.

After PR2.5's streaming pre-filter narrows the candidate set to ~``n_pre_filter``
features, ``recompute_feature_matrix`` builds the
``(n_samples, n_pre_filter)`` value matrix needed by ``add_stat``.

Implementation (PR2.5+ vectorized): per ``(split_type, part)`` bucket, the
per-sample list of split-position arrays is computed **once** by applying the
SplitRange function to a sentinel ``np.arange(L_i)``. The resulting positions
are stored in a single ``(n_samples, n_splits, max_split_len)`` ``int64``
buffer (with ``L_max`` as a NaN-padding sentinel index). For each scale-chunk,
one fancy-indexed gather + ``np.nanmean`` produces ``(n_samples, n_splits, K)``
means for K scales at a time — no per-sample Python loop in the inner
hot path.

Bit-identical to legacy via the same ``np.round(..., 5)`` applied in
``utils_feature._feature_value``.
"""
import gc
from collections import defaultdict

import numpy as np

from .._split import SplitRange
from ..utils_feature import _get_split_info


# I Helper Functions
def _parse_feature(feature):
    """Split 'PART-SPLIT-SCALE' into its components (PART preserves case for label-back)."""
    part_upper, split_str, scale = feature.split("-")
    return part_upper, split_str, scale


def build_position_buffer(split_type=None, split_type_args=None, seq_lens=None,
                          L_max=None, spr=None):
    """Build the per-sample split-position buffer for one ``(split_type, part)``.

    For each sample, applies ``f(arange(L_i), **split_type_args)`` to extract
    per-split index arrays (sentinel-sequence trick — the SplitRange functions
    are residue-content-agnostic so applying them to ``arange(L_i)`` returns
    the residue indices directly).

    Parameters
    ----------
    split_type : str
        ``"Segment"`` / ``"Pattern"`` / ``"PeriodicPattern"``.
    split_type_args : dict
        Forwarded to the SplitRange function.
    seq_lens : np.ndarray, shape (n_samples,)
        Per-sample real (non-padded) sequence length for this part.
    L_max : int
        Length-axis size of the per-part value tensor (so the buffer can use
        ``L_max`` as a NaN-padding sentinel index).
    spr : SplitRange
        SplitRange instance (``type_str=False``).

    Returns
    -------
    pos_buf : np.ndarray, shape (n_samples, n_splits, max_split_len) int64
        Per-(sample, split, position_slot) residue index into the per-part
        tensor. Slots beyond a sample's actual split length use ``L_max``.
    labels_splits : list[str]
        Split labels in the same order as ``pos_buf``'s split axis. Mapping
        from feature's split-string to its column in ``pos_buf``.
    max_split_len : int
        Max per-(sample, split) residue count — the buffer's last axis size.
    """
    f = getattr(spr, split_type.lower())
    labels_splits = getattr(spr, "labels_" + split_type.lower())(**split_type_args)
    n_splits = len(labels_splits)
    n_samples = len(seq_lens)

    list_positions = []
    max_split_len = 0
    for i in range(n_samples):
        L_i = int(seq_lens[i])
        if L_i == 0:
            list_positions.append([np.array([], dtype=np.int64)] * n_splits)
            continue
        sentinel = np.arange(L_i, dtype=np.int64)
        positions_i = f(seq=sentinel, **split_type_args)
        normalized = [np.asarray(p, dtype=np.int64) for p in positions_i]
        list_positions.append(normalized)
        for p in normalized:
            if len(p) > max_split_len:
                max_split_len = len(p)

    pos_buf = np.full((n_samples, n_splits, max(max_split_len, 1)), L_max, dtype=np.int64)
    for i, positions_i in enumerate(list_positions):
        for s_idx, p in enumerate(positions_i):
            if len(p):
                pos_buf[i, s_idx, : len(p)] = p

    return pos_buf, labels_splits, max(max_split_len, 1)


def iter_scale_chunks(arr_3d=None, scale_indices=None, pos_buf=None,
                       max_mem_mb=64):
    """Yield ``(chunk_start, chunk_means)`` for chunks of the scale axis.

    Reuses a single set of pre-allocated buffers (``arr_chunk_padded``,
    ``gathered``, ``chunk_means``) across iterations to avoid the macOS
    allocator-fragmentation pattern that causes RSS to grow with chunk count
    even when individual chunks are tiny.

    Yields
    ------
    (chunk_start, chunk_means) : (int, np.ndarray)
        ``chunk_means.shape == (n_samples, n_splits, chunk_size)`` — note
        that ``chunk_size`` may shrink on the LAST chunk; callers must use
        ``chunk_means.shape[2]`` rather than caching the value. Yielded
        array IS the internal buffer; caller must consume before the next
        iteration (do not retain).
    """
    n_samples, L_max, _ = arr_3d.shape
    max_split_len = pos_buf.shape[2]
    K = len(scale_indices)
    if K == 0:
        return

    per_scale_bytes = n_samples * pos_buf.shape[1] * max_split_len * 8
    chunk_size = max(1, int(max_mem_mb * 1024 * 1024 / per_scale_bytes))
    chunk_size = min(chunk_size, K)
    i_row = np.arange(n_samples)[:, None, None]

    # Pre-allocate buffers used inside the loop. Reused across chunks → one
    # set of large allocations instead of K/chunk_size sets.
    arr_chunk_padded_buf = np.empty((n_samples, L_max + 1, chunk_size), dtype=np.float64)
    gathered_buf = np.empty(
        (n_samples, pos_buf.shape[1], max_split_len, chunk_size), dtype=np.float64
    )
    chunk_means_buf = np.empty(
        (n_samples, pos_buf.shape[1], chunk_size), dtype=np.float64
    )

    for chunk_start in range(0, K, chunk_size):
        chunk_idx = scale_indices[chunk_start : chunk_start + chunk_size]
        Kc = len(chunk_idx)
        # Take contiguous slice (zero-copy view) when possible; else fancy-index copy.
        first = chunk_idx[0]
        if Kc > 1 and chunk_idx[-1] == first + Kc - 1 and \
                all(chunk_idx[k] == first + k for k in range(Kc)):
            arr_chunk = arr_3d[:, :, first : first + Kc]
        else:
            arr_chunk = arr_3d[:, :, chunk_idx]

        # Fill arr_chunk_padded_buf[..., :Kc] in-place via np.take (supports out=).
        # take on axis=2 of arr_3d would also work but we already have arr_chunk.
        # Copy into the pre-allocated padded buffer:
        arr_chunk_padded_buf[:, :L_max, :Kc] = arr_chunk
        arr_chunk_padded_buf[:, L_max, :Kc] = np.nan

        # Fancy index. numpy doesn't support out= for fancy indexing, so this
        # is the only step that still allocates per chunk. The result is then
        # immediately copied into our reusable gathered_buf, after which the
        # transient fancy-index array is released.
        gathered_buf[..., :Kc] = arr_chunk_padded_buf[i_row, pos_buf][..., :Kc]

        # In-place nanmean via mean(np.where(isnan, 0, x)) / count is awkward.
        # Use np.nanmean output to chunk_means_buf via copy.
        np.nanmean(gathered_buf[..., :Kc], axis=2, out=chunk_means_buf[..., :Kc])

        yield chunk_start, chunk_means_buf[..., :Kc]


def gather_means_chunked(arr_3d=None, scale_indices=None, pos_buf=None,
                         max_mem_mb=256):
    """Materialize the full ``(n_samples, n_splits, K)`` means matrix.

    Convenience wrapper around ``iter_scale_chunks`` that allocates the full
    output at once. **Avoid this for large K** — use ``iter_scale_chunks``
    directly so the per-chunk intermediate is the only resident allocation.
    """
    n_samples = arr_3d.shape[0]
    n_splits = pos_buf.shape[1]
    K = len(scale_indices)
    means = np.empty((n_samples, n_splits, K), dtype=np.float64)
    for chunk_start, chunk_means in iter_scale_chunks(
        arr_3d=arr_3d, scale_indices=scale_indices, pos_buf=pos_buf,
        max_mem_mb=max_mem_mb,
    ):
        means[:, :, chunk_start : chunk_start + chunk_means.shape[2]] = chunk_means
    return means


# II Main Functions
def recompute_feature_matrix(
    dict_part_vals=None,
    dict_part_lens=None,
    list_scales=None,
    features=None,
    split_kws=None,
    max_chunk_mb=1024,
):
    """Build (n_samples, n_features) value matrix from the (n, L, D) tensor.

    Parameters
    ----------
    dict_part_vals : Dict[str, np.ndarray]
        Per-part ``(n_samples, L_part_max, D)`` float64 tensor.
    dict_part_lens : Dict[str, np.ndarray]
        Per-part ``(n_samples,)`` int64 real sequence length.
    list_scales : List[str]
        Ordered list of scale names matching the D axis.
    features : Sequence[str]
        Feature IDs (``PART-SPLIT-SCALE``) to compute. Output order matches.
    split_kws : dict
        Resolves per-split-type kwargs.
    max_chunk_mb : int, default=1024
        Soft cap on the per-chunk gather-buffer allocation; reduces peak RSS.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features) float64
        Per-sample per-feature values, rounded to 5 decimals (bit-identical
        with legacy ``get_feature_matrix_``).
    """
    n_samples = list(dict_part_vals.values())[0].shape[0]
    n_features = len(features)
    X = np.full((n_samples, n_features), np.nan, dtype=np.float64)
    if n_features == 0:
        return X

    scale_to_idx = {s: i for i, s in enumerate(list_scales)}
    spr = SplitRange(split_type_str=False)

    # Group by (split_type, part); within: list of (col_idx, split_label_normalized, scale).
    grouped = defaultdict(list)
    for col_idx, feature in enumerate(features):
        part_upper, split_str, scale = _parse_feature(feature)
        split_type, _ = _get_split_info(split=split_str)
        part = part_upper.lower()
        grouped[(split_type, part)].append((col_idx, split_str.replace(" ", ""), scale))

    for (split_type, part), feats in grouped.items():
        split_type_args = split_kws[split_type]
        arr_3d = dict_part_vals[part]
        seq_lens = dict_part_lens[part]
        L_max = arr_3d.shape[1]

        pos_buf, labels_splits, _ = build_position_buffer(
            split_type=split_type, split_type_args=split_type_args,
            seq_lens=seq_lens, L_max=L_max, spr=spr,
        )
        label_to_idx = {lbl.replace(" ", ""): i for i, lbl in enumerate(labels_splits)}

        # Group features by scale; we need each unique scale only once.
        feats_by_scale = defaultdict(list)
        for col_idx, split_label, scale in feats:
            feats_by_scale[scale].append((col_idx, split_label))

        scales_in_bucket = list(feats_by_scale.keys())
        scale_indices = [scale_to_idx[s] for s in scales_in_bucket]

        # Stream per-chunk to keep memory peak bounded by chunk_size, not by
        # the full per-bucket (n, n_splits, K) means matrix.
        for chunk_start, chunk_means in iter_scale_chunks(
            arr_3d=arr_3d, scale_indices=scale_indices, pos_buf=pos_buf,
            max_mem_mb=max_chunk_mb,
        ):
            chunk_means = np.round(chunk_means, 5)
            Kc = chunk_means.shape[2]
            for k_local in range(Kc):
                scale = scales_in_bucket[chunk_start + k_local]
                for col_idx, split_label in feats_by_scale[scale]:
                    split_idx = label_to_idx[split_label]
                    X[:, col_idx] = chunk_means[:, split_idx, k_local]

    return X
