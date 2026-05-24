"""
This is a script for the backend of CPP's numerical-mode residue-value assignment
stage: ``assign_scale_values_to_seq_num`` looks up scale values for every residue
in ``df_parts`` and returns one 3D tensor per part (instead of the per-(scale,
part) 2D dict produced by ``_filters._assign``).

Output contract (PR1, seq-mode only):
- ``dict_part_vals[part]`` = ``(n_samples, L_part_max, D)`` float32 ndarray.
  Positions beyond a sample's sequence length are ``np.nan``; non-canonical
  residues (or '_' padding chars) also resolve to ``np.nan`` via the
  ``aa_idx == n_aa`` sentinel row of ``scale_matrix``.
- ``dict_part_lens[part]`` = ``(n_samples,)`` int64 ndarray giving each
  sample's real sequence length for that part (lets downstream stages slice
  off the NaN tail without scanning).

# DEV: PR3 adds a sibling ``assign_dict_num_to_parts`` that produces the same
# shape from a user-supplied ``dict_num: Dict[entry, (L, D)]`` tensor instead
# of the AA-letter scale lookup. Both functions emit the same output contract
# so the downstream stages (``_stat_filter``, ``_pre_filter``, ``_add_stat``,
# ``_redundancy_filter``) are dict_num-agnostic.
"""
import os
import numpy as np
from joblib import Parallel, delayed

import aaanalysis.utils as ut
from ._progress import (
    _resolve_shared,
    _reset_progress,
    _cleanup_mp_manager,
)


# I Helper Functions
def _build_scale_matrix(dict_all_scales=None):
    """Build (n_aa + 1, n_scales) float32 lookup. Row ``n_aa`` is NaN (padding / unknown).

    Matches legacy ``_filters/_assign.py``'s float32 dtype so the pre-filter
    stats in seq-mode are bit-identical to ``CPP.run``'s pre-filter stats
    (both compute means over float32-truncated values promoted to float64).
    The downstream pass 2 in ``cpp_run_num`` delegates to legacy
    ``get_feature_matrix_`` for full-precision add_stat values, so the
    scale_matrix dtype here only affects pre_filter parity.
    """
    list_scales = list(dict_all_scales)
    list_aa = list(dict_all_scales[list_scales[0]].keys())
    aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
    n_aa = len(list_aa)
    n_scales = len(list_scales)
    scale_matrix = np.full((n_aa + 1, n_scales), np.nan, dtype=np.float32)
    for s, scale in enumerate(list_scales):
        for aa, idx in aa_to_idx.items():
            scale_matrix[idx, s] = dict_all_scales[scale][aa]
    return scale_matrix, aa_to_idx, n_aa


def _build_aa_idx_matrix(seqs=None, aa_to_idx=None, n_aa=None, L_max=None):
    """Return (n, L_max) int32 AA-index matrix with sentinel ``n_aa`` for padding/unknown."""
    n_samples = len(seqs)
    X_chars = np.full((n_samples, L_max), "_", dtype="<U1")
    for i, s in enumerate(seqs):
        X_chars[i, : len(s)] = list(s)
    aa_idx_matrix = np.full(X_chars.shape, n_aa, dtype=np.int32)
    for aa, idx in aa_to_idx.items():
        aa_idx_matrix[X_chars == aa] = idx
    return aa_idx_matrix


def _assign_scale_values_to_seq_num(
    df_parts=None,
    dict_all_scales=None,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """Assign scale values for one chunk of scales, emit (dict_part_vals, dict_part_lens)."""
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)

    list_parts = list(df_parts)
    list_scales = list(dict_all_scales)
    if len(list_scales) == 0:
        return {}, {}

    scale_matrix, aa_to_idx, n_aa = _build_scale_matrix(dict_all_scales=dict_all_scales)
    dict_part_vals = {}
    dict_part_lens = {}

    progress_interval = max(1, len(list_parts) // 10) if verbose else len(list_parts)

    for j, part in enumerate(list_parts):
        seqs = df_parts[part].tolist()
        seq_lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        L_max = int(seq_lens.max()) if len(seq_lens) else 0
        aa_idx_matrix = _build_aa_idx_matrix(
            seqs=seqs, aa_to_idx=aa_to_idx, n_aa=n_aa, L_max=L_max
        )
        # Single fancy-index over (n_aa + 1, D) lookup -> (n, L_max, D) float32.
        # NaN sentinel row handles both padding and non-canonical chars.
        dict_part_vals[part] = scale_matrix[aa_idx_matrix]
        dict_part_lens[part] = seq_lens

        if verbose and (j % progress_interval == 0 or j == len(list_parts) - 1):
            ut.print_progress(i=j, n_total=len(list_parts), **args_p)

    return dict_part_vals, dict_part_lens


def _merge_part_vals(results=None, list_parts=None):
    """Concatenate per-chunk dict_part_vals along the D axis. Lens identical across chunks."""
    merged_vals = {}
    merged_lens = {}
    for part in list_parts:
        chunks = [r[0][part] for r in results]
        merged_vals[part] = np.concatenate(chunks, axis=-1)
        # Lengths only depend on df_parts, not the scale chunk
        merged_lens[part] = results[0][1][part]
    return merged_vals, merged_lens


def _assign_scale_values_to_seq_num_b(df_parts=None, dict_all_scales=None, verbose=True):
    """Option B: single (n_samples, n_parts, L_global, D) 4D tensor; dict_part_vals are views.

    Trade-off vs option A (per-part 3D tensors): one contiguous allocation but
    each part is padded to the global L_max across all parts (wastes memory on
    parts shorter than L_global). The dev bench in
    ``dev_scripts/bench_filters_num.py`` picks the winner; the loser is removed
    after one PR cycle (see ADR-0001).
    """
    list_parts = list(df_parts)
    list_scales = list(dict_all_scales)
    if len(list_scales) == 0:
        return {}, {}

    scale_matrix, aa_to_idx, n_aa = _build_scale_matrix(dict_all_scales=dict_all_scales)
    D = len(list_scales)
    n_samples = len(df_parts)
    n_parts = len(list_parts)

    seq_lens_per_part = {}
    seqs_per_part = {}
    L_max_per_part = {}
    for part in list_parts:
        seqs = df_parts[part].tolist()
        seq_lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=n_samples)
        seq_lens_per_part[part] = seq_lens
        seqs_per_part[part] = seqs
        L_max_per_part[part] = int(seq_lens.max()) if len(seq_lens) else 0
    L_global = max(L_max_per_part.values()) if L_max_per_part else 0

    # Single 4D allocation. NaN init handles padding for parts < L_global.
    X_all = np.full((n_samples, n_parts, L_global, D), np.nan, dtype=np.float64)
    dict_part_lens = {}
    for j, part in enumerate(list_parts):
        aa_idx_matrix = _build_aa_idx_matrix(
            seqs=seqs_per_part[part], aa_to_idx=aa_to_idx, n_aa=n_aa, L_max=L_global
        )
        X_all[:, j, :, :] = scale_matrix[aa_idx_matrix]
        dict_part_lens[part] = seq_lens_per_part[part]

    # Views into the 4D tensor — downstream stages treat these as (n, L_global, D).
    dict_part_vals = {part: X_all[:, j, :, :] for j, part in enumerate(list_parts)}
    return dict_part_vals, dict_part_lens


# II Main Functions
def assign_scale_values_to_seq_num(df_parts=None, df_scales=None, verbose=False, n_jobs=None,
                                   _staging_shape="A"):
    """Seq-mode residue-value assignment producing the (dict_part_vals, dict_part_lens) contract.

    Parameters
    ----------
    _staging_shape : {"A", "B"}, default="A"
        Internal flag for the dev A-vs-B bench. ``"A"`` is the production shape
        (per-part 3D tensors, L_part_max per part). ``"B"`` uses a single 4D
        tensor padded to a global L across all parts; dict values are views.
        Switched off by default; only the bench script flips this.

    Returns
    -------
    dict_part_vals : Dict[str, np.ndarray]
        ``part -> (n_samples, L_part_max, D)`` float64, NaN-padded for short
        sequences and non-canonical residues.
    dict_part_lens : Dict[str, np.ndarray]
        ``part -> (n_samples,)`` int64, real per-sample sequence length for
        that part.
    """
    list_scales = list(df_scales)
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col])) for col in list_scales}
    list_parts = list(df_parts)

    if _staging_shape == "B":
        return _assign_scale_values_to_seq_num_b(
            df_parts=df_parts, dict_all_scales=dict_all_scales, verbose=verbose
        )

    if n_jobs is None:
        n_samples, n_scales = len(df_parts), len(dict_all_scales)
        n_jobs = min(os.cpu_count() or 1, max(min(int(n_scales / 100), int(n_samples / 100)), 1))

    prefer_mp_progress = bool(n_jobs and n_jobs > 1)
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        prefer_multiprocessing=prefer_mp_progress
    )
    args_print = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    args = dict(df_parts=df_parts, verbose=verbose, prefer_multiprocessing=prefer_mp_progress, **args_print)

    _reset_progress(shared_max_progress, shared_value_lock)

    if n_jobs == 1:
        out = _assign_scale_values_to_seq_num(dict_all_scales=dict_all_scales, **args)
        if verbose:
            ut.print_end_progress(
                add_new_line=False,
                shared_max_progress=shared_max_progress,
                shared_value_lock=shared_value_lock,
            )
        if prefer_mp_progress:
            _cleanup_mp_manager()
        return out

    def _mp_scale_assignment(scales_chunk, shared_max_progress, shared_value_lock, print_lock):
        chunked_dict_scales = {scale: dict_all_scales[scale] for scale in scales_chunk}
        return _assign_scale_values_to_seq_num(
            dict_all_scales=chunked_dict_scales,
            df_parts=df_parts,
            verbose=verbose,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
            print_lock=print_lock,
            prefer_multiprocessing=False,
        )

    scale_chunks = np.array_split(list(dict_all_scales.keys()), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_mp_scale_assignment)(chunk, shared_max_progress, shared_value_lock, print_lock)
            for chunk in scale_chunks
        )

    merged = _merge_part_vals(results=results, list_parts=list_parts)
    if verbose:
        ut.print_end_progress(
            add_new_line=False,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
        )
    if prefer_mp_progress:
        _cleanup_mp_manager()
    return merged


# ---------------------------------------------------------------------------
# dict_num path: per-residue numerical tensors instead of AA-letter lookups
# ---------------------------------------------------------------------------
def _slice_dict_num_to_basic_parts(emb=None, tmd_start=None, tmd_stop=None,
                                   jmd_n_len=None, jmd_c_len=None):
    """Slice a per-protein ``(L, D)`` tensor into ``(tmd, jmd_n, jmd_c)`` arrays.

    Mirrors ``_part.Parts``'s string slicing on tensor rows (axis 0). N/C-terminal
    NaN padding when the sequence is too short for the requested JMD length.
    """
    L, D = emb.shape
    tmd_start = int(tmd_start)
    tmd_stop = int(tmd_stop)
    n_terminus_len = tmd_start - 1
    c_terminus_len = L - tmd_stop
    tmd = emb[tmd_start - 1 : tmd_stop, :]
    if n_terminus_len >= jmd_n_len:
        jmd_n = emb[tmd_start - jmd_n_len - 1 : tmd_start - 1, :]
    else:
        pad = np.full((jmd_n_len - n_terminus_len, D), np.nan, dtype=np.float64)
        jmd_n = np.concatenate([pad, emb[0:n_terminus_len, :]], axis=0)
    if c_terminus_len >= jmd_c_len:
        jmd_c = emb[tmd_stop : tmd_stop + jmd_c_len, :]
    else:
        pad = np.full((jmd_c_len - c_terminus_len, D), np.nan, dtype=np.float64)
        jmd_c = np.concatenate([emb[tmd_stop:, :], pad], axis=0)
    return tmd, jmd_n, jmd_c


def _get_dict_part_num(tmd=None, jmd_n=None, jmd_c=None):
    """Tensor analog of ``ut.get_dict_part_seq``: per-(part name) sliced tensors.

    Uses ``options['ext_len']`` for ``ext_n`` / ``ext_c`` analogously to the
    string version (axis-0 slicing replaces character slicing; ``np.concatenate``
    on axis 0 replaces string ``+``).
    """
    ext_len = ut.options["ext_len"] or 0
    tmd_n = tmd[0 : round(len(tmd) / 2), :]
    tmd_c = tmd[round(len(tmd) / 2) :, :]
    D = tmd.shape[1]
    if ext_len > 0:
        ext_n = jmd_n[-ext_len:, :] if len(jmd_n) >= ext_len else jmd_n
        ext_c = jmd_c[:ext_len, :] if len(jmd_c) >= ext_len else jmd_c
    else:
        ext_n = np.empty((0, D), dtype=np.float64)
        ext_c = np.empty((0, D), dtype=np.float64)
    return {
        "tmd": tmd,
        "tmd_e": np.concatenate([ext_n, tmd, ext_c], axis=0),
        "tmd_n": tmd_n,
        "tmd_c": tmd_c,
        "jmd_n": jmd_n,
        "jmd_c": jmd_c,
        "ext_n": ext_n,
        "ext_c": ext_c,
        "tmd_jmd": np.concatenate([jmd_n, tmd, jmd_c], axis=0),
        "jmd_n_tmd_n": np.concatenate([jmd_n, tmd_n], axis=0),
        "tmd_c_jmd_c": np.concatenate([tmd_c, jmd_c], axis=0),
        "ext_n_tmd_n": np.concatenate([ext_n, tmd_n], axis=0),
        "tmd_c_ext_c": np.concatenate([tmd_c, ext_c], axis=0),
    }


def assign_dict_num_to_parts(df_seq=None, dict_num=None, list_parts=None,
                             jmd_n_len=None, jmd_c_len=None):
    """Build ``(dict_part_vals, dict_part_lens)`` from a per-residue tensor dict.

    Parameters
    ----------
    df_seq : pd.DataFrame
        Must contain ``entry``, ``tmd_start``, ``tmd_stop`` columns (the
        position-based schema). Row order defines the n_samples axis.
    dict_num : Dict[str, np.ndarray]
        ``entry -> (L, D)`` per-residue numerical tensor.
    list_parts : list[str]
        Part names to materialize (same options as ``get_df_parts``).
    jmd_n_len, jmd_c_len : int
        JMD lengths; matches the constructor.

    Returns
    -------
    dict_part_vals : Dict[str, np.ndarray (n_samples, L_part_max, D) float64]
        NaN-padded per-part tensors. The same output contract as
        ``assign_scale_values_to_seq_num`` so downstream stages are agnostic
        of the value source.
    dict_part_lens : Dict[str, np.ndarray (n_samples,) int64]
        Real per-sample sequence length for each part.
    """
    n_samples = len(df_seq)
    # All entries share D (validated upstream in CPP.run_num).
    D = next(iter(dict_num.values())).shape[1]

    list_part_arrays = {part: [] for part in list_parts}
    list_part_lens = {part: np.zeros(n_samples, dtype=np.int64) for part in list_parts}

    entries = df_seq[ut.COL_ENTRY].to_list()
    tmd_starts = df_seq[ut.COL_TMD_START].to_list()
    tmd_stops = df_seq[ut.COL_TMD_STOP].to_list()
    for i, entry in enumerate(entries):
        emb = dict_num[entry]
        tmd, jmd_n, jmd_c = _slice_dict_num_to_basic_parts(
            emb=emb, tmd_start=tmd_starts[i], tmd_stop=tmd_stops[i],
            jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
        )
        part_arrs = _get_dict_part_num(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c)
        for part in list_parts:
            arr_i = part_arrs[part]
            list_part_arrays[part].append(arr_i)
            list_part_lens[part][i] = arr_i.shape[0]

    dict_part_vals = {}
    dict_part_lens = {}
    for part in list_parts:
        L_max = max(a.shape[0] for a in list_part_arrays[part]) if list_part_arrays[part] else 0
        if L_max == 0:
            dict_part_vals[part] = np.empty((n_samples, 0, D), dtype=np.float64)
            dict_part_lens[part] = list_part_lens[part]
            continue
        arr = np.full((n_samples, L_max, D), np.nan, dtype=np.float64)
        for i, a in enumerate(list_part_arrays[part]):
            if a.shape[0]:
                arr[i, : a.shape[0], :] = a
        dict_part_vals[part] = arr
        dict_part_lens[part] = list_part_lens[part]

    return dict_part_vals, dict_part_lens
