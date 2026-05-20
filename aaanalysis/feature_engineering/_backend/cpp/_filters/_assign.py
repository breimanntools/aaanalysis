"""
This is a script for the backend of CPP's residue-value assignment stage:
``assign_scale_values_to_seq`` looks up scale values for every residue in
``df_parts`` and returns a per-(scale, part) numerical matrix.
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
def _assign_scale_values_to_seq(
    df_parts=None,
    dict_all_scales=None,
    verbose=True,
    shared_max_progress=None,
    shared_value_lock=None,
    print_lock=None,
    prefer_multiprocessing=False,
):
    """Assign scale values to every residue in one pass per part.

    Inverted pipeline (PR 2): builds a per-AA × per-scale lookup table once,
    then for each part fancy-indexes the whole (n_samples, len_seq_max,
    n_scales) tensor in a single vectorized operation. Output structure
    (``dict[scale][part] -> 2D float64 ndarray``) and per-cell numerical
    values are byte-identical to the original per-scale masking loop —
    aggregation is linear, lookup is a pure function of AA letter, so order
    of operations doesn't change the result. The change unlocks PR 3
    (run_embed) by exposing the per-residue numerical tensor as the seam
    where scales lookup can be swapped for precomputed PLM embeddings.
    """
    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        shared_max_progress=shared_max_progress,
        shared_value_lock=shared_value_lock,
        print_lock=print_lock,
        prefer_multiprocessing=prefer_multiprocessing,
    )
    args_p = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)

    list_seq_max = df_parts.map(len).max().tolist()
    list_parts = list(df_parts)
    list_scales = list(dict_all_scales)
    n_scales = len(list_scales)

    if n_scales == 0:
        return {}

    # Build the per-residue lookup table: scale_matrix[aa_idx, scale_idx].
    # All scales in dict_all_scales share the same AA alphabet (built from
    # df_scales.index in the public wrapper). Row n_aa is reserved for
    # "unknown/padding" → NaN, matching the original code's behavior for
    # characters that aren't in the scale dict.
    list_aa = list(dict_all_scales[list_scales[0]].keys())
    aa_to_idx = {aa: i for i, aa in enumerate(list_aa)}
    n_aa = len(list_aa)
    scale_matrix = np.full((n_aa + 1, n_scales), np.nan, dtype=np.float32)
    for s, scale in enumerate(list_scales):
        for aa, idx in aa_to_idx.items():
            scale_matrix[idx, s] = dict_all_scales[scale][aa]

    # Convert sequences into a padded NumPy matrix ("_" as padding).
    # Built once per part, reused across all scales.
    dict_seq_matrix = {}
    len_seq_max_global = max(list_seq_max)
    for part in list_parts:
        X_seq = np.full((len(df_parts), len_seq_max_global), "_", dtype="<U1")
        for i, seq in enumerate(df_parts[part]):
            X_seq[i, : len(seq)] = list(seq)
        dict_seq_matrix[part] = X_seq

    seq_lengths = df_parts.map(len).values
    dict_scale_part_vals = {scale: {} for scale in list_scales}

    # Progress unit changed from per-scale to per-part — each part is one
    # vectorized lookup now. Final state still reports completion.
    progress_interval = max(1, len(list_parts) // 10) if verbose else len(list_parts)

    for j, (part, len_seq_max) in enumerate(zip(list_parts, list_seq_max)):
        X_seq = dict_seq_matrix[part][:, :len_seq_max]
        # Map each residue character to its AA index once per part. The
        # per-AA mask sweep happens once here (n_aa numpy ops per part) and
        # is then amortized across all scales — that's the core win of the
        # inversion versus the original n_scales × n_aa loop.
        aa_idx_matrix = np.full(X_seq.shape, n_aa, dtype=np.int32)
        for aa, idx in aa_to_idx.items():
            aa_idx_matrix[X_seq == aa] = idx
        seq_length_column = seq_lengths[:, j:j + 1]
        # Per-scale fancy index → 2D float32 array, concat with int
        # seq-length column → float64 (matches the original dtype after
        # concatenate). Allocates one 2D array per (scale, part), so peak
        # memory tracks the stored output exactly — no 3D staging tensor.
        for s, scale in enumerate(list_scales):
            X_scale_part_f32 = scale_matrix[aa_idx_matrix, s]
            dict_scale_part_vals[scale][part] = np.concatenate(
                [X_scale_part_f32, seq_length_column], axis=1
            )

        if verbose and (j % progress_interval == 0 or j == len(list_parts) - 1):
            ut.print_progress(i=j, n_total=len(list_parts), **args_p)

    return dict_scale_part_vals


# II Main Functions
def assign_scale_values_to_seq(df_parts=None, df_scales=None, verbose=False, n_jobs=None):
    """Assign scale values to each sequence with optimized dictionary lookups."""
    list_scales = list(df_scales)
    dict_all_scales = {col: dict(zip(df_scales.index.to_list(), df_scales[col])) for col in list_scales}

    if n_jobs is None:
        n_samples, n_scales = len(df_parts), len(dict_all_scales)
        n_jobs = min(os.cpu_count() or 1, max(min(int(n_scales / 100), int(n_samples / 100)), 1))

    # Use multiprocessing-shared progress only if we truly use >1 process
    prefer_mp_progress = bool(n_jobs and n_jobs > 1)

    shared_max_progress, shared_value_lock, print_lock = _resolve_shared(
        prefer_multiprocessing=prefer_mp_progress
    )
    args_print = dict(shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock, print_lock=print_lock)
    args = dict(df_parts=df_parts, verbose=verbose, prefer_multiprocessing=prefer_mp_progress, **args_print)

    _reset_progress(shared_max_progress, shared_value_lock)

    if n_jobs == 1:
        out = _assign_scale_values_to_seq(dict_all_scales=dict_all_scales, **args)
        if verbose:
            ut.print_end_progress(add_new_line=False, shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)
        if prefer_mp_progress:
            _cleanup_mp_manager()
        return out

    def _mp_scale_assignment(scales_chunk, shared_max_progress, shared_value_lock, print_lock):
        # Workers use the passed-in shared objects; they must NOT create a Manager.
        chunked_dict_scales = {scale: dict_all_scales[scale] for scale in scales_chunk}
        return _assign_scale_values_to_seq(
            dict_all_scales=chunked_dict_scales,
            df_parts=df_parts,
            verbose=verbose,
            shared_max_progress=shared_max_progress,
            shared_value_lock=shared_value_lock,
            print_lock=print_lock,
            prefer_multiprocessing=False,  # shared objects already provided
        )

    scale_chunks = np.array_split(list(dict_all_scales.keys()), n_jobs)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_mp_scale_assignment)(chunk, shared_max_progress, shared_value_lock, print_lock)
            for chunk in scale_chunks
        )

    dict_scale_part_vals = {}
    for r in results:
        dict_scale_part_vals.update(r)

    if verbose:
        ut.print_end_progress(add_new_line=False, shared_max_progress=shared_max_progress, shared_value_lock=shared_value_lock)

    # Cleanup Manager if we created it
    if prefer_mp_progress:
        _cleanup_mp_manager()

    return dict_scale_part_vals
