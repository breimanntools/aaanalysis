"""
This is a script for the backend of the AAWindowSampler.sample_different_protein() method.
"""
import warnings

import aaanalysis.utils as ut
from ._utils import (candidate_centers_,
                     sample_pool_iteratively_,
                     window_offsets)


# I Helper Functions
def _slice_window(seq, center, half_left, window_size):
    return seq[center - half_left:center - half_left + window_size]


def _draw_batch_from_pool(pool, seqs, half_left, window_size):
    """Return a stateful ``draw_batch`` walking a pre-shuffled pool of (entry_idx, center)."""
    cursor = [0]

    def draw_batch(needed):
        start, stop = cursor[0], cursor[0] + needed
        cursor[0] = stop
        return [(_slice_window(seqs[i], c, half_left, window_size), (i, c))
                for i, c in pool[start:stop]]

    return draw_batch


# II Main Functions
def sample_different_protein(*, df_seq, positions, n, window_size,
                               candidate_proteins, test_windows,
                               allowed_positions,
                               max_similarity_to_test, max_similarity_within_ref,
                               motif_pwm, motif_score_threshold, motif_match,
                               max_sampling_attempts, filter_iteratively,
                               rng, verbose, custom_filter=None):
    """Build the pool of accepted different-protein windows.

    Parameters
    ----------
    allowed_positions : list of list of int or None
        Per ``df_seq`` row, the 1-based positions that pass the per-residue context
        filter (e.g. from ``_filter_aa_context``). When ``None`` no context
        constraint is applied.

    Returns
    -------
    rows : list of [entry, sequence, window, source_position]
    source_indices : list of int
    sampled_centers : list of list of int
    """
    entries = df_seq[ut.COL_ENTRY].tolist()
    seqs = df_seq[ut.COL_SEQ].tolist()
    eligible_idx = [i for i, p in enumerate(positions) if not p]
    if candidate_proteins is not None:
        cand_set = set(candidate_proteins)
        eligible_idx = [i for i in eligible_idx if entries[i] in cand_set]
        kept_set = {entries[i] for i in eligible_idx}
        missing = sorted(cand_set - kept_set)
        if missing and verbose:
            preview = missing[:5]
            more = "..." if len(missing) > 5 else ""
            warnings.warn(f"{len(missing)} 'candidate_proteins' were not eligible "
                          f"(missing or contain test positions): {preview}{more}",
                          UserWarning)
    if not eligible_idx:
        raise ValueError("No eligible proteins (proteins with no test positions) "
                         "for sample_different_protein.")
    half_left, _ = window_offsets(window_size)
    pool = []
    for i in eligible_idx:
        centers = candidate_centers_(len(seqs[i]), window_size)
        if allowed_positions is not None:
            allowed_centers = {p - 1 for p in allowed_positions[i]}
            centers = [c for c in centers if c in allowed_centers]
        pool.extend((i, c) for c in centers)
    if not pool:
        raise ValueError(f"No eligible protein is long enough for window_size={window_size}.")
    rng.shuffle(pool)
    draw_batch = _draw_batch_from_pool(pool, seqs, half_left, window_size)
    # ``payload`` is ``(entry_idx, center)``; bind it to (window, entry, pos).
    predicate = None
    if custom_filter is not None:
        predicate = lambda window, payload: custom_filter(
            window, entries[payload[0]], payload[1] + 1)
    accepted = sample_pool_iteratively_(
        draw_batch=draw_batch, target_n=n, test_windows=test_windows,
        max_similarity_to_test=max_similarity_to_test,
        max_similarity_within_ref=max_similarity_within_ref,
        motif_pwm=motif_pwm,
        motif_score_threshold=motif_score_threshold,
        motif_match=motif_match,
        max_attempts=max_sampling_attempts,
        filter_iteratively=filter_iteratively,
        custom_predicate=predicate,
    )
    if len(accepted) < n and verbose:
        warnings.warn(f"Only {len(accepted)}/{n} windows kept across eligible proteins "
                      f"after filtering.",
                      RuntimeWarning)
    rows, source_indices = [], []
    sampled_centers = [[] for _ in entries]
    for window, (i, c) in accepted:
        rows.append([entries[i], seqs[i], window, c + 1])
        source_indices.append(i)
        sampled_centers[i].append(c)
    return rows, source_indices, sampled_centers
