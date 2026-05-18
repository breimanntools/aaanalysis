"""
This is a script for the backend of the AAWindowSampler.sample_same_protein() method.

Protein iteration order is randomized under the seed so the output depends only
on ``df_seq`` *content* and the seed, not on ``df_seq`` row order. The
cross-protein redundancy buffer (``max_similarity_within_ref`` applied across
proteins) was previously sensitive to row order.
"""
import warnings

import aaanalysis.utils as ut
from ._utils import (candidate_centers_,
                     sample_pool_iteratively_,
                     window_offsets)


# I Helper Functions
def _slice_window(seq, center, half_left, window_size):
    return seq[center - half_left:center - half_left + window_size]


def _draw_batch_from_centers(centers, seq, half_left, window_size):
    """Return a stateful ``draw_batch`` walking ``centers`` once."""
    cursor = [0]

    def draw_batch(needed):
        start, stop = cursor[0], cursor[0] + needed
        cursor[0] = stop
        return [(_slice_window(seq, c, half_left, window_size), c)
                for c in centers[start:stop]]

    return draw_batch


# II Main Functions
def sample_same_protein(*, df_seq, positions, n_per_positive, window_size,
                          min_distance_to_positive, test_windows,
                          allowed_positions,
                          max_similarity_to_test, max_similarity_within_ref,
                          motif_pwm, motif_score_threshold, motif_match,
                          max_sampling_attempts, filter_iteratively, rng, verbose):
    """Build the pool of accepted same-protein windows.

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
        For each row, the corresponding ``df_seq`` row index (used for context-col copy).
    sampled_centers : list of list of int
        Per ``df_seq`` row, the 0-based centers retained (used for sequences-mode output).
    """
    entries = df_seq[ut.COL_ENTRY].tolist()
    seqs = df_seq[ut.COL_SEQ].tolist()
    half_left, _ = window_offsets(window_size)
    rows, source_indices = [], []
    sampled_centers = [[] for _ in entries]
    # Shared across all per-protein sampling calls so max_similarity_within_ref
    # is enforced across protein boundaries within one call.
    cross_protein_accepted_windows = []
    # Randomize protein iteration order under the seed so the cross-protein
    # redundancy buffer evolves deterministically from (content + seed), not
    # from df_seq row order. Sort by entry name first (content-stable key) so
    # the shuffle output depends only on (set of entries, seed), not on the
    # original df_seq row order.
    eligible_idx = sorted((i for i, p in enumerate(positions) if p),
                          key=lambda i: entries[i])
    rng.shuffle(eligible_idx)
    for i in eligible_idx:
        pos_list = positions[i]
        entry, seq = entries[i], seqs[i]
        target = n_per_positive * len(pos_list)
        centers = candidate_centers_(len(seq), window_size,
                                      exclude_positions=pos_list,
                                      min_distance=min_distance_to_positive)
        if allowed_positions is not None:
            allowed_centers = {p - 1 for p in allowed_positions[i]}
            centers = [c for c in centers if c in allowed_centers]
        if not centers:
            if verbose:
                warnings.warn(f"No valid windows for entry '{entry}' "
                              f"(seq_len={len(seq)}, window_size={window_size}, "
                              f"min_distance_to_positive={min_distance_to_positive})",
                              RuntimeWarning)
            continue
        rng.shuffle(centers)
        draw_batch = _draw_batch_from_centers(centers, seq, half_left, window_size)
        accepted = sample_pool_iteratively_(
            draw_batch=draw_batch, target_n=target, test_windows=test_windows,
            max_similarity_to_test=max_similarity_to_test,
            max_similarity_within_ref=max_similarity_within_ref,
            motif_pwm=motif_pwm,
            motif_score_threshold=motif_score_threshold,
            motif_match=motif_match,
            max_attempts=max_sampling_attempts,
            filter_iteratively=filter_iteratively,
            accepted_windows=cross_protein_accepted_windows,
        )
        if len(accepted) < target and verbose:
            warnings.warn(f"Only {len(accepted)}/{target} windows kept "
                          f"for entry '{entry}' after filtering.",
                          RuntimeWarning)
        for window, c in accepted:
            rows.append([entry, seq, window, c + 1])
            source_indices.append(i)
            sampled_centers[i].append(c)
    return rows, source_indices, sampled_centers
