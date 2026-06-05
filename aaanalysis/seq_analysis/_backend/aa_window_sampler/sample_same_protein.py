"""
This is a script for the backend of the AAWindowSampler.sample_same_protein() method.

Protein iteration order is randomized under the seed so the output depends only
on ``df_seq`` *content* and the seed, not on ``df_seq`` row order. The
cross-protein redundancy buffer (``max_similarity_within_ref`` applied across
proteins) was previously sensitive to row order.

The total budget ``n`` is split uniformly across eligible source proteins
(each gets ~``n / n_proteins``; the first ``n % n_proteins`` proteins in
shuffled iteration order get one extra). If a protein cannot supply its quota
(small candidate pool or aggressive filtering), the shortfall is redistributed
via a round-robin top-up over proteins with remaining capacity.
"""
import warnings

import aaanalysis.utils as ut
from ._utils import (candidate_centers_,
                     make_safe_custom_predicate_,
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


def _uniform_quotas(n, n_proteins):
    """Split ``n`` into ``n_proteins`` integer quotas summing to ``n``.

    Each protein gets ``n // n_proteins`` windows; the first ``n % n_proteins``
    proteins (in caller-provided iteration order) get one extra.
    """
    base = n // n_proteins
    rem = n - base * n_proteins
    quotas = [base] * n_proteins
    for i in range(rem):
        quotas[i] += 1
    return quotas


# II Main Functions
def sample_same_protein(*, df_seq, positions, n, window_size,
                          min_distance_to_pos, max_distance_to_pos,
                          test_windows, allowed_positions,
                          max_similarity_to_test, max_similarity_within_ref,
                          motif_pwm, motif_score_threshold, motif_match,
                          max_sampling_attempts, filter_iteratively, rng, verbose,
                          custom_filter=None):
    """Build the pool of accepted same-protein windows.

    Parameters
    ----------
    n : int
        Total target number of sampled windows across all eligible proteins.
    min_distance_to_pos, max_distance_to_pos : int or None
        Distance band (in residues) from the nearest positive on the same
        protein. ``None`` on either side drops that bound.
    allowed_positions : list of list of int or None
        Per ``df_seq`` row, the 1-based positions that pass the per-residue context
        filter (e.g. from ``_filter_aa_context``). When ``None`` no context
        constraint is applied.

    Returns
    -------
    rows : list of [entry, sequence, window, source_position]
    source_indices : list of int
        For each row, the corresponding ``df_seq`` row index.
    sampled_centers : list of list of int
        Per ``df_seq`` row, the 0-based centers retained (used for sequences-mode output).
    """
    entries = df_seq[ut.COL_ENTRY].tolist()
    seqs = df_seq[ut.COL_SEQ].tolist()
    half_left, _ = window_offsets(window_size)
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
    # Build per-protein candidate-center lists up front; drop proteins with no
    # valid centers so the quota split only considers proteins that can supply.
    prot_data = []  # list of (df_seq_index, centers, draw_batch)
    for i in eligible_idx:
        seq = seqs[i]
        centers = candidate_centers_(len(seq), window_size,
                                      exclude_positions=positions[i],
                                      min_distance=min_distance_to_pos,
                                      max_distance=max_distance_to_pos)
        if allowed_positions is not None:
            allowed_centers = {p - 1 for p in allowed_positions[i]}
            centers = [c for c in centers if c in allowed_centers]
        if not centers:
            if verbose:
                warnings.warn(f"No valid windows for entry '{entries[i]}' "
                              f"(seq_len={len(seq)}, window_size={window_size}, "
                              f"min_distance_to_pos={min_distance_to_pos}, "
                              f"max_distance_to_pos={max_distance_to_pos})",
                              RuntimeWarning)
            continue
        rng.shuffle(centers)
        draw_batch = _draw_batch_from_centers(centers, seq, half_left, window_size)
        prot_data.append((i, centers, draw_batch))
    if not prot_data:
        return [], [], sampled_centers
    quotas = _uniform_quotas(n, len(prot_data))

    accepted_per_protein = [[] for _ in prot_data]

    def _draw_for(slot, target):
        if target <= 0:
            return 0
        i, _, draw_batch = prot_data[slot]
        # ``payload`` is the 0-based center ``c``; bind this protein's entry so
        # the user filter sees (window, entry, 1-based source_position).
        predicate = None
        if custom_filter is not None:
            entry_i = entries[i]
            predicate = make_safe_custom_predicate_(
                custom_filter, lambda c, _e=entry_i: (_e, c + 1))
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
            custom_predicate=predicate,
        )
        accepted_per_protein[slot].extend(accepted)
        return len(accepted)

    # Pass 1: each protein samples up to its quota.
    for slot, target in enumerate(quotas):
        _draw_for(slot, target)
    total_accepted = sum(len(a) for a in accepted_per_protein)
    # Pass 2: round-robin top-up — if any protein fell short of its quota,
    # redistribute the shortfall to proteins with remaining candidates.
    while total_accepted < n:
        progress = 0
        for slot in range(len(prot_data)):
            if total_accepted >= n:
                break
            got = _draw_for(slot, 1)
            progress += got
            total_accepted += got
        if progress == 0:
            break  # every protein is exhausted

    if total_accepted < n and verbose:
        warnings.warn(f"Only {total_accepted}/{n} windows kept across eligible "
                      f"proteins after filtering.",
                      RuntimeWarning)
    # Build output rows in shuffled-protein iteration order, matching the
    # behavior before the quota refactor.
    rows, source_indices = [], []
    for slot, (i, _, _) in enumerate(prot_data):
        entry, seq = entries[i], seqs[i]
        for window, c in accepted_per_protein[slot]:
            rows.append([entry, seq, window, c + 1])
            source_indices.append(i)
            sampled_centers[i].append(c)
    return rows, source_indices, sampled_centers
