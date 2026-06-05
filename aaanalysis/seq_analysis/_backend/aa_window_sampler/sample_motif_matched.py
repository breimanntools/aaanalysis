"""
This is a script for the backend of the AAWindowSampler.sample_motif_matched() method.

Walks every position of every candidate row in ``df_seq``, scores each window of
``window_size`` against the user-supplied PWM (sum of per-position log-odds), and
returns the hits whose score meets the user-supplied threshold. Equivalent to a
FIMO scan with a score-based threshold; no external CLI dependencies.
"""
import warnings
import numpy as np

import aaanalysis.utils as ut
from ._utils import sample_pool_iteratively_, window_offsets


# I Helper Functions
def _scan_protein_(seq, pwm, window_size, aa_index, threshold,
                    allowed_centers=None):
    """Return a list of ``(score, center)`` for every candidate window in ``seq``
    whose PWM score is ``>= threshold``. ``center`` is 0-based.
    """
    half_left, half_right = window_offsets(window_size)
    n = len(seq)
    if n < window_size:
        return []
    # Per-position residue index (-1 for non-canonical).
    aa_idx_arr = np.array([aa_index.get(c, -1) for c in seq], dtype=np.int64)
    hits = []
    lo, hi = half_left, n - half_right + 1
    for c in range(lo, hi):
        if allowed_centers is not None and c not in allowed_centers:
            continue
        score = 0.0
        start = c - half_left
        for i in range(window_size):
            j = aa_idx_arr[start + i]
            if j >= 0:
                score += float(pwm[i, j])
        if score >= threshold:
            hits.append((score, c))
    return hits


def _slice_window_(seq, center, half_left, window_size):
    return seq[center - half_left:center - half_left + window_size]


def _draw_batch_from_pool_(pool, seqs, half_left, window_size):
    """Stateful ``draw_batch`` walking a pre-sorted pool of
    ``(score, entry_name, entry_idx, center)``."""
    cursor = [0]

    def draw_batch(needed):
        start, stop = cursor[0], cursor[0] + needed
        cursor[0] = stop
        return [(_slice_window_(seqs[i], c, half_left, window_size), (i, c, score))
                for score, _entry_name, i, c in pool[start:stop]]

    return draw_batch


# II Main Functions
def sample_motif_matched(*, df_seq, positions, n, window_size,
                          motif_pwm, motif_score_threshold,
                          test_windows, allowed_positions,
                          max_similarity_to_test, max_similarity_within_ref,
                          max_sampling_attempts, filter_iteratively,
                          rng, verbose, custom_filter=None):
    """Build the pool of motif-matched candidate windows.

    Parameters
    ----------
    motif_pwm : np.ndarray
        Position-weight matrix of shape ``(window_size, n_aa)`` with columns
        ordered by ``ut.LIST_CANONICAL_AA``. Required.
    motif_score_threshold : float
        Score threshold (sum of per-position log-odds). Required.
    allowed_positions : list of list of int or None
        Per ``df_seq`` row, the 1-based positions allowed by the per-residue
        context filter. ``None`` = no constraint.

    Returns
    -------
    rows : list of [entry, sequence, window, source_position]
    source_indices : list of int
    sampled_centers : list of list of int
    sampled_scores : list of float
        For each row in ``rows``, the raw PWM score that earned its inclusion
        (sum of per-position values over the window).
    """
    entries = df_seq[ut.COL_ENTRY].tolist()
    seqs = df_seq[ut.COL_SEQ].tolist()
    aa_index = {a: i for i, a in enumerate(ut.LIST_CANONICAL_AA)}
    half_left, _ = window_offsets(window_size)

    # Build the candidate pool from rows with NO positives (eligible_idx),
    # mirroring sample_different_protein's row-eligibility rule.
    eligible_idx = [i for i, p in enumerate(positions) if not p]
    if not eligible_idx:
        raise ValueError("No eligible candidate proteins (rows with no test "
                         "positions) for sample_motif_matched.")
    pool = []
    for i in eligible_idx:
        seq_i = seqs[i]
        entry_i = entries[i]
        allowed_centers = (None if allowed_positions is None
                           else {p - 1 for p in allowed_positions[i]})
        for score, c in _scan_protein_(seq_i, motif_pwm, window_size, aa_index,
                                        motif_score_threshold,
                                        allowed_centers=allowed_centers):
            pool.append((score, entry_i, i, c))
    if not pool:
        if verbose:
            warnings.warn(f"No candidate windows met the motif score threshold "
                          f"({motif_score_threshold}); returning empty result.",
                          RuntimeWarning)
        return [], [], [[] for _ in entries], []
    # Rank by descending score; deterministic tiebreak by entry name then
    # 0-based center. The same key is used by the FIMO wrapper, so the two
    # paths return identical hit sets when the user caps at ``n``.
    pool.sort(key=lambda t: (-t[0], t[1], t[3]))

    draw_batch = _draw_batch_from_pool_(pool, seqs, half_left, window_size)
    # ``payload`` is ``(entry_idx, center, score)``; bind it to (window, entry, pos).
    predicate = None
    if custom_filter is not None:
        predicate = lambda window, payload: custom_filter(
            window, entries[payload[0]], payload[1] + 1)
    accepted = sample_pool_iteratively_(
        draw_batch=draw_batch, target_n=n, test_windows=test_windows,
        max_similarity_to_test=max_similarity_to_test,
        max_similarity_within_ref=max_similarity_within_ref,
        motif_pwm=None, motif_score_threshold=None, motif_match=None,
        max_attempts=max_sampling_attempts,
        filter_iteratively=filter_iteratively,
        custom_predicate=predicate,
    )
    if len(accepted) < n and verbose:
        warnings.warn(f"Only {len(accepted)}/{n} motif-matched windows kept "
                      f"after filtering.",
                      RuntimeWarning)
    rows, source_indices, sampled_scores = [], [], []
    sampled_centers = [[] for _ in entries]
    for window, (i, c, score) in accepted:
        rows.append([entries[i], seqs[i], window, c + 1])
        source_indices.append(i)
        sampled_centers[i].append(c)
        sampled_scores.append(score)
    return rows, source_indices, sampled_centers, sampled_scores
