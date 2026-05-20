"""
This is a script for the shared backend utilities of the AAWindowSampler class:
position parsing, candidate-center construction, test-window collection,
identity-based similarity, similarity / redundancy filters, and the iterative
sampling loop.
"""
import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
def _is_missing(val):
    """Treat ``None`` / ``NaN`` / empty string as 'no positive positions'."""
    return (val is None
            or (isinstance(val, float) and np.isnan(val))
            or (isinstance(val, str) and val.strip() == ""))


def _parse_pos_value(entry, val, pos_col):
    """Parse a single ``pos_col`` cell into a list of 1-based int positions."""
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        try:
            ints = [int(v) for v in val]
        except (TypeError, ValueError):
            raise ValueError(f"'{pos_col}' for entry '{entry}' must contain int-castable "
                             f"positions; got {val!r}")
    else:
        try:
            ints = [int(val)]
        except (TypeError, ValueError):
            raise ValueError(f"'{pos_col}' for entry '{entry}' must be int or iterable of "
                             f"ints; got {type(val).__name__}")
    if any(p < 1 for p in ints):
        raise ValueError(f"Positions in '{pos_col}' for entry '{entry}' must use 1-based "
                         f"indexing (>=1); got {ints}")
    return ints


# II Main Functions
def window_offsets(window_size):
    """Return ``(half_left, half_right)`` with ``half_left + half_right == window_size``.

    Floors left, ceils right — for even ``window_size`` the window is
    right-heavy. The anchor residue ``c`` is the **P1** residue under
    Schechter–Berger cleavage convention; window indices are
    ``[c - half_left, c + half_right)``. Odd sizes split symmetrically.
    """
    half_left = (window_size - 1) // 2
    half_right = window_size - half_left
    return half_left, half_right


def parse_pos_col(df_seq, pos_col):
    """Parse ``pos_col`` into a list-of-lists of 1-based int positions, one entry per row."""
    if pos_col not in df_seq.columns:
        raise ValueError(f"'pos_col' ('{pos_col}') is not a column of 'df_seq'. "
                         f"Columns: {list(df_seq.columns)}")
    parsed = []
    for entry, val in zip(df_seq[ut.COL_ENTRY], df_seq[pos_col]):
        ints = [] if _is_missing(val) else _parse_pos_value(entry, val, pos_col)
        parsed.append(ints)
    return parsed


def candidate_centers_(seq_len, window_size, exclude_positions=None,
                       min_distance=None, max_distance=None):
    """Return the 0-based center indices for valid windows of ``window_size``.

    A center ``c`` is the **P1** residue under cleavage convention; the window
    spans ``[c - half_left, c + half_right)`` per :func:`window_offsets` and
    must lie fully within ``[0, seq_len)``.

    When ``exclude_positions`` (1-based) is non-empty, the ``(min_distance,
    max_distance)`` band filters centers by their L1 distance to the *nearest*
    excluded position:

    - ``min_distance`` (``None`` drops the lower bound): admit ``c`` only if the
      nearest excluded position is at least ``min_distance`` residues away.
    - ``max_distance`` (``None`` drops the upper bound): admit ``c`` only if the
      nearest excluded position is at most ``max_distance`` residues away.

    With both bounds ``None`` (or no excluded positions), every fully-fitting
    center is returned.
    """
    half_left, half_right = window_offsets(window_size)
    lo, hi = half_left, seq_len - half_right + 1
    if hi <= lo:
        return []
    centers = list(range(lo, hi))
    if not exclude_positions:
        return centers
    if min_distance is None and max_distance is None:
        return centers
    excl_0based = [p - 1 for p in exclude_positions]
    result = []
    for c in centers:
        d = min(abs(c - p) for p in excl_0based)
        if min_distance is not None and d < min_distance:
            continue
        if max_distance is not None and d > max_distance:
            continue
        result.append(c)
    return result


def collect_test_windows(df_seq, pos_col, window_size):
    """Extract every fully-fitting test window of length ``window_size`` from ``df_seq``."""
    positions = parse_pos_col(df_seq, pos_col)
    half_left, _ = window_offsets(window_size)
    windows = []
    for seq, pos_list in zip(df_seq[ut.COL_SEQ], positions):
        seq_len = len(seq)
        for p in pos_list:
            start = (p - 1) - half_left
            stop = start + window_size
            if 0 <= start and stop <= seq_len:
                windows.append(seq[start:stop])
    return windows


def window_identity(a, b):
    """Per-position residue identity for two equal-length windows.

    .. math::
        identity(a, b) = \\frac{1}{L} \\sum_{i=0}^{L-1} \\mathbf{1}[a_i = b_i]

    where ``L = len(a) = len(b)``. Returns ``0.0`` if the windows have different lengths.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / n


def filter_similarity_to_test(windows, test_windows, max_similarity):
    """Drop windows whose identity to any test window exceeds ``max_similarity``.

    Returns
    -------
    kept : list of str
    mask : np.ndarray of bool, shape (n_windows,)
    """
    if max_similarity is None or not test_windows:
        return list(windows), np.ones(len(windows), dtype=bool)
    mask = np.array([
        not any(window_identity(w, tw) > max_similarity for tw in test_windows)
        for w in windows
    ])
    return [w for w, k in zip(windows, mask) if k], mask


def filter_redundancy(windows, max_similarity):
    """Greedy redundancy filter: keep each window unless its identity to a previously
    kept window exceeds ``max_similarity``.

    Returns
    -------
    kept : list of str
    mask : np.ndarray of bool, shape (n_windows,)
    """
    if max_similarity is None:
        return list(windows), np.ones(len(windows), dtype=bool)
    kept, mask = [], []
    for w in windows:
        is_dup = any(window_identity(w, k) > max_similarity for k in kept)
        mask.append(not is_dup)
        if not is_dup:
            kept.append(w)
    return kept, np.array(mask)


def apply_similarity_filters(window, test_windows, accepted_windows,
                              max_similarity_to_test, max_similarity_within_ref):
    """Return ``True`` if ``window`` passes both similarity filters."""
    if max_similarity_to_test is not None and test_windows:
        if any(window_identity(window, tw) > max_similarity_to_test for tw in test_windows):
            return False
    if max_similarity_within_ref is not None and accepted_windows:
        if any(window_identity(window, kw) > max_similarity_within_ref
               for kw in accepted_windows):
            return False
    return True


def score_window_pwm_(window, pwm):
    """Score a window against a position-weight matrix.

    PWM has shape ``(window_size, len(ut.LIST_CANONICAL_AA))`` and columns are
    ordered by ``ut.LIST_CANONICAL_AA``. The score is the sum over positions of
    ``pwm[i, aa_index[w[i]]]``; non-canonical residues contribute zero.
    """
    aa_index = {a: i for i, a in enumerate(ut.LIST_CANONICAL_AA)}
    score = 0.0
    for i, aa in enumerate(window):
        j = aa_index.get(aa)
        if j is not None:
            score += float(pwm[i, j])
    return score


def passes_motif_filter_(window, motif_pwm, motif_score_threshold, motif_match):
    """Return ``True`` if ``window`` passes the motif-match filter.

    ``motif_match='in'`` keeps windows with ``score >= threshold``; ``'out'``
    keeps windows with ``score < threshold``. When ``motif_pwm`` is ``None`` the
    filter is a no-op and always returns ``True``.
    """
    if motif_pwm is None:
        return True
    score = score_window_pwm_(window, motif_pwm)
    if motif_match == ut.STR_MOTIF_IN:
        return score >= motif_score_threshold
    return score < motif_score_threshold


def sample_pool_iteratively_(*, draw_batch, target_n, test_windows,
                             max_similarity_to_test, max_similarity_within_ref,
                             motif_pwm, motif_score_threshold, motif_match,
                             max_attempts, filter_iteratively,
                             accepted_windows=None):
    """Iteratively draw and filter ``(window, payload)`` candidates.

    Parameters
    ----------
    draw_batch : callable(int) -> list of (window, payload)
        Returns up to ``needed`` next candidates. An empty list signals that no further
        candidates are available (finite pool exhausted).
    target_n : int
    test_windows : list of str
    max_similarity_to_test : float or None
    max_similarity_within_ref : float or None
    motif_pwm : np.ndarray or None
        Position-weight matrix of shape ``(window_size, n_aa)`` with columns
        ordered by ``ut.LIST_CANONICAL_AA``. When ``None`` the motif filter is
        skipped.
    motif_score_threshold : float or None
        Score threshold for the motif filter. Required when ``motif_pwm`` is set.
    motif_match : str or None
        ``'in'`` (keep ``score >= threshold``) or ``'out'`` (keep ``score < threshold``).
    max_attempts : int
        Hard cap on draw rounds.
    filter_iteratively : bool
        If ``False``, stop after the first round (no re-draw).
    accepted_windows : list of str, optional
        External list used as the within-ref filter buffer. When provided, it is
        mutated in place: newly accepted windows are appended so the caller can
        carry the buffer across multiple invocations (cross-call redundancy).

    Returns
    -------
    list of (window, payload)
        Up to ``target_n`` accepted candidates.
    """
    accepted = []
    if accepted_windows is None:
        accepted_windows = []
    no_filter = (max_similarity_to_test is None
                 and max_similarity_within_ref is None
                 and motif_pwm is None)
    attempts = 0
    while len(accepted) < target_n and attempts < max_attempts:
        batch = draw_batch(target_n - len(accepted))
        if not batch:
            break
        for window, payload in batch:
            if not passes_motif_filter_(window, motif_pwm,
                                         motif_score_threshold, motif_match):
                continue
            if apply_similarity_filters(window, test_windows, accepted_windows,
                                         max_similarity_to_test,
                                         max_similarity_within_ref):
                accepted.append((window, payload))
                accepted_windows.append(window)
                if len(accepted) >= target_n:
                    break
        attempts += 1
        if not filter_iteratively or no_filter:
            break
    return accepted
