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
# Canonical amino-acid -> column index (built once; identical to rebuilding it per call).
_AA_INDEX = {a: i for i, a in enumerate(ut.LIST_CANONICAL_AA)}


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
    # Nearest-excluded L1 distance per center via binary search into the sorted excluded
    # positions: the nearest excluded position is one of the two neighbours bracketing the
    # center. O(n_centers · log n_excl) time and O(n_centers) memory — no n_centers × n_excl
    # matrix. Distances are integers, so the band comparisons reproduce the scalar decisions.
    centers_arr = np.asarray(centers)
    excl = np.unique(np.asarray([p - 1 for p in exclude_positions]))
    pos = np.searchsorted(excl, centers_arr)
    left = np.abs(centers_arr - excl[np.clip(pos - 1, 0, len(excl) - 1)])
    right = np.abs(centers_arr - excl[np.clip(pos, 0, len(excl) - 1)])
    d = np.minimum(left, right)
    keep = np.ones(len(centers_arr), dtype=bool)
    if min_distance is not None:
        keep &= d >= min_distance
    if max_distance is not None:
        keep &= d <= max_distance
    return centers_arr[keep].tolist()


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


def _encode_equal_length(seqs):
    """Encode a list of equal-length strings as a ``(n, L)`` uint8 array.

    Returns ``None`` when the list is empty or the strings are ragged, so callers
    fall back to the exact scalar path (which handles ragged windows by treating
    different-length pairs as identity 0.0).
    """
    if not seqs:
        return None
    length = len(seqs[0])
    if length == 0 or any(len(s) != length for s in seqs):
        return None
    try:
        buf = "".join(seqs).encode("latin-1")
    except UnicodeEncodeError:
        # Non-latin-1 residues (never present in real protein windows): fall back to the
        # exact scalar path, which compares characters directly like the original.
        return None
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(seqs), length)


def filter_similarity_to_test(windows, test_windows, max_similarity):
    """Drop windows whose identity to any test window exceeds ``max_similarity``.

    Returns
    -------
    kept : list of str
    mask : np.ndarray of bool, shape (n_windows,)
    """
    if max_similarity is None or not test_windows:
        return list(windows), np.ones(len(windows), dtype=bool)
    W = _encode_equal_length(windows)
    T = _encode_equal_length(test_windows)
    if W is None or T is None or W.shape[1] != T.shape[1]:
        # Ragged / mismatched lengths: identity uses the exact scalar rule (0.0 across lengths).
        mask = np.array([
            not any(window_identity(w, tw) > max_similarity for tw in test_windows)
            for w in windows
        ])
        return [w for w, k in zip(windows, mask) if k], mask
    # identity(w, tw) == fraction of matching positions (an exact integer ratio), so the
    # vectorized comparison reproduces the scalar decisions exactly. Loop over the (few)
    # test windows to keep memory at O(n_windows) instead of O(n_windows * n_test).
    mask = np.ones(len(windows), dtype=bool)
    for t in T:
        mask &= ~((W == t).mean(axis=1) > max_similarity)
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
    W = _encode_equal_length(windows)
    if W is None:
        # Ragged windows: keep the exact scalar greedy path.
        kept, mask = [], []
        for w in windows:
            is_dup = any(window_identity(w, k) > max_similarity for k in kept)
            mask.append(not is_dup)
            if not is_dup:
                kept.append(w)
        return kept, np.array(mask)
    # Same greedy order/decisions, but each "is this a duplicate of a kept window?" check
    # compares against all kept windows at once. Kept rows accumulate in a preallocated
    # buffer so the per-window check reads a contiguous view (no per-iteration gather/copy).
    n, length = W.shape
    mask = np.zeros(n, dtype=bool)
    kept_buf = np.empty((n, length), dtype=W.dtype)
    m = 0
    for i in range(n):
        wi = W[i]
        # (matches / length) reproduces window_identity bit-for-bit, so the decision matches.
        if m and bool(((kept_buf[:m] == wi).mean(axis=1) > max_similarity).any()):
            continue
        mask[i] = True
        kept_buf[m] = wi
        m += 1
    return [windows[i] for i in range(n) if mask[i]], mask


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
    score = 0.0
    for i, aa in enumerate(window):
        j = _AA_INDEX.get(aa)
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


def make_safe_custom_predicate_(custom_filter, resolve):
    """Adapt a user ``(window, entry, source_position) -> bool`` keep-filter into the
    backend ``(window, payload) -> bool`` predicate consumed by
    ``sample_pool_iteratively_``.

    ``resolve(payload)`` maps the sampler-specific payload to the user-facing
    ``(entry, source_position)`` pair (1-based anchor). The return value is
    ``bool()``-coerced. If the user filter raises, the error is re-raised as a
    ``RuntimeError`` naming the offending window (chained from the original via
    ``from e``) so the user sees which window triggered it.
    """
    def predicate(window, payload):
        entry, source_position = resolve(payload)
        try:
            return bool(custom_filter(window, entry, source_position))
        except Exception as e:
            raise RuntimeError(
                f"'custom_filter' raised on window {window!r} "
                f"(entry={entry!r}, source_position={source_position})"
            ) from e
    return predicate


def sample_pool_iteratively_(*, draw_batch, target_n, test_windows,
                             max_similarity_to_test, max_similarity_within_ref,
                             motif_pwm, motif_score_threshold, motif_match,
                             max_attempts, filter_iteratively,
                             accepted_windows=None, custom_predicate=None):
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
    custom_predicate : callable(window, payload) -> bool, optional
        User-supplied keep-predicate (already bound by the caller to the window's
        source entry / position). Runs **last** in the per-window pipeline
        (after motif and both similarity filters); a window is accepted only when
        it returns ``True``. ``None`` disables the hook.

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
                 and motif_pwm is None
                 and custom_predicate is None)
    attempts = 0
    while len(accepted) < target_n and attempts < max_attempts:
        batch = draw_batch(target_n - len(accepted))
        if not batch:
            break
        for window, payload in batch:
            if not passes_motif_filter_(window, motif_pwm,
                                         motif_score_threshold, motif_match):
                continue
            if not apply_similarity_filters(window, test_windows, accepted_windows,
                                            max_similarity_to_test,
                                            max_similarity_within_ref):
                continue
            if custom_predicate is not None and not custom_predicate(window, payload):
                continue
            accepted.append((window, payload))
            accepted_windows.append(window)
            if len(accepted) >= target_n:
                break
        attempts += 1
        if not filter_iteratively or no_filter:
            break
    return accepted
