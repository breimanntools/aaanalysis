"""Equivalence tests for the Batch-3 window-sampler vectorizations.

`candidate_centers_` (integer L1 distances), `_scan_protein_` (left-to-right PWM
sums), and `score_window_pwm_` (hoisted aa-index) are all bit-identical to their
original scalar forms; these tests pin that against reference implementations."""
import numpy as np
import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis.seq_analysis._backend.aa_window_sampler._utils import (
    candidate_centers_, score_window_pwm_, window_offsets)
from aaanalysis.seq_analysis._backend.aa_window_sampler.sample_motif_matched import _scan_protein_

aa.options["verbose"] = False
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(ut.LIST_CANONICAL_AA)}


# ---------------- reference (original) implementations ----------------
def _ref_candidate_centers(seq_len, window_size, exclude_positions=None, min_distance=None, max_distance=None):
    half_left, half_right = window_offsets(window_size)
    lo, hi = half_left, seq_len - half_right + 1
    if hi <= lo:
        return []
    centers = list(range(lo, hi))
    if not exclude_positions or (min_distance is None and max_distance is None):
        return centers
    excl = [p - 1 for p in exclude_positions]
    result = []
    for c in centers:
        d = min(abs(c - p) for p in excl)
        if min_distance is not None and d < min_distance:
            continue
        if max_distance is not None and d > max_distance:
            continue
        result.append(c)
    return result


def _ref_scan_protein(seq, pwm, window_size, aa_index, threshold, allowed_centers=None):
    half_left, half_right = window_offsets(window_size)
    n = len(seq)
    if n < window_size:
        return []
    aa_idx_arr = np.array([aa_index.get(c, -1) for c in seq], dtype=np.int64)
    hits = []
    for c in range(half_left, n - half_right + 1):
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


def _ref_score_window_pwm(window, pwm):
    score = 0.0
    for i, aa_ in enumerate(window):
        j = AA_INDEX.get(aa_)
        if j is not None:
            score += float(pwm[i, j])
    return score


# ---------------- tests ----------------
class TestCandidateCentersEquivalence:
    @pytest.mark.parametrize("min_d,max_d", [(5, 50), (3, None), (None, 20), (None, None), (0, 0)])
    @pytest.mark.parametrize("seq_len,ws", [(2000, 15), (500, 9), (60, 7)])
    def test_matches_reference(self, min_d, max_d, seq_len, ws):
        excl = list(range(1, 200, 3)) + [seq_len - 5]
        excl = [p for p in excl if 1 <= p <= seq_len]
        kw = dict(seq_len=seq_len, window_size=ws, exclude_positions=excl,
                  min_distance=min_d, max_distance=max_d)
        assert candidate_centers_(**kw) == _ref_candidate_centers(**kw)

    def test_edge_cases(self):
        assert candidate_centers_(8, 15) == _ref_candidate_centers(8, 15)            # hi<=lo
        assert candidate_centers_(100, 9, [], 5, 10) == _ref_candidate_centers(100, 9, [], 5, 10)
        assert candidate_centers_(100, 9, [5]) == _ref_candidate_centers(100, 9, [5])  # both bounds None


class TestScanProteinEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    @pytest.mark.parametrize("threshold", [-1e9, 0.0, 5.0])
    def test_matches_reference(self, seed, threshold):
        rng = np.random.default_rng(seed)
        ws = 15
        pwm = rng.standard_normal((ws, 20))
        seq = "".join(rng.choice(list(AA + "XBZ"), 800))  # include non-canonical residues
        assert _scan_protein_(seq, pwm, ws, AA_INDEX, threshold) == \
               _ref_scan_protein(seq, pwm, ws, AA_INDEX, threshold)

    def test_allowed_centers_and_short_seq(self):
        rng = np.random.default_rng(3)
        ws = 9
        pwm = rng.standard_normal((ws, 20))
        seq = "".join(rng.choice(list(AA), 200))
        allowed = set(range(40, 160))
        assert _scan_protein_(seq, pwm, ws, AA_INDEX, 1.0, allowed) == \
               _ref_scan_protein(seq, pwm, ws, AA_INDEX, 1.0, allowed)
        assert _scan_protein_("ACDEF", pwm, ws, AA_INDEX, 0.0) == []  # n < window_size


class TestScoreWindowPwmEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        ws = 15
        pwm = rng.standard_normal((ws, 20))
        for _ in range(50):
            w = "".join(rng.choice(list(AA + "X"), ws))
            assert score_window_pwm_(w, pwm) == _ref_score_window_pwm(w, pwm)
