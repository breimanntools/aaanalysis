"""This is a script to test the backend identity-based similarity / redundancy filters
in :mod:`aaanalysis.seq_analysis._backend.aa_window_sampler._utils`."""
import numpy as np
import pytest
from hypothesis import settings
import aaanalysis as aa
from aaanalysis.seq_analysis._backend.aa_window_sampler._utils import (
    filter_similarity_to_test,
    filter_redundancy,
    window_identity,
)

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# I Helper Functions
TEST_WINDOWS = ["AAAA", "CCCC", "GGGG"]
CAND_WINDOWS = ["AAAA", "AAAB", "DEFG", "CCCD", "WWWW"]


def _ref_filter_similarity_to_test(windows, test_windows, max_similarity):
    """Reference (non-vectorized) implementation used to pin the optimized output."""
    if max_similarity is None or not test_windows:
        return list(windows), np.ones(len(windows), dtype=bool)
    mask = np.array([not any(window_identity(w, tw) > max_similarity for tw in test_windows)
                     for w in windows])
    return [w for w, k in zip(windows, mask) if k], mask


def _ref_filter_redundancy(windows, max_similarity):
    """Reference (non-vectorized) greedy implementation used to pin the optimized output."""
    if max_similarity is None:
        return list(windows), np.ones(len(windows), dtype=bool)
    kept, mask = [], []
    for w in windows:
        is_dup = any(window_identity(w, k) > max_similarity for k in kept)
        mask.append(not is_dup)
        if not is_dup:
            kept.append(w)
    return kept, np.array(mask)


def _random_windows(n, length, seed):
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    windows = ["".join(rng.choice(aas, length)) for _ in range(n)]
    # inject duplicates / near-duplicates so the filters actually drop windows
    for i in range(0, min(n - 1, 40), 2):
        windows[i + 1] = windows[i]
    return windows


# II Test Classes
class TestFilterSimilarityToTest:
    """Test the backend ``filter_similarity_to_test`` helper."""

    def test_valid_filter_similarity_to_test(self):
        kept, mask = filter_similarity_to_test(CAND_WINDOWS, TEST_WINDOWS, 0.5)
        assert isinstance(kept, list)
        assert isinstance(mask, np.ndarray)
        assert len(mask) == len(CAND_WINDOWS)
        assert "AAAA" not in kept

    def test_valid_filter_similarity_to_test_explicit_threshold(self):
        kept_loose, _ = filter_similarity_to_test(CAND_WINDOWS, TEST_WINDOWS, 0.95)
        kept_strict, _ = filter_similarity_to_test(CAND_WINDOWS, TEST_WINDOWS, 0.0)
        assert len(kept_strict) <= len(kept_loose)

    def test_valid_no_threshold_returns_all(self):
        kept, mask = filter_similarity_to_test(CAND_WINDOWS, TEST_WINDOWS, None)
        assert kept == CAND_WINDOWS
        assert mask.all()


class TestFilterRedundancy:
    """Test the backend ``filter_redundancy`` helper."""

    def test_valid_filter_redundancy(self):
        kept, mask = filter_redundancy(["AAAA", "AAAB", "CCCC"], 0.5)
        assert kept[0] == "AAAA"
        assert "AAAB" not in kept
        assert "CCCC" in kept

    def test_valid_no_threshold_returns_all(self):
        kept, mask = filter_redundancy(CAND_WINDOWS, None)
        assert kept == CAND_WINDOWS
        assert mask.all()


class TestFilterEquivalence:
    """The vectorized filters must return output identical to the reference scalar path.

    Identity is an exact integer ratio (matches / length), so keep/drop decisions are
    bit-identical, not merely close."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    @pytest.mark.parametrize("max_similarity", [0.0, 0.5, 0.7, 0.95, 1.0])
    def test_filter_redundancy_matches_reference(self, seed, max_similarity):
        windows = _random_windows(n=300, length=12, seed=seed)
        kept, mask = filter_redundancy(windows, max_similarity)
        ref_kept, ref_mask = _ref_filter_redundancy(windows, max_similarity)
        assert kept == ref_kept
        assert np.array_equal(mask, ref_mask)

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    @pytest.mark.parametrize("max_similarity", [0.0, 0.5, 0.7, 0.95])
    def test_filter_similarity_matches_reference(self, seed, max_similarity):
        windows = _random_windows(n=300, length=12, seed=seed)
        test_windows = _random_windows(n=15, length=12, seed=seed + 100) + windows[:5]
        kept, mask = filter_similarity_to_test(windows, test_windows, max_similarity)
        ref_kept, ref_mask = _ref_filter_similarity_to_test(windows, test_windows, max_similarity)
        assert kept == ref_kept
        assert np.array_equal(mask, ref_mask)

    def test_non_latin1_falls_back_to_scalar(self):
        # Equal-length windows with a non-latin-1 character must not crash; they fall back
        # to the exact scalar path (alpha = U+03B1 is outside latin-1).
        windows = ["AAAA", "AAαA", "AAAA"]
        kept, mask = filter_redundancy(windows, 0.5)
        ref_kept, ref_mask = _ref_filter_redundancy(windows, 0.5)
        assert kept == ref_kept and np.array_equal(mask, ref_mask)
        kept, mask = filter_similarity_to_test(windows, ["AAαA"], 0.5)
        ref_kept, ref_mask = _ref_filter_similarity_to_test(windows, ["AAαA"], 0.5)
        assert kept == ref_kept and np.array_equal(mask, ref_mask)

    def test_ragged_lengths_match_reference(self):
        # Mixed-length inputs must fall back to the exact scalar path.
        windows = ["AAAA", "AAAAA", "AAAA", "CCCC", "CC"]
        kept, mask = filter_redundancy(windows, 0.5)
        ref_kept, ref_mask = _ref_filter_redundancy(windows, 0.5)
        assert kept == ref_kept and np.array_equal(mask, ref_mask)
        kept, mask = filter_similarity_to_test(["AAAA", "CCCC"], ["AAA", "AAAA"], 0.7)
        ref_kept, ref_mask = _ref_filter_similarity_to_test(["AAAA", "CCCC"], ["AAA", "AAAA"], 0.7)
        assert kept == ref_kept and np.array_equal(mask, ref_mask)


class TestWindowIdentity:
    """Test the backend ``window_identity`` helper."""

    def test_valid_window_identity_formula(self):
        # 3/4 identical
        assert window_identity("AAAA", "AAAB") == 0.75
        # All identical
        assert window_identity("ABCD", "ABCD") == 1.0
        # All different
        assert window_identity("AAAA", "BBBB") == 0.0
        # Different lengths -> 0
        assert window_identity("AAA", "AAAA") == 0.0
