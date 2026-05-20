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

settings.register_profile("ci", deadline=400)
settings.load_profile("ci")


# I Helper Functions
TEST_WINDOWS = ["AAAA", "CCCC", "GGGG"]
CAND_WINDOWS = ["AAAA", "AAAB", "DEFG", "CCCD", "WWWW"]


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
