"""This is a script to test AAWindowSampler() construction."""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# I Helper Functions


# II Test Classes
class TestAAWindowSamplerInit:
    """Test AAWindowSampler class initialization."""

    # Positive tests
    @settings(max_examples=10, deadline=None)
    @given(verbose=some.booleans())
    def test_valid_verbose(self, verbose):
        aa.options["verbose"] = "off"
        aaws = aa.AAWindowSampler(verbose=verbose)
        assert aaws.verbose == verbose

    @settings(max_examples=10, deadline=None)
    @given(rs=some.integers(min_value=0, max_value=10000))
    def test_valid_random_state(self, rs):
        aaws = aa.AAWindowSampler(random_state=rs)
        assert aaws._random_state == rs

    @settings(max_examples=10, deadline=None)
    @given(thr=some.floats(min_value=0.0, max_value=1.0))
    def test_valid_max_similarity_to_test(self, thr):
        aaws = aa.AAWindowSampler(max_similarity_to_test=thr)
        assert aaws._max_similarity_to_test == thr

    @settings(max_examples=10, deadline=None)
    @given(thr=some.floats(min_value=0.0, max_value=1.0))
    def test_valid_max_similarity_within_ref(self, thr):
        aaws = aa.AAWindowSampler(max_similarity_within_ref=thr)
        assert aaws._max_similarity_within_ref == thr

    @settings(max_examples=5, deadline=None)
    @given(flag=some.booleans())
    def test_valid_filter_iteratively(self, flag):
        aaws = aa.AAWindowSampler(filter_iteratively=flag)
        assert aaws._filter_iteratively == flag

    @settings(max_examples=10, deadline=None)
    @given(n=some.integers(min_value=1, max_value=50))
    def test_valid_max_sampling_attempts(self, n):
        aaws = aa.AAWindowSampler(max_sampling_attempts=n)
        assert aaws._max_sampling_attempts == n

    # Negative tests
    def test_invalid_verbose(self):
        # `check_verbose` defers to the global option when it is not "off";
        # set "off" here so the per-call argument is the one being validated.
        aa.options["verbose"] = "off"
        for invalid in [None, 1, "true", []]:
            with pytest.raises(ValueError):
                aa.AAWindowSampler(verbose=invalid)

    def test_invalid_random_state(self):
        for invalid in [-1, "1", 1.5, []]:
            with pytest.raises(ValueError):
                aa.AAWindowSampler(random_state=invalid)

    def test_invalid_max_similarity_to_test(self):
        for invalid in [-0.1, 1.1, "0.5", []]:
            with pytest.raises(ValueError):
                aa.AAWindowSampler(max_similarity_to_test=invalid)

    def test_invalid_max_similarity_within_ref(self):
        for invalid in [-0.1, 1.1, "0.5", []]:
            with pytest.raises(ValueError):
                aa.AAWindowSampler(max_similarity_within_ref=invalid)

    def test_invalid_filter_iteratively(self):
        for invalid in [None, 1, "true", []]:
            with pytest.raises(ValueError):
                aa.AAWindowSampler(filter_iteratively=invalid)

    def test_invalid_max_sampling_attempts(self):
        for invalid in [0, -1, None, "5", 1.5, []]:
            with pytest.raises(ValueError):
                aa.AAWindowSampler(max_sampling_attempts=invalid)
