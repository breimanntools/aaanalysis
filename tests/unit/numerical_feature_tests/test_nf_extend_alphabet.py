"""This is a script to test the NumericalFeature().extend_alphabet() method."""
from hypothesis import given, settings, strategies as st
import pandas as pd
import pytest
import numpy as np
from pandas.testing import assert_frame_equal

# Import the target function
import aaanalysis as aa

# Helper Functions and Constants
VALID_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VALID_VALUE_TYPES = ["min", "mean", "median", "max"]
INVALID_VALUE_TYPES = ["average", "total", 123, True, None, []]

def generate_random_df():
    """Generate a random DataFrame representing amino acid scales."""
    np.random.seed(0)  # for reproducibility
    data = np.random.rand(len(VALID_LETTERS), 5)
    df = pd.DataFrame(data, index=VALID_LETTERS, columns=[f"scale_{i}" for i in range(5)])
    return df

def check_already_in(df_scales=None, letter_new=None):
    alphabet = df_scales.index.to_list()
    return letter_new in alphabet

# Test Classes
class TestExtendAlphabet:
    """Test class for the 'extend_alphabet' function."""

    # Positive tests
    @settings(max_examples=10, deadline=1000)
    @given(df_scales=st.just(generate_random_df()),
           letter_new=st.sampled_from(VALID_LETTERS),
           value_type=st.sampled_from(VALID_VALUE_TYPES))
    def test_valid_inputs(self, df_scales, letter_new, value_type):
        """Positive test with valid inputs."""
        original_len = len(df_scales)
        nf = aa.NumericalFeature()
        if not check_already_in(df_scales=df_scales, letter_new=letter_new):
            result = nf.extend_alphabet(df_scales, letter_new, value_type)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == original_len + 1
            assert letter_new in result.index

    def test_change_default_scales(self):
        df_scales = aa.load_scales()
        nf = aa.NumericalFeature()
        df_scales_new = nf.extend_alphabet(df_scales=df_scales, new_letter="X")
        aa.options["df_scales"] = df_scales_new
        df_scales_old = aa.load_scales()
        cpp_plot = aa.CPPPlot()
        df_scales_new_default = cpp_plot._df_scales
        assert_frame_equal(df_scales, df_scales_old)
        assert_frame_equal(df_scales_new, df_scales_new_default)

    # Negative tests
    @settings(max_examples=10, deadline=1000)
    @given(df_scales=st.just(generate_random_df()),
           letter_new=st.sampled_from(VALID_LETTERS),
           value_type=st.sampled_from(INVALID_VALUE_TYPES))
    def test_invalid_value_type(self, df_scales, letter_new, value_type):
        """Negative test for invalid 'value_type'."""
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.extend_alphabet(df_scales, letter_new, value_type)

    @settings(max_examples=10, deadline=1000)
    @given(df_scales=st.just(generate_random_df()),
           letter_new=st.text(),
           value_type=st.sampled_from(VALID_VALUE_TYPES))
    def test_invalid_letter_new(self, df_scales, letter_new, value_type):
        """Negative test for invalid 'letter_new'."""
        nf = aa.NumericalFeature()
        if check_already_in(df_scales=df_scales, letter_new=letter_new):
            with pytest.raises(ValueError):
                nf.extend_alphabet(df_scales, letter_new, value_type)
        with pytest.raises(ValueError):
                nf.extend_alphabet(df_scales, pd.DataFrame, value_type)
        with pytest.raises(ValueError):
                nf.extend_alphabet(df_scales, None, value_type)


    @settings(max_examples=10, deadline=1000)
    @given(df_scales=st.lists(st.floats(), min_size=1),
           letter_new=st.sampled_from(VALID_LETTERS),
           value_type=st.sampled_from(VALID_VALUE_TYPES))
    def test_invalid_df_scales_structure(self, df_scales, letter_new, value_type):
        """Negative test for invalid 'df_scales' structure."""
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.extend_alphabet(df_scales, letter_new, value_type)


class TestExtendAlphabetComplex:
    """Complex tests for the 'extend_alphabet' function."""

    @settings(max_examples=10, deadline=1000)
    @given(df_scales=st.just(generate_random_df()),
           letter_new=st.sampled_from(VALID_LETTERS),
           value_type=st.sampled_from(VALID_VALUE_TYPES))
    def test_valid_combinations(self, df_scales, letter_new, value_type):
        """Positive test with valid combinations of parameters."""
        nf = aa.NumericalFeature()
        if not check_already_in(df_scales=df_scales, letter_new=letter_new):
            result = nf.extend_alphabet(df_scales, letter_new, value_type)
            assert isinstance(result, pd.DataFrame)
            assert letter_new in result.index


    @settings(max_examples=10, deadline=1000)
    @given(df_scales=st.just(generate_random_df()),
           letter_new=st.sampled_from(VALID_LETTERS),
           value_type=st.sampled_from(INVALID_VALUE_TYPES))
    def test_invalid_combinations(self, df_scales, letter_new, value_type):
        """Negative test with invalid combinations of parameters."""
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError):
            nf.extend_alphabet(df_scales, letter_new, value_type)

