"""
This is a script for testing the aa.display_df function.
"""
from hypothesis import given, settings
from hypothesis import strategies as st
import pandas as pd
import pytest
import aaanalysis as aa

@pytest.fixture(scope="module")
def sample_dataframe():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })

class TestDisplayDf:
    """Test display_df function for individual parameters"""

    def test_valid_dataframe(self):
        # Test with valid DataFrame
        df = aa.load_scales()
        aa.display_df(df)

    @settings(max_examples=10)
    @given(max_width_pct=st.integers(min_value=1, max_value=100))
    def test_max_width_pct(self, max_width_pct, sample_dataframe):
        aa.display_df(sample_dataframe, max_width_pct=max_width_pct)

    @settings(max_examples=10)
    @given(max_height=st.integers(min_value=1))
    def test_max_height(self, max_height, sample_dataframe):
        aa.display_df(sample_dataframe, max_height=max_height)

    @settings(max_examples=10)
    @given(char_limit=st.integers(min_value=1))
    def test_char_limit(self, char_limit, sample_dataframe):
        aa.display_df(sample_dataframe, char_limit=char_limit)

    @settings(max_examples=10)
    @given(n_rows=st.integers(min_value=1))
    def test_n_rows(self, n_rows, sample_dataframe):
        if n_rows < len(sample_dataframe):
            aa.display_df(sample_dataframe, n_rows=n_rows)

    @settings(max_examples=10)
    @given(n_cols=st.integers(min_value=1))
    def test_n_cols(self, n_cols, sample_dataframe):
        if n_cols < len(list(sample_dataframe)):
            aa.display_df(sample_dataframe, n_cols=n_cols)

    def test_row_to_show(self, sample_dataframe):
        for row in list(sample_dataframe.T):
            aa.display_df(sample_dataframe, row_to_show=row)

    def test_col_to_show(self, sample_dataframe):
        for col in list(sample_dataframe):
            aa.display_df(sample_dataframe, col_to_show=col)

    def test_show_shape(self, sample_dataframe):
        for show_shape in [True, False]:
            aa.display_df(sample_dataframe, show_shape=show_shape)

    # Negative Tests
    @pytest.mark.parametrize("invalid_value", [-1, 0, 101, 'invalid'])
    def test_invalid_max_width_pct(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, max_width_pct=invalid_value)

    @pytest.mark.parametrize("invalid_value", [-1, 0, 'invalid'])
    def test_invalid_max_height(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, max_height=invalid_value)

    @pytest.mark.parametrize("invalid_value", [-1, 0, 'invalid'])
    def test_invalid_char_limit(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, char_limit=invalid_value)

    @pytest.mark.parametrize("invalid_value", [-100, 'invalid'])
    def test_invalid_n_rows(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, n_rows=invalid_value)

    @pytest.mark.parametrize("invalid_value", [-100, 'invalid'])
    def test_invalid_n_cols(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, n_cols=invalid_value)

    @pytest.mark.parametrize("invalid_value", [-100, "0", 5.5])
    def test_invalid_row_to_show(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, row_to_show=invalid_value)

    @pytest.mark.parametrize("invalid_value", [-100, "0", 5.5])
    def test_invalid_col_to_show(self, invalid_value, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, col_to_show=invalid_value)

class TestDisplayDfComplex:
    """Test display_df function with complex scenarios"""

    @settings(max_examples=5)
    @given(
        max_width_pct=st.integers(min_value=1, max_value=100),
        max_height=st.integers(min_value=1),
        char_limit=st.integers(min_value=1),
        show_shape=st.booleans(),
        n_rows=st.integers(min_value=1),
        n_cols=st.integers(min_value=1),
    )
    def test_valid_combinations(self, max_width_pct, max_height, char_limit, show_shape, n_rows, n_cols, sample_dataframe):
        if n_cols < len(list(sample_dataframe)) and n_rows < len(sample_dataframe):
            aa.display_df(sample_dataframe, max_width_pct, max_height, char_limit, show_shape, n_rows, n_cols)

    @settings(max_examples=5)
    @given(
        max_width_pct=st.integers(min_value=101),
        max_height=st.integers(max_value=0),
        char_limit=st.integers(max_value=0),
        n_rows=st.integers(max_value=0),
        n_cols=st.integers(max_value=0)
    )
    def test_invalid_combinations(self, max_width_pct, max_height, char_limit, n_rows, n_cols, sample_dataframe):
        with pytest.raises(ValueError):
            aa.display_df(sample_dataframe, max_width_pct, max_height, char_limit, n_rows=n_rows, n_cols=n_cols)
