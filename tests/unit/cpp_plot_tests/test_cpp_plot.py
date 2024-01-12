"""
This is a script for testing the CPPPlot class.
"""
import pytest
from hypothesis import given, settings, strategies as st
import aaanalysis as aa
from pandas.testing import assert_frame_equal


class TestCPPPlot:
    """Positive Test cases for CPPPlot class."""

    # Positive tests
    def test_df_scales(self):
        """Positive test for df_scales parameter."""
        for i in range(1, 60):
            df_scales = aa.load_scales(top60_n=i)
            cpp_plot = aa.CPPPlot(df_scales=df_scales)
            assert_frame_equal(df_scales, cpp_plot._df_scales)

    def test_df_cat(self):
        """Positive test for df_cat parameter."""
        df_cat = aa.load_scales(name="scales_cat")
        for i in range(1, 60):
            df_scales = aa.load_scales(top60_n=i)
            scales = list(df_scales)
            _df_cat = df_cat[df_cat["scale_id"].isin(scales)]
            cpp_plot = aa.CPPPlot(df_scales=df_scales, df_cat=_df_cat)
            assert_frame_equal(_df_cat, cpp_plot._df_cat)

    @settings(max_examples=10, deadline=1000)
    @given(jmd_n_len=st.integers(min_value=0))
    def test_jmd_n_len(self, jmd_n_len):
        """Positive test for jmd_n_len parameter."""
        cpp_plot = aa.CPPPlot(jmd_n_len=jmd_n_len)
        assert cpp_plot._jmd_n_len == jmd_n_len

    @settings(max_examples=10, deadline=1000)
    @given(jmd_c_len=st.integers(min_value=0))
    def test_jmd_c_len(self, jmd_c_len):
        """Positive test for jmd_c_len parameter."""
        cpp_plot = aa.CPPPlot(jmd_c_len=jmd_c_len)
        assert cpp_plot._jmd_c_len == jmd_c_len

    def test_accept_gaps(self):
        """Positive test for accept_gaps parameter."""
        for accept_gaps in [True, False]:
            cpp_plot = aa.CPPPlot(accept_gaps=accept_gaps)
            assert cpp_plot._accept_gaps is accept_gaps

    def test_verbose(self):
        """Positive test for verbose parameter."""
        aa.options["verbose"] = "off"
        for verbose in [True, False]:
            cpp_plot = aa.CPPPlot(verbose=verbose)
            assert cpp_plot._verbose is verbose

    # Negative Tests
    def test_invalid_df_scales(self):
        """Negative test for df_scales parameter with invalid DataFrame structure."""
        with pytest.raises(ValueError):
            aa.CPPPlot(df_scales="str")
        with pytest.raises(ValueError):
            aa.CPPPlot(df_scales=2)
        with pytest.raises(ValueError):
            df_scales = aa.load_scales()
            df_scales.columns = ["Invalid"] * len(list(df_scales))
            aa.CPPPlot(df_scales=df_scales)
        with pytest.raises(ValueError):
            df_scales = aa.load_scales()
            df_scales.columns = ["Invalid", "Invalid"] + list(df_scales)[2:]
            aa.CPPPlot(df_scales=df_scales)


    def test_invalid_df_cat(self):
        """Positive test for df_cat parameter."""
        df_cat = aa.load_scales(name="scales_cat")
        for i in range(1, 60):
            df_scales = aa.load_scales(top60_n=i)
            scales = list(df_scales)
            n = len(scales)
            _df_cat = df_cat[df_cat["scale_id"].isin(scales[0:n-1])]
            with pytest.raises(ValueError):
                cpp_plot = aa.CPPPlot(df_scales=df_scales, df_cat=_df_cat)

    def test_invalid_jmd_n_len(self):
        """Negative test for jmd_n_len parameter with invalid values."""
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_n_len=-1)
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_n_len=None)
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_n_len="invalid")
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_n_len=[0, 2])

    # Test for jmd_c_len parameter with invalid values
    def test_invalid_jmd_c_len(self):
        """Negative test for jmd_c_len parameter with invalid values."""
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_c_len=-1)
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_c_len=None)
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_c_len="invalid")
        with pytest.raises(ValueError):
            aa.CPPPlot(jmd_c_len=[0, 2])


    def test_invalid_accept_gaps(self):
        """Positive test for accept_gaps parameter."""
        for accept_gaps in [1, "invalid", [False, True], ""]:
            with pytest.raises(ValueError):
                cpp_plot = aa.CPPPlot(accept_gaps=accept_gaps)

    def test_invalid_verbose(self):
        """Positive test for verbose parameter."""
        for verbose in [1, "invalid", [False, True], ""]:
            with pytest.raises(ValueError):
                cpp_plot = aa.CPPPlot(verbose=verbose)

class TestCPPPlotComplex:

    def test_complex_positive(self):
        """Complex positive test involving multiple parameters."""
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")
        jmd_n_len = 5
        jmd_c_len = 5
        accept_gaps = True
        verbose = True
        cpp_plot = aa.CPPPlot(df_scales=df_scales, df_cat=df_cat, jmd_n_len=jmd_n_len,
                              jmd_c_len=jmd_c_len, accept_gaps=accept_gaps, verbose=verbose)
        # Assertions to verify that parameters are correctly set
        assert_frame_equal(df_scales, cpp_plot._df_scales)
        assert_frame_equal(df_cat, cpp_plot._df_cat)
        assert cpp_plot._jmd_n_len == jmd_n_len
        assert cpp_plot._jmd_c_len == jmd_c_len
        assert cpp_plot._accept_gaps is accept_gaps
        assert cpp_plot._verbose is verbose

    def test_complex_negative(self):
        """Complex negative test with invalid parameter combinations."""
        df_scales = aa.load_scales()
        df_cat = aa.load_scales(name="scales_cat")  # Assume this is valid
        jmd_n_len = -1  # Invalid
        jmd_c_len = -1  # Invalid
        accept_gaps = True  # Valid
        verbose = "not a boolean"  # Invalid
        with pytest.raises(ValueError):
            aa.CPPPlot(df_scales=df_scales, df_cat=df_cat, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                       accept_gaps=accept_gaps, verbose=verbose)