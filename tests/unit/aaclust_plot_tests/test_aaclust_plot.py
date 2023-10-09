"""
This is a script for testing the aa.AAclustPlot class.
"""
from hypothesis import given
import hypothesis.strategies as st
from sklearn.decomposition import PCA
import pytest

import aaanalysis as aa


class TestAAclustPlot:
    """Test AAclustPlot class"""

    # Positive Test Cases
    @given(model_kwargs=st.dictionaries(keys=st.just("n_components"), values=st.integers(1, 5)))
    def test_positive_model_kwargs(self, model_kwargs):
        """Test positive cases for model_kwargs parameter."""
        aac_plot = aa.AAclustPlot(model_class=PCA, model_kwargs=model_kwargs)
        assert aac_plot.model_kwargs == model_kwargs
        assert aac_plot.model_class == PCA

    # Negative Test Cases
    def test_invalid_key_model_kwargs(self, model_kwargs):
        """Test invalid keys for model_kwargs parameter."""
        model_kwargs = {"Wrong key": 1}
        with pytest.raises(ValueError):  # Adjust depending on real check functions
            aac_plot = aa.AAclustPlot(model_class=PCA, model_kwargs=model_kwargs)


class TestAAclustPlotComplex:
    """Test AAclustPlot class for complex cases"""

    # Positive Test Cases
    @given(model_kwargs=st.dictionaries(keys=st.just("n_components"), values=st.integers(1, 5)))
    def test_complex_positive_model_kwargs(self, model_kwargs):
        """Test complex positive cases involving multiple parameters."""
        aac_plot = aa.AAclustPlot(model_class=PCA, model_kwargs=model_kwargs)


