"""
This is a script for testing the aa.load_features function.
"""
import aaanalysis as aa
from pandas import DataFrame
import pytest


class TestLoadFeatures:
    """Test load_scales function"""

    # Basic positive tests
    def test_load_features_default(self):
        """Test the default parameters."""
        df = aa.load_features()
        assert isinstance(df, DataFrame)
        df = aa.load_features(name="DOM_GSEC")
        assert isinstance(df, DataFrame)

    # Negative tests
    def test_load_scales_invalid_name(self):
        """Test with an invalid dataset name."""
        with pytest.raises(ValueError):
            aa.load_features(name="invalid_name")

    def test_empty_name(self):
        """Test with an empty string as dataset name."""
        with pytest.raises(ValueError):
            aa.load_features(name="")

