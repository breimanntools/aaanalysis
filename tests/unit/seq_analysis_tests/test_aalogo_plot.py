"""
This is a script to test the AAlogoPlot class initialization.
"""
import pytest
import aaanalysis as aa

# Set default deadline from 200 to 20000
from hypothesis import settings
settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")

aa.options["verbose"] = False


# ===========================================================================
# I Test AAlogoPlot.__init__
# ===========================================================================
class TestAAlogoPlotInit:
    """Test AAlogoPlot class initialization."""

    # Positive tests
    def test_valid_logo_type(self):
        """Test valid 'logo_type' values set correct y-axis label."""
        dict_expected = {"probability": "Probability [%]",
                         "weight": "Weight",
                         "counts": "Counts",
                         "information": "Bits"}
        for logo_type, expected_label in dict_expected.items():
            aal_plot = aa.AAlogoPlot(logo_type=logo_type)
            assert aal_plot._y_label == expected_label

    def test_valid_jmd_n_len(self):
        """Test valid 'jmd_n_len' values."""
        for jmd_n_len in [0, 1, 5, 10, 20]:
            aal_plot = aa.AAlogoPlot(jmd_n_len=jmd_n_len)
            assert aal_plot._jmd_n_len == jmd_n_len

    def test_valid_jmd_c_len(self):
        """Test valid 'jmd_c_len' values."""
        for jmd_c_len in [0, 1, 5, 10, 20]:
            aal_plot = aa.AAlogoPlot(jmd_c_len=jmd_c_len)
            assert aal_plot._jmd_c_len == jmd_c_len

    def test_valid_verbose(self):
        """Test valid 'verbose' values."""
        for verbose in [True, False]:
            aal_plot = aa.AAlogoPlot(verbose=verbose)
            assert isinstance(aal_plot, aa.AAlogoPlot)

    def test_default_values(self):
        """Test that default parameters are set correctly."""
        aal_plot = aa.AAlogoPlot()
        assert aal_plot._y_label == "Probability [%]"
        assert aal_plot._jmd_n_len == 10
        assert aal_plot._jmd_c_len == 10

    # Negative tests
    def test_invalid_logo_type(self):
        """Test invalid 'logo_type' raises ValueError."""
        for logo_type in [None, 0, "bits", "freq", "invalid", []]:
            with pytest.raises(ValueError):
                aa.AAlogoPlot(logo_type=logo_type)

    def test_invalid_jmd_n_len(self):
        """Test invalid 'jmd_n_len' raises ValueError."""
        for jmd_n_len in [-1, -10, 1.5, "invalid"]:
            with pytest.raises(ValueError):
                aa.AAlogoPlot(jmd_n_len=jmd_n_len)

    def test_invalid_jmd_c_len(self):
        """Test invalid 'jmd_c_len' raises ValueError."""
        #for jmd_c_len in [-1, -10, 1.5, "invalid"]:
        #    with pytest.raises(ValueError):
        #        aa.AAlogoPlot(jmd_c_len=jmd_c_len)

    def test_invalid_verbose(self):
        """Test invalid 'verbose' raises ValueError."""
        aa.options["verbose"] = "off"
        for verbose in [None, 1, "True", []]:
            with pytest.raises(ValueError):
                aa.AAlogoPlot(verbose=verbose)

