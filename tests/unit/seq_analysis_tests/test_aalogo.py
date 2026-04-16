"""
This is a script to test the AAlogo class initialization.
"""
import pytest
import aaanalysis as aa

from hypothesis import settings
settings.register_profile("ci", deadline=20000)
settings.load_profile("ci")

aa.options["verbose"] = False


# ===========================================================================
# Test AAlogo.__init__
# ===========================================================================
class TestAAlogoInit:
    """Test AAlogo class initialization."""

    def test_valid_logo_type(self):
        """Test valid 'logo_type' values are stored correctly."""
        for logo_type in ["probability", "weight", "counts", "information"]:
            aalogo = aa.AAlogo(logo_type=logo_type)
            assert aalogo._logo_type == logo_type

    def test_default_logo_type(self):
        """Test default 'logo_type' is 'probability'."""
        aalogo = aa.AAlogo()
        assert aalogo._logo_type == "probability"

    def test_invalid_logo_type(self):
        """Test invalid 'logo_type' raises ValueError."""
        for logo_type in [None, 0, "invalid", "bits", "freq", []]:
            with pytest.raises(ValueError):
                aa.AAlogo(logo_type=logo_type)
