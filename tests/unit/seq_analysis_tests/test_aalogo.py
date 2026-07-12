"""
This is a script to test the AALogo class initialization.
"""
import pytest
import aaanalysis as aa

from hypothesis import settings
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

aa.options["verbose"] = False


# ===========================================================================
# Test AALogo.__init__
# ===========================================================================
class TestAALogoInit:
    """Test AALogo class initialization."""

    def test_valid_logo_type(self):
        """Test valid 'logo_type' values are stored correctly."""
        for logo_type in ["probability", "weight", "counts", "information"]:
            aalogo = aa.AALogo(logo_type=logo_type)
            assert aalogo._logo_type == logo_type

    def test_default_logo_type(self):
        """Test default 'logo_type' is 'probability'."""
        aalogo = aa.AALogo()
        assert aalogo._logo_type == "probability"

    def test_invalid_logo_type(self):
        """Test invalid 'logo_type' raises ValueError."""
        for logo_type in [None, 0, "invalid", "bits", "freq", []]:
            with pytest.raises(ValueError):
                aa.AALogo(logo_type=logo_type)
