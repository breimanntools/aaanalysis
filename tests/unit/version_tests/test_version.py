"""This is a script to test the public aa.__version__ attribute (decision D13)."""
import re
from importlib.metadata import version as _meta_version

import aaanalysis as aa

aa.options["verbose"] = False


class TestVersion:
    """Normal cases for the exposed package version string."""

    def test_attribute_exists(self):
        assert hasattr(aa, "__version__")

    def test_is_non_empty_str(self):
        assert isinstance(aa.__version__, str)
        assert aa.__version__ != ""

    def test_matches_importlib_metadata(self):
        # Single source of truth is the installed package metadata (pyproject).
        assert aa.__version__ == _meta_version("aaanalysis")

    def test_not_the_uninstalled_fallback(self):
        # The package is installed in the test env, so the fallback must not show.
        assert aa.__version__ != "0.0.0+unknown"

    def test_looks_like_pep440(self):
        # Leading X.Y.Z, optionally followed by a pre/post/local segment.
        assert re.match(r"^\d+\.\d+\.\d+", aa.__version__)

    def test_listed_via_module_dir(self):
        assert "__version__" in dir(aa)
