"""This is a script to test the beta-feature (experimental API) overview.

The beta set is discovered from the ``**Experimental.**`` docstring marker rather than
hardcoded, so these tests pin the properties that keep the overview honest:
- discovery never comes back empty (a changed marker text would silently empty the page);
- every discovered symbol is well-formed and carries a curated one-phrase purpose;
- the committed page matches what the generator renders (doc sync), and leads with prose.
"""
import pathlib

import pytest

import aaanalysis as aa
import aaanalysis.utils as ut
from aaanalysis import _beta

DOC_PATH = (pathlib.Path(aa.__file__).resolve().parent.parent
            / "docs/source/index/usage_principles/beta_features.rst")
# A [pro]-gated beta symbol is absent from __all__ in a base install, which renders a
# shorter page than the committed one. CI installs .[pro], so the gate holds there.
PRO_INSTALLED = "CPPStructurePlot" in aa.__all__


# --------------------------------------------------------------------------- discovery
class TestBetaDiscovery:
    def test_beta_set_is_non_empty(self):
        """A renamed/reworded marker must fail loudly instead of emptying the page."""
        assert ut.get_beta_symbols(), (
            f"no beta symbols discovered; the {_beta.BETA_MARKER!r} docstring marker text "
            "may have changed")

    def test_records_are_well_formed(self):
        for rec in ut.get_beta_symbols():
            assert rec["name"] in aa.__all__, rec["name"]
            assert isinstance(rec["purpose"], str) and rec["purpose"].endswith("."), rec
            assert rec["versionadded"] != "n/a", (
                f"{rec['name']} carries the beta marker but no '.. versionadded::'")

    def test_every_discovered_symbol_has_a_curated_purpose(self):
        """A newly marked beta class needs a one-phrase purpose in DICT_BETA_PURPOSE."""
        missing = [rec["name"] for rec in ut.get_beta_symbols()
                   if rec["name"] not in _beta.DICT_BETA_PURPOSE]
        assert not missing, f"add a purpose phrase for: {missing}"

    def test_discovered_symbols_actually_carry_the_marker(self):
        for rec in ut.get_beta_symbols():
            doc = getattr(aa, rec["name"]).__doc__ or ""
            assert _beta.BETA_MARKER in doc, rec["name"]


# ----------------------------------------------------------------------------- render
class TestRenderedPage:
    def test_page_leads_with_prose_not_a_table(self):
        body = ut.render_beta_rst().split("=============\n", 1)[1].lstrip()
        assert not body.startswith((".. list-table::", ".. toctree::")), body[:60]
        assert "without the usual deprecation cycle" in body

    def test_page_lists_every_discovered_symbol(self):
        rendered = ut.render_beta_rst()
        for rec in ut.get_beta_symbols():
            assert f":class:`~aaanalysis.{rec['name']}`" in rendered, rec["name"]


# --------------------------------------------------------------------------- doc sync
class TestDocSync:
    @pytest.mark.skipif(not PRO_INSTALLED,
                        reason="pro extra missing: pro-gated beta symbols are absent from "
                               "__all__, so the render is legitimately shorter")
    def test_committed_doc_matches_render(self):
        assert DOC_PATH.exists(), "beta_features.rst not generated"
        assert DOC_PATH.read_text() == ut.render_beta_rst(), (
            "beta_features.rst out of sync; run .github/scripts/gen_beta_features_doc.py")
