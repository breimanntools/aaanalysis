"""This is a script to test the beta-feature (experimental API) overview.

The beta set is discovered from the ``**Experimental.**`` docstring marker rather than
hardcoded, across both public namespaces (``aaanalysis`` and ``aaanalysis.pipe``), so these
tests pin the properties that keep the overview honest:
- discovery never comes back empty (a changed marker text would silently empty the page);
- both namespaces are scanned, so dropping the ``aaanalysis.pipe`` pass fails loudly
  instead of silently shortening the page;
- every discovered symbol is well-formed and carries a curated one-phrase purpose;
- the committed page matches what the generator renders (doc sync), leads with prose, and
  shows the calling namespace so no reader expects a top-level ``aa.find_features``.
"""
import pathlib

import pytest

import aaanalysis as aa
import aaanalysis.pipe as ap
import aaanalysis.utils as ut
from aaanalysis import _beta

DOC_PATH = (pathlib.Path(aa.__file__).resolve().parent.parent
            / "docs/source/index/usage_principles/beta_features.rst")
# A [pro]-gated beta symbol is not discovered in a base install (absent from __all__, or a
# bare install-hint stub without the marker), which renders a shorter page than the
# committed one. CI installs .[pro], so the gate holds there.
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
            module = ap if rec["namespace"] == "aaanalysis.pipe" else aa
            assert rec["name"] in module.__all__, rec["name"]
            assert isinstance(rec["purpose"], str) and rec["purpose"].endswith("."), rec
            assert rec["versionadded"] != "n/a", (
                f"{rec['display']} carries the beta marker but no '.. versionadded::'")

    def test_every_discovered_symbol_has_a_curated_purpose(self):
        """A newly marked beta symbol needs a one-phrase purpose in DICT_BETA_PURPOSE."""
        missing = [rec["display"] for rec in ut.get_beta_symbols()
                   if rec["display"] not in _beta.DICT_BETA_PURPOSE]
        assert not missing, f"add a purpose phrase for: {missing}"

    def test_discovered_symbols_actually_carry_the_marker(self):
        for rec in ut.get_beta_symbols():
            module = ap if rec["namespace"] == "aaanalysis.pipe" else aa
            doc = getattr(module, rec["name"]).__doc__ or ""
            assert _beta.BETA_MARKER in doc, rec["display"]

    def test_top_level_classes_are_discovered(self):
        """The top-level pass must keep contributing; a class is reached through ``aa``."""
        names = [rec["name"] for rec in ut.get_beta_symbols()
                 if rec["namespace"] == "aaanalysis"]
        assert "AAPred" in names and "ReliabilityModel" in names, names

    @pytest.mark.skipif(not PRO_INSTALLED,
                        reason="pro extra missing: explain_features degrades to a stub")
    def test_pipe_symbols_are_discovered(self):
        """The second (aaanalysis.pipe) pass must keep contributing: dropping it would
        silently shorten the page instead of failing."""
        displays = [rec["display"] for rec in ut.get_beta_symbols()
                    if rec["namespace"] == "aaanalysis.pipe"]
        for expected in ["ap.find_features", "ap.predict_samples", "ap.explain_features"]:
            assert expected in displays, f"{expected} missing from {displays}"

    def test_non_beta_pipe_helpers_are_not_listed(self):
        """Discovery keys on the marker, not on the namespace: an unmarked pipeline
        (obtain_samples, plot_eval) must not be swept in."""
        displays = [rec["display"] for rec in ut.get_beta_symbols()]
        assert "ap.obtain_samples" not in displays and "ap.plot_eval" not in displays


# ----------------------------------------------------------------------------- render
class TestRenderedPage:
    def test_page_leads_with_prose_not_a_table(self):
        body = ut.render_beta_rst().split("=============\n", 1)[1].lstrip()
        assert not body.startswith((".. list-table::", ".. toctree::")), body[:60]
        assert "without the usual deprecation cycle" in body

    def test_page_lists_every_discovered_symbol(self):
        rendered = ut.render_beta_rst()
        for rec in ut.get_beta_symbols():
            assert _beta._render_tool(rec) in rendered, rec["display"]

    def test_pipe_entries_show_their_namespace(self):
        """The reader must not take a pipeline for a top-level ``aa`` symbol."""
        rendered = ut.render_beta_rst()
        for rec in ut.get_beta_symbols():
            if rec["namespace"] != "aaanalysis.pipe":
                continue
            assert f":func:`ap.{rec['name']} <aaanalysis.pipe.{rec['name']}>`" in rendered
        assert "import aaanalysis.pipe as ap" in rendered

    def test_top_level_classes_render_unprefixed(self):
        rendered = ut.render_beta_rst()
        assert ":class:`~aaanalysis.AAPred`" in rendered
        assert "ap.AAPred" not in rendered


# --------------------------------------------------------------------------- doc sync
class TestDocSync:
    @pytest.mark.skipif(not PRO_INSTALLED,
                        reason="pro extra missing: pro-gated beta symbols are not "
                               "discovered, so the render is legitimately shorter")
    def test_committed_doc_matches_render(self):
        assert DOC_PATH.exists(), "beta_features.rst not generated"
        assert DOC_PATH.read_text() == ut.render_beta_rst(), (
            "beta_features.rst out of sync; run .github/scripts/gen_beta_features_doc.py")
