"""Live network tests for the package's online fetches.

Deselected by default (``-m "not network"``); run on demand or in the nightly so
that an upstream API / file-version change — e.g. AlphaFold DB's ``v4`` -> ``v6``
file rename, which silently broke ``fetch_alphafold`` because the URL version was
hardcoded — is caught here instead of slipping past the mocked unit tests.

Add a live test here for every method that reaches an external endpoint.
"""
import tempfile

import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.data_handling_pro._backend.struct_preproc import _alphafold as af

pytestmark = pytest.mark.network

# A small, stable, reviewed UniProt accession that is in AlphaFold DB.
KNOWN_ENTRY = "P05067"  # human amyloid precursor protein (APP)


class TestAlphaFoldFetchLive:
    """fetch_alphafold against the real AlphaFold-DB endpoint."""

    def test_resolve_urls_returns_a_current_model_url(self):
        # Pins the API contract fetch_alphafold depends on. If AlphaFold renames
        # fields or bumps the file version again, this fails loudly.
        resolved = af._af_resolve_urls(KNOWN_ENTRY, "pdb", timeout=30.0)
        assert resolved is not None, (
            "AlphaFold API returned no record for a known entry "
            f"({KNOWN_ENTRY}) — the API endpoint/shape may have changed")
        model_url, pae_url = resolved
        assert model_url.endswith(".pdb")
        assert f"AF-{KNOWN_ENTRY}-F1-model" in model_url
        assert pae_url is not None and pae_url.endswith(".json")

    def test_unknown_accession_is_soft_not_found(self):
        # A malformed accession is the soft "not in AF-DB" case, not a crash.
        assert af._af_resolve_urls("NOTAREALACCESSION", "pdb", timeout=30.0) is None

    def test_fetch_alphafold_downloads_a_real_structure(self):
        df = pd.DataFrame({"entry": [KNOWN_ENTRY], "sequence": ["MKV"], "label": [1]})
        strp = aa.StructurePreprocessor(verbose=False)
        out = tempfile.mkdtemp()
        status = strp.fetch_alphafold(df_seq=df, out_folder=out, on_failure="nan")
        assert bool(status.iloc[0]["alphafold_ok"]) is True, (
            f"fetch_alphafold failed for a known AlphaFold-DB entry ({KNOWN_ENTRY}) "
            "— the upstream file naming/version likely changed; see _af_resolve_urls")
