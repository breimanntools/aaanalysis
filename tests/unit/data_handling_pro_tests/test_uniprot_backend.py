"""This is a script to test the UniProt fetch/map backend in
``data_handling_pro/_backend/annot_preproc/_uniprot.py``
(fetch_uniprot_json / fetch_and_map / map_record_to_rows / _feature_key /
_get_sequence and the evidence/classification helpers).

The network endpoint is never hit by the suite; the existing frontend test calls
``map_record_to_rows`` on hand-built JSON dicts. Here we additionally cover the
``requests``-backed ``fetch_uniprot_json`` / ``fetch_and_map`` paths by mocking
``requests.get`` (no network), plus the feature-classification and skip branches.
"""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

import aaanalysis.utils as ut
from aaanalysis.data_handling_pro._backend.annot_preproc import _uniprot as up

MODULE = "aaanalysis.data_handling_pro._backend.annot_preproc._uniprot"

SEQ = "ACDEFGHIKLMNPQRSTVWY"


# I Helper Functions
def _record(features, seq=SEQ):
    return {"sequence": {"value": seq}, "features": features}


def _feat(ftype, start, end, desc="", eco="ECO:0000269"):
    ev = [{"evidenceCode": eco}] if eco else []
    return {
        "type": ftype,
        "description": desc,
        "location": {"start": {"value": start}, "end": {"value": end}},
        "evidences": ev,
    }


def _resp(status=200, payload=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = payload or {}
    return m


# II Test Classes
class TestFetchUniprotJson:
    """fetch_uniprot_json: success, non-200, transport error."""

    def test_valid_returns_json(self):
        with patch(f"{MODULE}.requests.get",
                   return_value=_resp(200, {"primaryAccession": "P1"})):
            out = up.fetch_uniprot_json("P1")
        assert out["primaryAccession"] == "P1"

    def test_valid_passes_timeout(self):
        with patch(f"{MODULE}.requests.get",
                   return_value=_resp(200, {})) as mg:
            up.fetch_uniprot_json("P1", timeout=12.0)
        assert mg.call_args.kwargs["timeout"] == 12.0

    def test_invalid_non_200(self):
        with patch(f"{MODULE}.requests.get", return_value=_resp(404, {})):
            with pytest.raises(RuntimeError, match="HTTP 404"):
                up.fetch_uniprot_json("MISSING")

    def test_invalid_transport_error(self):
        with patch(f"{MODULE}.requests.get",
                   side_effect=requests.RequestException("boom")):
            with pytest.raises(RuntimeError, match="failed"):
                up.fetch_uniprot_json("P1")


class TestGetSequence:
    """_get_sequence: present vs missing."""

    def test_valid_sequence(self):
        assert up._get_sequence(_record([])) == SEQ

    def test_invalid_missing_sequence(self):
        with pytest.raises(RuntimeError, match="sequence.value"):
            up._get_sequence({"sequence": {}})

    def test_invalid_empty_sequence(self):
        with pytest.raises(RuntimeError, match="sequence.value"):
            up._get_sequence({"sequence": {"value": ""}})


class TestFeatureKey:
    """_feature_key: every routing branch + drop."""

    def test_valid_modres_phospho(self):
        assert up._feature_key("Modified residue", "Phosphoserine") == "phospho"

    def test_valid_modres_other(self):
        assert up._feature_key("Modified residue", "N6-acetyllysine") == "mod_res_other"

    def test_valid_glyco_n(self):
        assert up._feature_key("Glycosylation", "N-linked (GlcNAc)") == "glyco_n"

    def test_valid_glyco_o(self):
        assert up._feature_key("Glycosylation", "O-linked (GalNAc)") == "glyco_o"

    def test_valid_lipid(self):
        assert up._feature_key("Lipidation", "N-myristoyl glycine") == "lipid"

    def test_valid_binding(self):
        assert up._feature_key("Binding site", "") == "binding"

    def test_valid_act_site(self):
        assert up._feature_key("Active site", "") == "act_site"

    def test_valid_dna_bind(self):
        assert up._feature_key("DNA binding", "") == "dna_bind"

    def test_valid_site_cleavage(self):
        assert up._feature_key("Site", "Cleavage; by thrombin") == "cleavage_site"

    def test_valid_site_noncleavage_dropped(self):
        assert up._feature_key("Site", "Interaction with X") is None

    def test_valid_unknown_type_dropped(self):
        assert up._feature_key("Chain", "whatever") is None


class TestMapRecordToRows:
    """map_record_to_rows: bond / processing / range + skip branches."""

    def test_valid_modres_single_row(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Modified residue", 3, 3, "Phosphoserine")]),
            None, None)
        assert len(rows) == 1
        assert rows[0][4] == "phospho"

    def test_valid_disulfide_two_endpoints(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Disulfide bond", 2, 8)]), None, None)
        assert len(rows) == 2
        # shared bond_id (last column)
        assert rows[0][-1] == rows[1][-1]
        assert rows[0][-1] is not None

    def test_valid_processing_uses_span_end(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Signal", 1, 5)]), None, None)
        assert len(rows) == 1
        assert rows[0][1] == 5  # END anchor

    def test_valid_range_one_row_per_residue(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Binding site", 4, 6)]), None, None)
        assert len(rows) == 3  # positions 4,5,6

    def test_valid_skip_missing_location(self):
        feat = {"type": "Modified residue", "description": "Phosphoserine",
                "location": {"end": {"value": 5}}, "evidences": []}  # no start
        rows = up.map_record_to_rows("P1", _record([feat]), None, None)
        assert rows == []

    def test_valid_evidence_filter_drops(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Modified residue", 3, 3, "Phosphoserine",
                                 eco="ECO:0000250")]),
            None, ["ECO:0000269"])  # feature ECO not in allow-set
        assert rows == []

    def test_valid_allowed_features_filters_processing(self):
        # Signal -> signal_cleavage, but only 'phospho' is allowed -> dropped.
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Signal", 1, 5)]), ["phospho"], None)
        assert rows == []

    def test_valid_allowed_features_filters_bond(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Disulfide bond", 2, 8)]), ["phospho"], None)
        assert rows == []

    def test_valid_site_noncleavage_skipped(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Site", 3, 3, "Interaction")]), None, None)
        assert rows == []

    def test_invalid_missing_sequence(self):
        with pytest.raises(RuntimeError, match="sequence.value"):
            up.map_record_to_rows("P1", {"features": []}, None, None)


class TestFetchAndMap:
    """fetch_and_map: loops entries, concatenates, honors verbose."""

    def test_valid_single_entry(self):
        rec = _record([_feat("Modified residue", 3, 3, "Phosphoserine")])
        with patch(f"{MODULE}.fetch_uniprot_json", return_value=rec):
            df = up.fetch_and_map(["P1"], None, None, verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(ut.COLS_ANNOT)
        assert len(df) == 1

    def test_valid_multiple_entries_concatenated(self):
        rec = _record([_feat("Binding site", 4, 6)])
        with patch(f"{MODULE}.fetch_uniprot_json", return_value=rec):
            df = up.fetch_and_map(["P1", "P2"], None, None, verbose=False)
        assert len(df) == 6  # 3 residues x 2 entries

    def test_valid_verbose_prints(self, capsys):
        rec = _record([])
        with patch(f"{MODULE}.fetch_uniprot_json", return_value=rec):
            up.fetch_and_map(["P9"], None, None, verbose=True)
        assert "P9" in capsys.readouterr().out

    def test_valid_empty_entries_empty_df(self):
        with patch(f"{MODULE}.fetch_uniprot_json", return_value=_record([])):
            df = up.fetch_and_map([], None, None, verbose=False)
        assert df.empty
        assert list(df.columns) == list(ut.COLS_ANNOT)

    def test_invalid_propagates_network_error(self):
        with patch(f"{MODULE}.fetch_uniprot_json",
                   side_effect=RuntimeError("net down")):
            with pytest.raises(RuntimeError, match="net down"):
                up.fetch_and_map(["P1"], None, None, verbose=False)


class TestUniprotComplex:
    """Cross-cutting combinations across the mapping pipeline."""

    def test_complex_mixed_feature_types(self):
        rec = _record([
            _feat("Modified residue", 3, 3, "Phosphoserine"),
            _feat("Disulfide bond", 2, 8),
            _feat("Signal", 1, 5),
            _feat("Binding site", 10, 11),
        ])
        rows = up.map_record_to_rows("P1", rec, None, None)
        # 1 phospho + 2 disulfide + 1 signal + 2 binding = 6
        assert len(rows) == 6

    def test_complex_aa_from_sequence(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Modified residue", 1, 1, "Phosphoserine")]),
            None, None)
        assert rows[0][3] == SEQ[0]  # aa at pos 1

    def test_complex_out_of_bounds_pos_empty_aa(self):
        rec = _record([_feat("Modified residue", 999, 999, "Phosphoserine")])
        rows = up.map_record_to_rows("P1", rec, None, None)
        assert rows[0][3] == ""  # out-of-range -> empty aa

    def test_complex_evidence_none_disables_filter(self):
        rows = up.map_record_to_rows(
            "P1", _record([_feat("Modified residue", 3, 3, "Phosphoserine",
                                 eco="ECO:9999999")]),
            None, None)  # evidence filter off
        assert len(rows) == 1

    def test_complex_fetch_and_map_end_to_end(self):
        rec1 = _record([_feat("Modified residue", 3, 3, "Phosphoserine")])
        rec2 = _record([_feat("Disulfide bond", 2, 8)])
        recs = iter([rec1, rec2])
        with patch(f"{MODULE}.fetch_uniprot_json",
                   side_effect=lambda e, timeout=30.0: next(recs)):
            df = up.fetch_and_map(["P1", "P2"], None, None, verbose=False)
        assert len(df) == 3  # 1 + 2
