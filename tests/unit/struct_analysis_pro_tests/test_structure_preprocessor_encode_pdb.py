"""This is a script to test StructurePreprocessor.encode_pdb()."""
import shutil
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=2000)
settings.load_profile("ci")

PDB_FIXTURES = Path(__file__).resolve().parents[3] / \
    "aaanalysis" / "_data" / "pdb_test"

msms_required = pytest.mark.skipif(
    shutil.which("msms") is None,
    reason="msms binary not on PATH")


# I Helper Functions
def _df_one():
    return pd.DataFrame({
        "entry": ["P1"],
        "sequence": ["ACDEFGHIKLMNPQRS"],
    })


def _df_two():
    return pd.DataFrame({
        "entry": ["P1", "P2"],
        "sequence": ["ACDEFGHIKLMNPQRS", "VLIMKRSTGADE"],
    })


# II Test Classes
class TestStpEncodePdb:
    """Single-parameter normal + invalid cases for encode_pdb."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_df_seq_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=None, pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"])

    def test_invalid_pdb_folder_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=None,
                           features=["bfactor"])

    def test_invalid_pdb_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_pdb(df_seq=_df_one(),
                           pdb_folder="/__nope__/__here__",
                           features=["bfactor"])

    def test_invalid_features_empty(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=[])

    def test_invalid_features_unknown(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=["mystery"])

    def test_invalid_features_wrong_method(self):
        # ss3 belongs to encode_dssp
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=["ss3"])

    def test_invalid_on_failure(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="on_failure"):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"], on_failure="ignore")

    def test_invalid_pdb_ok_collision(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df = _df_one()
        df["pdb_ok"] = [True]
        with pytest.raises(ValueError, match="pdb_ok"):
            stp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"])

    def test_invalid_depth_without_msms(self):
        # When msms is missing, requesting 'depth' must raise.
        stp = aa.StructurePreprocessor(verbose=False)
        if shutil.which("msms") is None:
            with pytest.raises(RuntimeError, match="msms"):
                stp.encode_pdb(df_seq=_df_one(),
                               pdb_folder=str(PDB_FIXTURES),
                               features=["depth"])
        else:
            pytest.skip("msms available — skip absence assertion")

    def test_invalid_unsafe_entry(self):
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="entry"):
            stp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                           features=["bfactor"])

    def test_invalid_raise_on_failure_propagates(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(tmp_path),
                               features=["bfactor"], on_failure="raise")

    # ----- POSITIVES (≥10) -----
    def test_valid_bfactor_returns_dict(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_one(),
                                       pdb_folder=str(PDB_FIXTURES),
                                       features=["bfactor"])
        assert isinstance(d, dict)
        assert "P1" in d

    def test_valid_bfactor_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert d["P1"].shape == (16, 1)

    def test_valid_bfactor_values_finite(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        vals = d["P1"].ravel()
        assert np.all(np.isfinite(vals))

    def test_valid_pdb_ok_column(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, df_out = stp.encode_pdb(df_seq=_df_one(),
                                       pdb_folder=str(PDB_FIXTURES),
                                       features=["bfactor"])
        assert "pdb_ok" in df_out.columns
        assert bool(df_out["pdb_ok"].iloc[0])

    def test_valid_missing_pdb_yields_nan(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_one(),
                                       pdb_folder=str(tmp_path),
                                       features=["bfactor"],
                                       on_failure="nan")
        assert d["P1"].shape == (16, 1)
        assert np.isnan(d["P1"]).all()
        assert not bool(df_out["pdb_ok"].iloc[0])

    def test_valid_on_failure_drop(self, tmp_path):
        df = _df_two()
        # Copy only P1 from fixtures; P2 missing.
        (tmp_path / "P1.pdb").write_text(
            (PDB_FIXTURES / "P1.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=df, pdb_folder=str(tmp_path),
                                       features=["bfactor"],
                                       on_failure="drop")
        assert "P1" in d and "P2" not in d
        assert df_out["entry"].tolist() == ["P1"]

    def test_valid_two_proteins_keys_present(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_two(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert set(d.keys()) == {"P1", "P2"}

    def test_valid_verbose_runs(self):
        stp = aa.StructurePreprocessor(verbose=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert d["P1"].shape == (16, 1)

    def test_valid_p2_strand_bfactor_lower(self):
        # P1 helix has bfactor=30, P2 strand has bfactor=25.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_two(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        p1_mean = float(np.nanmean(d["P1"]))
        p2_mean = float(np.nanmean(d["P2"]))
        assert p1_mean > p2_mean

    @msms_required
    def test_valid_depth_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_one(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["depth"])
        assert d["P1"].shape == (16, 1)


class TestStpEncodePdbComplex:
    """Cross-parameter combinations for encode_pdb."""

    def test_complex_bfactor_with_two_entries_independent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_two(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["bfactor"])
        assert d["P1"].shape == (16, 1)
        assert d["P2"].shape == (12, 1)

    def test_complex_drop_keeps_succeeded(self, tmp_path):
        (tmp_path / "P1.pdb").write_text(
            (PDB_FIXTURES / "P1.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_two(),
                                       pdb_folder=str(tmp_path),
                                       features=["bfactor"],
                                       on_failure="drop")
        assert set(d.keys()) == {"P1"}
        assert df_out["pdb_ok"].tolist() == [True]

    def test_complex_nan_default_two_entries(self, tmp_path):
        (tmp_path / "P1.pdb").write_text(
            (PDB_FIXTURES / "P1.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_two(),
                                       pdb_folder=str(tmp_path),
                                       features=["bfactor"])
        assert np.isnan(d["P2"]).all()
        assert not np.isnan(d["P1"]).all()
        assert df_out["pdb_ok"].tolist() == [True, False]

    def test_complex_raise_when_any_missing(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_two(), pdb_folder=str(tmp_path),
                               features=["bfactor"], on_failure="raise")

    def test_complex_features_typed_as_int_rejected(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features=[7])

    def test_complex_features_int_rejected_with_typed_list(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_one(), pdb_folder=str(PDB_FIXTURES),
                           features="bfactor")  # str, not list

    def test_complex_default_on_failure_is_nan(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=pd.DataFrame({
                "entry": ["P_missing"],
                "sequence": ["ACDE"]}),
                pdb_folder=str(PDB_FIXTURES),
                features=["bfactor"])
        assert np.isnan(d["P_missing"]).all()


# ----------------------------------------------------------------------
# v1.1 — AF model-file features
# ----------------------------------------------------------------------
AF_FIXTURE_SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"


def _df_af():
    return pd.DataFrame({"entry": ["AF_TINY"], "sequence": [AF_FIXTURE_SEQ]})


class TestStpEncodePdbAFFeatures:
    """v1.1 AF features: plddt, plddt_disorder, plddt_tier, chi1/chi2_sincos,
    centroid_dist (+norm), contact_count_8A / _12A — single-param positives
    and negatives. All outputs must lie in [0, 1] (or be NaN)."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_plddt_disorder_threshold_below(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="plddt_disorder_threshold"):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["plddt_disorder"],
                           plddt_disorder_threshold=-1.0)

    def test_invalid_plddt_disorder_threshold_above(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="plddt_disorder_threshold"):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["plddt_disorder"],
                           plddt_disorder_threshold=150.0)

    def test_invalid_plddt_disorder_threshold_not_number(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["plddt_disorder"],
                           plddt_disorder_threshold="seventy")

    def test_invalid_af_feature_with_wrong_method(self):
        # ss3 belongs to encode_dssp, not encode_pdb.
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["ss3"])

    def test_invalid_pdb_folder_for_af(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder="/__nope__",
                           features=["plddt"])

    def test_invalid_features_mix_unknown_with_known(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["plddt", "mystery_key"])

    def test_invalid_features_dropped_key_asa(self):
        # v1.1: 'asa' (absolute) was removed from the registry.
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["asa"])

    def test_invalid_missing_fixture_raises_on_raise(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(tmp_path),
                               features=["plddt"], on_failure="raise")

    def test_invalid_unsafe_entry_for_af(self):
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="entry"):
            stp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                           features=["plddt"])

    def test_invalid_plddt_disorder_threshold_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                           features=["plddt_disorder"],
                           plddt_disorder_threshold=None)

    # ----- POSITIVES (≥10) -----
    def test_valid_plddt_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt"])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)

    def test_valid_plddt_in_unit_interval(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt"])
        vals = d["AF_TINY"].ravel()
        finite = vals[~np.isnan(vals)]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_plddt_disorder_binary(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt_disorder"])
        vals = d["AF_TINY"].ravel()
        finite = vals[~np.isnan(vals)]
        assert set(np.unique(finite)).issubset({0.0, 1.0})

    def test_valid_plddt_disorder_threshold_increases_count(self):
        # A higher threshold should label more residues disordered.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_low, _ = stp.encode_pdb(df_seq=_df_af(),
                                      pdb_folder=str(PDB_FIXTURES),
                                      features=["plddt_disorder"],
                                      plddt_disorder_threshold=30.0)
            d_high, _ = stp.encode_pdb(df_seq=_df_af(),
                                       pdb_folder=str(PDB_FIXTURES),
                                       features=["plddt_disorder"],
                                       plddt_disorder_threshold=95.0)
        assert np.nansum(d_high["AF_TINY"]) >= np.nansum(d_low["AF_TINY"])

    def test_valid_plddt_tier_shape_and_one_hot(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt_tier"])
        v = d["AF_TINY"]
        assert v.shape == (len(AF_FIXTURE_SEQ), 4)
        # Every non-NaN row should be one-hot (sum=1).
        rowsum = v.sum(axis=1)
        finite = rowsum[~np.isnan(rowsum)]
        np.testing.assert_allclose(finite, 1.0)

    def test_valid_chi1_sincos_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["chi1_sincos"])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 2)

    def test_valid_chi1_sincos_in_unit_interval(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["chi1_sincos"])
        vals = d["AF_TINY"]
        finite = vals[~np.isnan(vals)]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_chi1_finite_for_non_ala_gly(self):
        # The AF_TINY fixture is stereochemically valid: chi1 is defined for
        # every residue except ALA and GLY (which have no CG / CB).
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["chi1_sincos"])
        v = d["AF_TINY"]
        finite_per_res = ~np.isnan(v[:, 0])
        expected = np.array([aa_letter not in ("A", "G")
                              for aa_letter in AF_FIXTURE_SEQ])
        # All non-A/G residues should have finite chi1; A/G must be NaN.
        np.testing.assert_array_equal(finite_per_res, expected)

    def test_valid_chi2_finite_for_chi2_eligible(self):
        # chi2 requires N-CA-CB-CG-CD (or equivalent). Standard chi2-eligible
        # residues: R, N, D, E, F, H, I, K, L, M, P, Q, W, Y. C/S/T/V have
        # chi1 only; A/G have neither.
        chi2_eligible = set("RNDEFHIKLMPQWY")
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["chi2_sincos"])
        v = d["AF_TINY"]
        finite_per_res = ~np.isnan(v[:, 0])
        expected = np.array([aa_letter in chi2_eligible
                              for aa_letter in AF_FIXTURE_SEQ])
        np.testing.assert_array_equal(finite_per_res, expected)
        # And the finite count must be non-zero — the fixture must actually
        # exercise chi2 (was 0/30 with the v1.1 sidechain-less fixture).
        assert finite_per_res.sum() >= 14

    def test_valid_ca_centroid_dist_shape_and_range(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["ca_centroid_dist"])
        v = d["AF_TINY"]
        assert v.shape == (len(AF_FIXTURE_SEQ), 1)
        finite = v[~np.isnan(v)]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_centroid_dist_norm_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["ca_centroid_dist_norm"])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)

    def test_valid_contact_count_8A_in_unit_interval(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["contact_count_8A"])
        vals = d["AF_TINY"].ravel()
        finite = vals[~np.isnan(vals)]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_contact_count_12A_has_more_than_8A(self):
        # The wider radius should never produce a smaller count per residue.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["contact_count_8A",
                                            "contact_count_12A"])
        v = d["AF_TINY"]
        # Compare in raw count space: multiply normalized values by their
        # saturation caps (30 / 80) and check the 12A column ≥ 8A column.
        c8 = v[:, 0] * 30.0
        c12 = v[:, 1] * 80.0
        # NaN-safe compare: ignore positions where either is NaN.
        mask = ~np.isnan(c8) & ~np.isnan(c12)
        assert (c12[mask] >= c8[mask] - 1e-9).all()

    def test_valid_combined_AF_features_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        feats = ["plddt", "plddt_disorder", "plddt_tier",
                 "ca_centroid_dist_norm", "contact_count_8A"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=feats)
        # 1 + 1 + 4 + 1 + 1 = 8
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 8)


class TestStpEncodePdbAFComplex:
    """Cross-parameter combinations for the v1.1 AF features."""

    def test_complex_pdb_gz_round_trip(self):
        # The fixture AF_TINY.pdb.gz exists alongside AF_TINY.pdb; the
        # resolver should prefer the .pdb file (resolution order favors .pdb).
        # We test the gz path by giving a folder with ONLY the .gz copy.
        import gzip
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            src = Path(str(PDB_FIXTURES) + "/AF_TINY.pdb.gz")
            dst = Path(td) / "AF_TINY.pdb.gz"
            dst.write_bytes(src.read_bytes())
            stp = aa.StructurePreprocessor(verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d, df_out = stp.encode_pdb(df_seq=_df_af(),
                                           pdb_folder=td,
                                           features=["plddt"])
            assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)
            assert bool(df_out["pdb_ok"].iloc[0])

    def test_complex_plddt_decreases_at_termini(self):
        # AF_TINY fixture has pLDDT tiered high in middle, low at ends.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt"])
        v = d["AF_TINY"].ravel()
        mid = float(np.nanmean(v[10:20]))
        ends = float(np.nanmean(np.concatenate([v[:3], v[-3:]])))
        assert mid > ends

    def test_complex_drop_failed_AF_entry(self, tmp_path):
        df = pd.DataFrame({"entry": ["AF_TINY", "GONE"],
                           "sequence": [AF_FIXTURE_SEQ, "ACDE"]})
        # Copy only AF_TINY.pdb to the tmp folder; second entry missing.
        from pathlib import Path
        src = Path(str(PDB_FIXTURES) + "/AF_TINY.pdb")
        (tmp_path / "AF_TINY.pdb").write_text(src.read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=df, pdb_folder=str(tmp_path),
                                       features=["plddt"],
                                       on_failure="drop")
        assert "AF_TINY" in d and "GONE" not in d
        assert df_out["entry"].tolist() == ["AF_TINY"]

    def test_complex_nan_default_with_missing_AF(self, tmp_path):
        df = pd.DataFrame({"entry": ["GONE"], "sequence": ["ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=df, pdb_folder=str(tmp_path),
                                       features=["plddt_tier"])
        assert np.isnan(d["GONE"]).all()
        assert d["GONE"].shape == (4, 4)
        assert not bool(df_out["pdb_ok"].iloc[0])

    def test_complex_two_features_concat_order(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt", "ca_centroid_dist"])
        v = d["AF_TINY"]
        # plddt is column 0, centroid_dist is column 1
        # both in [0, 1]
        assert v.shape[1] == 2

    def test_complex_plddt_and_bfactor_coexist(self):
        # Both feature keys read the B-factor column but emit DIFFERENT
        # subcategories — the corpus-derived df_scales lets the redundancy
        # filter spot the duplication (high corr).
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["plddt", "bfactor"])
        v = d["AF_TINY"]
        # Both come from the same raw column → after the same /100
        # normalization recipe, values should be identical.
        np.testing.assert_allclose(v[:, 0], v[:, 1], equal_nan=True)

    def test_complex_build_cat_for_all_AF_keys(self):
        stp = aa.StructurePreprocessor(verbose=False)
        feats = ["plddt", "plddt_disorder", "plddt_tier",
                 "chi1_sincos", "chi2_sincos",
                 "ca_centroid_dist", "ca_centroid_dist_norm",
                 "contact_count_8A", "contact_count_12A"]
        df_cat = stp.build_cat(features=feats)
        # 1 + 1 + 4 + 2 + 2 + 1 + 1 + 1 + 1 = 14
        assert df_cat.shape == (14, 5)
        assert (df_cat[ut.COL_CAT] == "Structure").all()


# ----------------------------------------------------------------------
# v1.1 — HSE (half-sphere exposure, Hamelryck 2005)
# ----------------------------------------------------------------------
class TestStpEncodePdbHSE:
    """encode_pdb feature key 'hse' — Bio.PDB.HSExposureCA-based.

    HSE is the Hamelryck-2005 directional ASA-like signal: counts of Cα atoms
    in the upper / lower half-sphere defined by the pseudo-Cβ direction at a
    13 Å radius. Computed via biopython's HSExposureCA (no Cβ requirement —
    works on glycines too, unlike HSExposureCB).
    """

    def test_invalid_hse_with_wrong_method(self):
        # 'hse' belongs to encode_pdb, not encode_dssp.
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                            features=["hse"])

    def test_valid_hse_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["hse"])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 2)

    def test_valid_hse_in_unit_interval(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["hse"])
        vals = d["AF_TINY"]
        finite = vals[~np.isnan(vals)]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_hse_finite_for_internal_residues(self):
        # HSExposureCA needs at least one neighbor on each side to define
        # the pseudo-Cβ direction, so terminal residues come back NaN.
        # The middle residues (≥ 2 from each terminus) should be finite.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["hse"])
        v = d["AF_TINY"]
        # Middle 20 residues (rows 5-25) must all have finite hse_up + hse_down.
        mid = v[5:25]
        assert np.isfinite(mid).all()

    def test_valid_hse_up_plus_down_correlates_with_cn(self):
        # At the same radius, hse_up + hse_down ≈ total Cα contact number.
        # On the helical AF_TINY fixture this should land in a sensible range
        # (sum > 0 for internal residues).
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["hse"])
        v = d["AF_TINY"]
        mid = v[5:25]
        total = mid.sum(axis=1)
        # On a 30-residue helix, each interior residue sees ~10 neighbors
        # within 13 Å → /30 normalization → values around 0.3-0.5.
        assert (total > 0).all()
        assert float(total.mean()) > 0.1

    def test_valid_hse_combined_with_other_pdb_features(self):
        # hse should compose cleanly with other encode_pdb keys.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["hse", "plddt", "bfactor"])
        # 2 + 1 + 1 = 4
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 4)

    def test_valid_hse_in_build_cat_resolves_to_structure(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["hse"])
        # 2 rows (hse_up + hse_down), both under category='Structure'.
        assert df_cat.shape == (2, 5)
        assert (df_cat[ut.COL_CAT] == "Structure").all()
        assert (df_cat[ut.COL_SUBCAT] == "Half-sphere exposure (HSE-CA)").all()

    def test_valid_hse_in_build_pseudo_scales(self):
        # hse values get per-AA averaged for the corpus df_scales.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_af(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["hse"])
            df_scales = stp.build_pseudo_scales(
                df_seq=_df_af(), dict_num=d, features=["hse"])
        assert df_scales.shape == (20, 2)
        assert list(df_scales.columns) == ["hse_up", "hse_down"]


# ----------------------------------------------------------------------
# v1.2 — Disulfide-bond participation
# ----------------------------------------------------------------------
SS_BOND_SEQ = "ACAACA"


def _df_ss():
    return pd.DataFrame({"entry": ["SS_BOND"], "sequence": [SS_BOND_SEQ]})


class TestStpEncodePdbDisulfide:
    """encode_pdb feature key 'disulfide' (v1.2).

    Detects CYS-CYS SG-SG distances < 2.5 Å. Per residue:
      column 0 (participates): 1.0 if CYS in bond, 0.0 if free CYS, NaN otherwise
      column 1 (partner_distance): SG-SG distance in Å normalized by /5, NaN otherwise
    Fixture SS_BOND.pdb has two bonded CYS at positions 2 and 5
    (sequence ACAACA).
    """

    # ----- NEGATIVES (≥5) -----
    def test_invalid_disulfide_with_wrong_method(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_dssp(df_seq=_df_ss(), pdb_folder=str(PDB_FIXTURES),
                            features=["disulfide"])

    def test_invalid_disulfide_in_pae_method(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=_df_ss(), pae_folder=str(PDB_FIXTURES),
                           features=["disulfide"])

    def test_invalid_disulfide_pdb_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            stp.encode_pdb(df_seq=_df_ss(), pdb_folder="/__nope__",
                           features=["disulfide"])

    def test_invalid_disulfide_features_empty(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pdb(df_seq=_df_ss(), pdb_folder=str(PDB_FIXTURES),
                           features=[])

    def test_invalid_disulfide_raise_on_missing_pdb(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pdb(df_seq=_df_ss(), pdb_folder=str(tmp_path),
                               features=["disulfide"], on_failure="raise")

    # ----- POSITIVES (≥10) -----
    def test_valid_disulfide_shape(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        assert d["SS_BOND"].shape == (len(SS_BOND_SEQ), 2)

    def test_valid_disulfide_in_unit_interval(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        vals = d["SS_BOND"]
        finite = vals[~np.isnan(vals)]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_disulfide_non_cys_are_nan(self):
        # Non-CYS residues must be NaN in both columns.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        v = d["SS_BOND"]
        for i, a in enumerate(SS_BOND_SEQ):
            if a != "C":
                assert np.isnan(v[i]).all()

    def test_valid_disulfide_bonded_cys_participates_one(self):
        # Both CYS in SS_BOND should participate (=1.0).
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        v = d["SS_BOND"]
        cys_indices = [i for i, a in enumerate(SS_BOND_SEQ) if a == "C"]
        for i in cys_indices:
            assert v[i, 0] == 1.0

    def test_valid_disulfide_partner_distance_finite(self):
        # Bonded CYS get a finite partner_distance (≤ 1.0 after /5 norm).
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        v = d["SS_BOND"]
        cys_indices = [i for i, a in enumerate(SS_BOND_SEQ) if a == "C"]
        for i in cys_indices:
            assert np.isfinite(v[i, 1])
            assert 0 < v[i, 1] <= 1.0

    def test_valid_disulfide_free_cys_participates_zero(self):
        # AF_TINY has 2 CYS that are too far apart (helical, 20 residues
        # apart) to form a disulfide → participates=0.0, distance=NaN.
        df = pd.DataFrame({"entry": ["AF_TINY"],
                           "sequence": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        v = d["AF_TINY"]
        # CYS at positions 2 and 22 (1-indexed) = idx 1 and 21.
        for i in (1, 21):
            assert v[i, 0] == 0.0
            assert np.isnan(v[i, 1])

    def test_valid_disulfide_build_cat(self):
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["disulfide"])
        assert df_cat.shape == (2, 5)
        assert (df_cat[ut.COL_CAT] == "Structure").all()
        assert (df_cat[ut.COL_SUBCAT] == "Disulfide bond (CYS-CYS)").all()

    def test_valid_disulfide_combined_with_other_features(self):
        # disulfide composes cleanly with bfactor / plddt.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide", "bfactor"])
        # 2 (disulfide) + 1 (bfactor) = 3
        assert d["SS_BOND"].shape == (len(SS_BOND_SEQ), 3)

    def test_valid_disulfide_missing_pdb_nan_default(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=_df_ss(),
                                       pdb_folder=str(tmp_path),
                                       features=["disulfide"])
        assert np.isnan(d["SS_BOND"]).all()
        assert not bool(df_out["pdb_ok"].iloc[0])

    def test_valid_disulfide_drop_on_missing(self, tmp_path):
        df = pd.DataFrame({"entry": ["SS_BOND", "GONE"],
                           "sequence": [SS_BOND_SEQ, "ACDE"]})
        # Copy SS_BOND.pdb to tmp; GONE is missing.
        from pathlib import Path
        (tmp_path / "SS_BOND.pdb").write_text(
            (PDB_FIXTURES / "SS_BOND.pdb").read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pdb(df_seq=df, pdb_folder=str(tmp_path),
                                       features=["disulfide"],
                                       on_failure="drop")
        assert "SS_BOND" in d and "GONE" not in d
        assert df_out["entry"].tolist() == ["SS_BOND"]

    def test_valid_disulfide_partner_distance_normalized(self):
        # SS_BOND fixture has SG-SG distance ≈ 1.308 Å → normalized ≈ 0.262.
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, _ = stp.encode_pdb(df_seq=_df_ss(),
                                  pdb_folder=str(PDB_FIXTURES),
                                  features=["disulfide"])
        v = d["SS_BOND"]
        # Both bonded CYS should have the same partner distance ≈ 0.262.
        np.testing.assert_allclose(v[1, 1], 0.262, atol=0.01)
        np.testing.assert_allclose(v[4, 1], 0.262, atol=0.01)
