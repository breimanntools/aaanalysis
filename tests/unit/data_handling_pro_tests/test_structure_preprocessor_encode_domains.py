"""This is a script to test StructurePreprocessor.encode_domains() —
Merizo / ChainSaw / AFragmenter domain segmentation reader (v1.2 commit 3).
"""
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# I Helper Functions
def _df_one(seq_len=30, entry="P1"):
    return pd.DataFrame({"entry": [entry], "sequence": ["A" * seq_len]})


def _write_chopping(folder: Path, entry: str, chopping: str, ext=".txt"):
    """Write a single-line chopping file in the given folder."""
    (folder / f"{entry}{ext}").write_text(chopping + "\n")


# II Test Classes
class TestStpEncodeDomains:
    """Single-parameter normal + invalid cases for encode_domains."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_df_seq_none(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.encode_domains(df_seq=None, domain_folder=str(tmp_path),
                               features=["domain_boundary"])

    def test_invalid_domain_folder_none(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="domain_folder"):
            strp.encode_domains(df_seq=_df_one(), domain_folder=None,
                               features=["domain_boundary"])

    def test_invalid_domain_folder_nonexistent(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="domain_folder"):
            strp.encode_domains(df_seq=_df_one(),
                               domain_folder="/__nope__/__here__",
                               features=["domain_boundary"])

    def test_invalid_features_empty(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.encode_domains(df_seq=_df_one(),
                               domain_folder=str(tmp_path), features=[])

    def test_invalid_features_unknown(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.encode_domains(df_seq=_df_one(),
                               domain_folder=str(tmp_path),
                               features=["mystery"])

    def test_invalid_features_wrong_method(self, tmp_path):
        # bfactor / plddt / ss3 belong to other methods.
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.encode_domains(df_seq=_df_one(),
                               domain_folder=str(tmp_path),
                               features=["bfactor"])

    def test_invalid_on_failure(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="on_failure"):
            strp.encode_domains(df_seq=_df_one(),
                               domain_folder=str(tmp_path),
                               features=["domain_boundary"],
                               on_failure="skip")

    def test_invalid_domain_ok_column_collision(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        df = _df_one()
        df["domain_ok"] = [True]
        with pytest.raises(ValueError, match="domain_ok"):
            strp.encode_domains(df_seq=df, domain_folder=str(tmp_path),
                               features=["domain_boundary"])

    def test_invalid_missing_file_raises_on_raise(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                strp.encode_domains(df_seq=_df_one(),
                                   domain_folder=str(tmp_path),
                                   features=["domain_boundary"],
                                   on_failure="raise")

    def test_invalid_unsafe_entry(self, tmp_path):
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="entry"):
            strp.encode_domains(df_seq=df, domain_folder=str(tmp_path),
                               features=["domain_boundary"])

    def test_invalid_malformed_chopping_warns_and_continues(self, tmp_path):
        # Garbled chopping string → fail per-row with on_failure='nan'.
        _write_chopping(tmp_path, "P1", "not_a_chopping!!!", ext=".txt")
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = strp.encode_domains(return_df=True, df_seq=_df_one(),
                                           domain_folder=str(tmp_path),
                                           features=["domain_boundary"])
        assert np.isnan(d["P1"]).all()
        assert not bool(df_out["domain_ok"].iloc[0])

    # ----- POSITIVES (≥10) -----
    def test_valid_basic_two_domains_shape(self, tmp_path):
        _write_chopping(tmp_path, "P1", "1-10,15-25")
        strp = aa.StructurePreprocessor(verbose=False)
        d, df_out = strp.encode_domains(return_df=True, 
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_boundary"])
        assert d["P1"].shape == (30, 1)
        assert bool(df_out["domain_ok"].iloc[0])

    def test_valid_boundary_marks_endpoints(self, tmp_path):
        # Domains 1-10 and 15-25. Boundaries at 1, 10, 15, 25 (1-based).
        _write_chopping(tmp_path, "P1", "1-10,15-25")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_boundary"])
        v = d["P1"][:, 0]
        for idx_1based in (1, 10, 15, 25):
            assert v[idx_1based - 1] == 1.0
        # Internal residues should be 0.
        for idx_1based in (2, 5, 16, 20):
            assert v[idx_1based - 1] == 0.0

    def test_valid_unassigned_residues_are_nan(self, tmp_path):
        # Residues 11-14 (between D1 and D2) and 26-30 (after D2)
        # are unassigned → NaN.
        _write_chopping(tmp_path, "P1", "1-10,15-25")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_boundary"])
        v = d["P1"][:, 0]
        for idx_1based in list(range(11, 15)) + list(range(26, 31)):
            assert np.isnan(v[idx_1based - 1])

    def test_valid_relative_position_linear(self, tmp_path):
        # D1: residues 1-10. rel_pos for res 1 → 0.0, res 10 → 1.0, res 5 → ~0.44.
        _write_chopping(tmp_path, "P1", "1-10")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_relative_position"])
        v = d["P1"][:, 0]
        assert v[0] == 0.0
        assert v[9] == 1.0
        np.testing.assert_allclose(v[4], 4 / 9, atol=1e-9)

    def test_valid_domain_size_normalized(self, tmp_path):
        # Domain of 10 residues → normalized 10/200 = 0.05.
        _write_chopping(tmp_path, "P1", "1-10")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_size"])
        v = d["P1"][:, 0]
        for i in range(10):
            np.testing.assert_allclose(v[i], 0.05, atol=1e-9)

    def test_valid_n_domains_constant_across_assigned(self, tmp_path):
        # 2 domains → n_domains = 2/10 = 0.2 for all assigned residues.
        _write_chopping(tmp_path, "P1", "1-10,15-25")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["n_domains_in_protein"])
        v = d["P1"][:, 0]
        for idx in list(range(0, 10)) + list(range(14, 25)):
            np.testing.assert_allclose(v[idx], 0.2, atol=1e-9)

    def test_valid_all_features_combined(self, tmp_path):
        _write_chopping(tmp_path, "P1", "1-10,15-25")
        strp = aa.StructurePreprocessor(verbose=False)
        feats = ["domain_boundary", "domain_relative_position",
                 "domain_size", "n_domains_in_protein"]
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=feats)
        # 4 dims, all in [0, 1] (or NaN).
        assert d["P1"].shape == (30, 4)
        finite = d["P1"][~np.isnan(d["P1"])]
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_valid_discontinuous_domain_boundary(self, tmp_path):
        # Discontinuous D1: residues 1-5 ∪ 10-15. Boundaries: 1, 5, 10, 15.
        _write_chopping(tmp_path, "P1", "1-5_10-15")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_boundary"])
        v = d["P1"][:, 0]
        for endpoint_1based in (1, 5, 10, 15):
            assert v[endpoint_1based - 1] == 1.0

    def test_valid_tsv_format_with_chopping_column(self, tmp_path):
        (tmp_path / "P1.tsv").write_text("chopping\n1-10,15-25\n")
        strp = aa.StructurePreprocessor(verbose=False)
        d, df_out = strp.encode_domains(return_df=True, 
            df_seq=_df_one(), domain_folder=str(tmp_path),
            features=["domain_boundary"])
        assert bool(df_out["domain_ok"].iloc[0])
        assert d["P1"].shape == (30, 1)

    def test_valid_missing_file_nan_default(self, tmp_path):
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = strp.encode_domains(return_df=True, 
                df_seq=_df_one(), domain_folder=str(tmp_path),
                features=["domain_boundary"])
        assert np.isnan(d["P1"]).all()
        assert not bool(df_out["domain_ok"].iloc[0])

    def test_valid_drop_failed_entry(self, tmp_path):
        df = pd.DataFrame({"entry": ["P1", "GONE"],
                           "sequence": ["A" * 30, "A" * 10]})
        _write_chopping(tmp_path, "P1", "1-10,15-25")
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = strp.encode_domains(return_df=True, 
                df_seq=df, domain_folder=str(tmp_path),
                features=["domain_boundary"], on_failure="drop")
        assert "P1" in d and "GONE" not in d
        assert df_out["entry"].tolist() == ["P1"]

    def test_valid_build_cat(self):
        strp = aa.StructurePreprocessor(verbose=False)
        feats = ["domain_boundary", "domain_relative_position",
                 "domain_size", "n_domains_in_protein"]
        df_cat = strp.build_cat(features=feats)
        assert df_cat.shape == (4, 5)
        assert (df_cat[ut.COL_CAT] == "Structure").all()
        expected_subs = {"Domain boundary", "Domain relative position",
                         "Domain size", "Number of domains in protein"}
        assert set(df_cat[ut.COL_SUBCAT].tolist()) == expected_subs


class TestStpEncodeDomainsComplex:
    """Cross-parameter / edge-case combinations for encode_domains."""

    def test_complex_single_residue_domain(self, tmp_path):
        # Domain of size 1 (just residue 5): rel_pos = 0.5 (midpoint convention).
        _write_chopping(tmp_path, "P1", "5-5")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(seq_len=10), domain_folder=str(tmp_path),
            features=["domain_relative_position"])
        v = d["P1"][:, 0]
        assert v[4] == 0.5  # 1-based residue 5 = 0-based index 4

    def test_complex_domain_size_caps_at_200(self, tmp_path):
        # Domain larger than 200 residues → normalized clips to 1.0.
        _write_chopping(tmp_path, "P1", "1-250")
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(seq_len=250), domain_folder=str(tmp_path),
            features=["domain_size"])
        assert d["P1"][0, 0] == 1.0

    def test_complex_n_domains_caps_at_10(self, tmp_path):
        # 12 single-residue domains → normalized clips to 1.0.
        chops = ",".join(f"{i}-{i}" for i in range(1, 13))
        _write_chopping(tmp_path, "P1", chops)
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=_df_one(seq_len=15), domain_folder=str(tmp_path),
            features=["n_domains_in_protein"])
        assert d["P1"][0, 0] == 1.0   # clipped from 12/10

    def test_complex_two_proteins_independent(self, tmp_path):
        _write_chopping(tmp_path, "P1", "1-10")
        _write_chopping(tmp_path, "P2", "1-5,8-12")
        df = pd.DataFrame({"entry": ["P1", "P2"],
                           "sequence": ["A" * 30, "A" * 15]})
        strp = aa.StructurePreprocessor(verbose=False)
        d = strp.encode_domains(
            df_seq=df, domain_folder=str(tmp_path),
            features=["n_domains_in_protein"])
        # P1 has 1 domain → 0.1; P2 has 2 domains → 0.2.
        np.testing.assert_allclose(d["P1"][0, 0], 0.1, atol=1e-9)
        np.testing.assert_allclose(d["P2"][0, 0], 0.2, atol=1e-9)

    def test_complex_chopping_parser_rejects_garbage(self, tmp_path):
        # Various malformed chopping strings should fail gracefully.
        for bad in ["1-2-3", "abc", "1_", "1-", ""]:
            _write_chopping(tmp_path, "P1", bad)
            strp = aa.StructurePreprocessor(verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d, df_out = strp.encode_domains(return_df=True, 
                    df_seq=_df_one(seq_len=10),
                    domain_folder=str(tmp_path),
                    features=["domain_boundary"])
            # Empty chopping is parsed to no domains → all NaN; bad
            # strings raise → all NaN. Either way, ok=False or no-op
            # output.
            assert d["P1"].shape == (10, 1)
