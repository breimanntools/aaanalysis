"""This is a script to test StructurePreprocessor.encode_pae() —
AlphaFold PAE sidecar features (commit 3 of v1.1).
"""
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa
import aaanalysis.utils as ut

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

PDB_FIXTURES = Path(__file__).resolve().parents[3] / \
    "aaanalysis" / "_data" / "pdb_test"
AF_FIXTURE_SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"


# I Helper Functions
def _df_af():
    return pd.DataFrame({"entry": ["AF_TINY"], "sequence": [AF_FIXTURE_SEQ]})


def _af_folder_primary_name() -> tempfile.TemporaryDirectory:
    """Build a temp folder containing AF_TINY.json (primary resolver name)."""
    td = tempfile.TemporaryDirectory()
    shutil.copy(PDB_FIXTURES / "AF_TINY_pae.json",
                Path(td.name) / "AF_TINY.json")
    return td


def _make_synthetic_pae_json(folder: Path, entry: str, L: int,
                             seed: int = 0,
                             name: str = None):
    """Write a synthetic PAE matrix as JSON in ``folder``."""
    rng = np.random.default_rng(seed)
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    sep = np.abs(i - j).astype(float)
    pae = 1.0 + 0.6 * sep + rng.uniform(-0.2, 0.2, size=(L, L))
    pae = np.clip(pae, 0.0, 31.75)
    if name is None:
        name = f"{entry}.json"
    (folder / name).write_text(json.dumps({
        "predicted_aligned_error": pae.tolist()}))
    return pae


# II Test Classes
class TestStpEncodePae:
    """Single-parameter normal + invalid cases for encode_pae."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_df_seq_none(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=None, pae_folder=td.name,
                           features=["pae_row_mean"])
        td.cleanup()

    def test_invalid_pae_folder_none(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pae_folder"):
            stp.encode_pae(df_seq=_df_af(), pae_folder=None,
                           features=["pae_row_mean"])

    def test_invalid_pae_folder_nonexistent(self):
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pae_folder"):
            stp.encode_pae(df_seq=_df_af(),
                           pae_folder="/__nope__/__here__",
                           features=["pae_row_mean"])

    def test_invalid_features_empty(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name, features=[])
        td.cleanup()

    def test_invalid_features_unknown(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["mystery"])
        td.cleanup()

    def test_invalid_features_wrong_method(self):
        # ss3 belongs to encode_dssp, not encode_pae.
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["ss3"])
        td.cleanup()

    def test_invalid_local_window_negative(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_local_mean"], local_window=-1)
        td.cleanup()

    def test_invalid_band_edges_not_pair(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pae_band_edges"):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_band_means"],
                           pae_band_edges=(5,))
        td.cleanup()

    def test_invalid_band_edges_not_ascending(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_band_means"],
                           pae_band_edges=(15, 5))
        td.cleanup()

    def test_invalid_on_failure(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="on_failure"):
            stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                           features=["pae_row_mean"], on_failure="skip")
        td.cleanup()

    def test_invalid_missing_pae_raises_on_raise(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                stp.encode_pae(df_seq=_df_af(), pae_folder=str(tmp_path),
                               features=["pae_row_mean"], on_failure="raise")

    def test_invalid_pae_ok_column_collision(self):
        td = _af_folder_primary_name()
        df = _df_af()
        df["pae_ok"] = [True]
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pae_ok"):
            stp.encode_pae(df_seq=df, pae_folder=td.name,
                           features=["pae_row_mean"])
        td.cleanup()

    def test_invalid_unsafe_entry_for_pae(self):
        td = _af_folder_primary_name()
        df = pd.DataFrame({"entry": ["../etc"], "sequence": ["ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="entry"):
            stp.encode_pae(df_seq=df, pae_folder=td.name,
                           features=["pae_row_mean"])
        td.cleanup()

    # ----- POSITIVES (≥10) -----
    def test_valid_row_mean_shape(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_row_mean"])
        td.cleanup()
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)

    def test_valid_row_mean_in_unit_interval(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_row_mean"])
        td.cleanup()
        vals = d["AF_TINY"].ravel()
        assert (vals >= 0).all() and (vals <= 1).all()

    def test_valid_row_min_le_row_max(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_row_min", "pae_row_max"])
        td.cleanup()
        v = d["AF_TINY"]
        # pae_row_min ≤ pae_row_max per residue
        assert (v[:, 0] <= v[:, 1] + 1e-9).all()

    def test_valid_local_mean_lt_distal_mean_realistic(self):
        # On a realistic PAE matrix (PAE grows with sequence separation),
        # local-window mean should typically be < distal mean.
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_local_mean",
                                            "pae_distal_mean"],
                                  local_window=5)
        td.cleanup()
        v = d["AF_TINY"]
        # In the middle of the protein both windows are populated.
        mid = v[10:20]
        assert float(np.nanmean(mid[:, 0])) < float(np.nanmean(mid[:, 1]))

    def test_valid_asymmetry_in_unit_interval(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_asymmetry"])
        td.cleanup()
        v = d["AF_TINY"].ravel()
        assert (v >= 0).all() and (v <= 1).all()

    def test_valid_band_means_shape(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_band_means"])
        td.cleanup()
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 3)

    def test_valid_band_means_bands_increasing(self):
        # On a realistic PAE (grows with sep), band means should
        # generally increase across bands [close, mid, far].
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_band_means"])
        td.cleanup()
        # Middle residues have all three bands populated.
        v = d["AF_TINY"][10:20]
        c, m, f = (float(np.nanmean(v[:, 0])),
                   float(np.nanmean(v[:, 1])),
                   float(np.nanmean(v[:, 2])))
        assert c < m < f

    def test_valid_combined_features_shape(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        feats = ["pae_row_mean", "pae_local_mean", "pae_distal_mean",
                 "pae_asymmetry", "pae_band_means"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=feats)
        td.cleanup()
        # 1 + 1 + 1 + 1 + 3 = 7
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 7)

    def test_valid_pae_ok_true(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, df_out = stp.encode_pae(return_df=True, df_seq=_df_af(), pae_folder=td.name,
                                       features=["pae_row_mean"])
        td.cleanup()
        assert bool(df_out["pae_ok"].iloc[0])

    def test_valid_missing_pae_nan_default(self, tmp_path):
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=_df_af(),
                                       pae_folder=str(tmp_path),
                                       features=["pae_row_mean"])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)
        assert np.isnan(d["AF_TINY"]).all()
        assert not bool(df_out["pae_ok"].iloc[0])

    def test_valid_af_canonical_filename_fallback(self, tmp_path):
        # When only the AF-DB canonical name is present, resolver finds it.
        src = PDB_FIXTURES / "AF-AF_TINY-F1-predicted_aligned_error_v4.json"
        (tmp_path / src.name).write_text(src.read_text())
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=_df_af(),
                                       pae_folder=str(tmp_path),
                                       features=["pae_row_mean"])
        assert bool(df_out["pae_ok"].iloc[0])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)


class TestStpEncodePaeComplex:
    """Cross-parameter combinations for encode_pae."""

    def test_complex_local_window_zero_yields_nan_local(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_local_mean"],
                                  local_window=0)
        td.cleanup()
        # window=0, self excluded → no local neighbors anywhere → all NaN
        assert np.isnan(d["AF_TINY"]).all()

    def test_complex_drop_failed_pae_entry(self, tmp_path):
        df = pd.DataFrame({"entry": ["AF_TINY", "GONE"],
                           "sequence": [AF_FIXTURE_SEQ, "ACDE"]})
        # Copy only AF_TINY's PAE to tmp folder; second entry missing.
        shutil.copy(PDB_FIXTURES / "AF_TINY_pae.json",
                    tmp_path / "AF_TINY.json")
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=df, pae_folder=str(tmp_path),
                                       features=["pae_row_mean"],
                                       on_failure="drop")
        assert "AF_TINY" in d and "GONE" not in d
        assert df_out["entry"].tolist() == ["AF_TINY"]

    def test_complex_two_entries_mixed_success(self, tmp_path):
        shutil.copy(PDB_FIXTURES / "AF_TINY_pae.json",
                    tmp_path / "AF_TINY.json")
        df = pd.DataFrame({"entry": ["AF_TINY", "GONE"],
                           "sequence": [AF_FIXTURE_SEQ, "ACDE"]})
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=df, pae_folder=str(tmp_path),
                                       features=["pae_row_mean"])
        assert df_out["pae_ok"].tolist() == [True, False]
        assert np.isnan(d["GONE"]).all()
        assert not np.isnan(d["AF_TINY"]).all()

    def test_complex_band_edges_kwarg(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=["pae_band_means"],
                                  pae_band_edges=(2, 8))
        td.cleanup()
        # Shape is still (L, 3) regardless of band_edges values.
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 3)

    def test_complex_pae_json_gz_round_trip(self, tmp_path):
        import gzip
        src = PDB_FIXTURES / "AF_TINY_pae.json"
        with open(src, "rb") as fin, \
             gzip.open(tmp_path / "AF_TINY.json.gz", "wb") as fout:
            shutil.copyfileobj(fin, fout)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=_df_af(),
                                       pae_folder=str(tmp_path),
                                       features=["pae_row_mean"])
        assert bool(df_out["pae_ok"].iloc[0])
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 1)

    def test_complex_malformed_json_warns_and_continues(self, tmp_path):
        (tmp_path / "AF_TINY.json").write_text("{not valid json")
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=_df_af(),
                                       pae_folder=str(tmp_path),
                                       features=["pae_row_mean"])
        assert not bool(df_out["pae_ok"].iloc[0])

    def test_complex_shape_mismatch_warns(self, tmp_path):
        # L=10 PAE matrix vs sequence length 30 → should fail validation
        _make_synthetic_pae_json(tmp_path, "AF_TINY", L=10, seed=2)
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d, df_out = stp.encode_pae(return_df=True, df_seq=_df_af(),
                                       pae_folder=str(tmp_path),
                                       features=["pae_row_mean"])
        assert not bool(df_out["pae_ok"].iloc[0])

    def test_complex_combined_pae_with_local_and_band(self):
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        feats = ["pae_local_mean", "pae_distal_mean", "pae_band_means",
                 "pae_asymmetry"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = stp.encode_pae(df_seq=_df_af(), pae_folder=td.name,
                                  features=feats, local_window=4,
                                  pae_band_edges=(3, 10))
        td.cleanup()
        # 1 + 1 + 3 + 1 = 6
        assert d["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 6)

    def test_complex_pae_band_means_three_categories_in_registry(self):
        # Registry assertion: pae_band_means contributes 3 dims; build_cat
        # reflects that as 3 rows.
        stp = aa.StructurePreprocessor(verbose=False)
        df_cat = stp.build_cat(features=["pae_band_means"])
        assert df_cat.shape == (3, 5)
        assert (df_cat[ut.COL_SUBCAT] == "AlphaFold PAE (3-band means)").all()

    def test_complex_pae_and_structural_combine_dict_nums(self):
        # End-to-end: PDB features + PAE features stitched via combine_dict_nums.
        td = _af_folder_primary_name()
        stp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_pdb = stp.encode_pdb(df_seq=_df_af(),
                                      pdb_folder=str(PDB_FIXTURES),
                                      features=["plddt"])
            d_pae = stp.encode_pae(df_seq=_df_af(),
                                      pae_folder=td.name,
                                      features=["pae_row_mean"])
        td.cleanup()
        dict_num = aa.combine_dict_nums(dict_nums=[d_pdb, d_pae])
        assert dict_num["AF_TINY"].shape == (len(AF_FIXTURE_SEQ), 2)
