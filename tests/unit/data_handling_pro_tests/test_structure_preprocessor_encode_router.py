"""Tests for StructurePreprocessor per-feature failure isolation and the
``encode`` feature/backend router (issue #340).

The isolation tests MOCK per-feature encoders / ``is_msms_available`` so they
run deterministically without the real ``mkdssp`` / ``msms`` binaries. The
router-dispatch tests MOCK the four ``encode_*`` methods to assert each feature
key is routed to its owning backend; a separate end-to-end test uses the
bundled ``AF_TINY`` PDB + PAE fixtures (binary-free) to check the merged
output matches manual composition.
"""
import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import aaanalysis as aa
from aaanalysis.data_handling_pro._backend.struct_preproc.feature_registry import (
    REGISTRY)

aa.options["verbose"] = False

MODULE = "aaanalysis.data_handling_pro._struct_preproc"
PDB_FIXTURES = Path(aa.__file__).resolve().parent / "_data" / "pdb_test"
AF_SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"   # AF_TINY (L=30)


# I Helper Functions
def _df_af():
    return pd.DataFrame({"entry": ["AF_TINY"], "sequence": [AF_SEQ]})


def _pae_dir():
    """Temp dir holding the bundled AF_TINY PAE sidecar as <entry>.json."""
    d = tempfile.mkdtemp()
    shutil.copy(PDB_FIXTURES / "AF_TINY_pae.json", Path(d) / "AF_TINY.json")
    return d


def _const_sub(df_seq, feats, val):
    """A valid ``(dict_num, df)`` sub-encoder result filled with ``val``.

    Used to stand in for a real ``encode_*`` method so the router's dispatch +
    reassembly can be checked without touching any structure files.
    """
    total = sum(REGISTRY[f]["num_dims"] for f in feats)
    d = {}
    for e, s in zip(df_seq["entry"].tolist(), df_seq["sequence"].tolist()):
        d[e] = np.full((len(s), total), float(val), dtype=np.float64)
    return d, df_seq.copy()


# II Test Classes
class TestEncodePdbDepthIsolation:
    """encode_pdb 'depth' isolation when msms is unavailable (mocked)."""

    def test_depth_isolated_nan_default(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                d = strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                                   features=["bfactor", "depth"])
        v = d["AF_TINY"]
        assert v.shape == (len(AF_SEQ), 2)
        assert not np.isnan(v[:, 0]).all()      # bfactor kept
        assert np.isnan(v[:, 1]).all()          # depth isolated to NaN

    def test_depth_isolation_warns_exactly_once(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                               features=["bfactor", "depth"])
        depth_warns = [x for x in rec if issubclass(x.category, UserWarning)
                       and "depth" in str(x.message) and "msms" in str(x.message)]
        assert len(depth_warns) == 1
        assert "other features kept" in str(depth_warns[0].message)

    def test_depth_isolation_warns_once_across_two_entries(self):
        # Two entries both missing depth -> still exactly ONE warning.
        df = pd.DataFrame({"entry": ["AF_TINY", "P1"],
                           "sequence": [AF_SEQ, "ACDEFGHIKLMNPQRS"]})
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                d = strp.encode_pdb(df_seq=df, pdb_folder=str(PDB_FIXTURES),
                                   features=["bfactor", "depth"])
        depth_warns = [x for x in rec if "depth" in str(x.message)
                       and "msms" in str(x.message)]
        assert len(depth_warns) == 1
        assert np.isnan(d["AF_TINY"][:, 1]).all()
        assert np.isnan(d["P1"][:, 1]).all()

    def test_depth_isolation_keeps_pdb_ok_true(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, df_out = strp.encode_pdb(return_df=True, df_seq=_df_af(),
                                           pdb_folder=str(PDB_FIXTURES),
                                           features=["bfactor", "depth"])
        assert bool(df_out["pdb_ok"].iloc[0]) is True

    def test_depth_only_all_nan_but_no_raise(self):
        # depth is the only feature -> whole tensor NaN, but still no raise.
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                                   features=["depth"])
        assert d["AF_TINY"].shape == (len(AF_SEQ), 1)
        assert np.isnan(d["AF_TINY"]).all()

    def test_depth_raises_under_on_failure_raise(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with pytest.raises(RuntimeError, match="msms"):
                strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                               features=["bfactor", "depth"], on_failure="raise")

    def test_happy_path_when_msms_present_unchanged(self):
        # msms mocked available AND encode_depth mocked to a finite block:
        # the feature composes normally, no isolation warning.
        block = (np.full((len(AF_SEQ), 1), 0.3), 1.0)
        with patch(f"{MODULE}.is_msms_available", return_value=True), \
             patch(f"{MODULE}.encode_depth", return_value=block):
            strp = aa.StructurePreprocessor(verbose=False)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                d = strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                                   features=["bfactor", "depth"])
        assert not np.isnan(d["AF_TINY"][:, 1]).any()   # depth finite
        assert not any("unavailable" in str(x.message) for x in rec)


class TestEncoderFailureIsolation:
    """A per-entry encoder RuntimeError NaNs only its own feature column."""

    def test_pdb_encoder_failure_isolated(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.encode_contact_count_8A",
                   side_effect=RuntimeError("synthetic")):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                d = strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                                   features=["plddt", "contact_count_8A"])
        v = d["AF_TINY"]
        assert not np.isnan(v[:, 0]).all()    # plddt kept
        assert np.isnan(v[:, 1]).all()        # contact_count_8A isolated
        warns = [x for x in rec if "contact_count_8A" in str(x.message)]
        assert len(warns) == 1

    def test_pdb_encoder_failure_raises_under_raise(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.encode_contact_count_8A",
                   side_effect=RuntimeError("synthetic")):
            with pytest.raises(RuntimeError, match="synthetic"):
                strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                               features=["contact_count_8A"], on_failure="raise")

    def test_pae_encoder_failure_isolated(self):
        pae = _pae_dir()
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.encode_pae_asymmetry",
                   side_effect=RuntimeError("synthetic")):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                d = strp.encode_pae(df_seq=_df_af(), pae_folder=pae,
                                   features=["pae_row_mean", "pae_asymmetry"])
        v = d["AF_TINY"]
        assert not np.isnan(v[:, 0]).all()    # pae_row_mean kept
        assert np.isnan(v[:, 1]).all()        # pae_asymmetry isolated


class TestEncodeRouterDispatch:
    """encode() routes each feature key to its owning backend."""

    def test_routes_each_key_to_right_backend(self):
        recorded = {}

        def _mk(name, val):
            def _fake(self, df_seq, *a, features, **kw):
                recorded[name] = list(features)
                return _const_sub(df_seq, features, val)
            return _fake

        feats = ["bfactor", "ss3", "pae_row_mean", "domain_boundary"]
        with patch.object(aa.StructurePreprocessor, "encode_pdb", _mk("pdb", 1.0)), \
             patch.object(aa.StructurePreprocessor, "encode_dssp", _mk("dssp", 2.0)), \
             patch.object(aa.StructurePreprocessor, "encode_pae", _mk("pae", 3.0)), \
             patch.object(aa.StructurePreprocessor, "encode_domains",
                          _mk("domains", 4.0)):
            strp = aa.StructurePreprocessor(verbose=False)
            d = strp.encode(df_seq=_df_af(), features=feats,
                           pdb_folder="x", pae_folder="y", domain_folder="z")
        assert recorded == {"pdb": ["bfactor"], "dssp": ["ss3"],
                            "pae": ["pae_row_mean"], "domains": ["domain_boundary"]}
        # Merged in requested order: bfactor(1) | ss3 x3 (2) | pae(3) | domain(4)
        v = d["AF_TINY"]
        assert v.shape == (len(AF_SEQ), 6)
        np.testing.assert_array_equal(v[0], [1., 2., 2., 2., 3., 4.])

    def test_router_matches_manual_composition(self):
        # Binary-free end-to-end: bfactor+plddt (pdb) interleaved with
        # pae_row_mean (pae) must equal manual per-encoder composition.
        pae = _pae_dir()
        strp = aa.StructurePreprocessor(verbose=False)
        feats = ["bfactor", "pae_row_mean", "plddt"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dn = strp.encode(df_seq=_df_af(), features=feats,
                            pdb_folder=str(PDB_FIXTURES), pae_folder=pae)
            d_pdb = strp.encode_pdb(df_seq=_df_af(), pdb_folder=str(PDB_FIXTURES),
                                   features=["bfactor", "plddt"])
            d_pae = strp.encode_pae(df_seq=_df_af(), pae_folder=pae,
                                   features=["pae_row_mean"])
        ref = np.concatenate([d_pdb["AF_TINY"][:, 0:1],
                              d_pae["AF_TINY"][:, 0:1],
                              d_pdb["AF_TINY"][:, 1:2]], axis=1)
        np.testing.assert_allclose(dn["AF_TINY"], ref, equal_nan=True)

    def test_router_return_df_has_encode_ok(self):
        pae = _pae_dir()
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dn, df_out = strp.encode(df_seq=_df_af(),
                                    features=["bfactor", "pae_row_mean"],
                                    pdb_folder=str(PDB_FIXTURES),
                                    pae_folder=pae, return_df=True)
        assert "encode_ok" in df_out.columns
        assert bool(df_out["encode_ok"].iloc[0]) is True
        assert df_out["entry"].tolist() == ["AF_TINY"]

    def test_router_drop_keeps_intersection(self, tmp_path):
        # AF_TINY has both pdb+pae; GONE has neither -> dropped from both.
        (tmp_path / "AF_TINY.pdb").write_text(
            (PDB_FIXTURES / "AF_TINY.pdb").read_text())
        pae = _pae_dir()
        df = pd.DataFrame({"entry": ["AF_TINY", "GONE"],
                           "sequence": [AF_SEQ, "ACDE"]})
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dn, df_out = strp.encode(df_seq=df,
                                    features=["bfactor", "pae_row_mean"],
                                    pdb_folder=str(tmp_path), pae_folder=pae,
                                    on_failure="drop", return_df=True)
        assert set(dn.keys()) == {"AF_TINY"}
        assert df_out["entry"].tolist() == ["AF_TINY"]

    def test_router_isolates_depth_across_backends(self):
        # depth (pdb, msms-gated) mixed with a pae key: depth isolates, the
        # rest are kept.
        pae = _pae_dir()
        strp = aa.StructurePreprocessor(verbose=False)
        with patch(f"{MODULE}.is_msms_available", return_value=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dn = strp.encode(df_seq=_df_af(),
                                features=["bfactor", "depth", "pae_row_mean"],
                                pdb_folder=str(PDB_FIXTURES), pae_folder=pae)
        v = dn["AF_TINY"]
        assert v.shape == (len(AF_SEQ), 3)
        assert not np.isnan(v[:, 0]).all()   # bfactor kept
        assert np.isnan(v[:, 1]).all()       # depth isolated
        assert not np.isnan(v[:, 2]).all()   # pae_row_mean kept

    # ----- negatives -----
    def test_router_missing_pae_folder(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pae_folder"):
            strp.encode(df_seq=_df_af(), features=["pae_row_mean"],
                       pdb_folder=str(PDB_FIXTURES))

    def test_router_missing_pdb_folder(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="pdb_folder"):
            strp.encode(df_seq=_df_af(), features=["bfactor"])

    def test_router_missing_domain_folder(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError, match="domain_folder"):
            strp.encode(df_seq=_df_af(), features=["domain_boundary"])

    def test_router_unknown_feature(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.encode(df_seq=_df_af(), features=["mystery"],
                       pdb_folder=str(PDB_FIXTURES))

    def test_router_empty_features(self):
        strp = aa.StructurePreprocessor(verbose=False)
        with pytest.raises(ValueError):
            strp.encode(df_seq=_df_af(), features=[],
                       pdb_folder=str(PDB_FIXTURES))

    def test_router_domain_inline_chopping_no_folder(self):
        # domain features route without a domain_folder when df_seq carries
        # an inline 'chopping' column.
        df = _df_af().copy()
        df["chopping"] = ["1-10,11-30"]
        strp = aa.StructurePreprocessor(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dn = strp.encode(df_seq=df, features=["domain_boundary"])
        assert dn["AF_TINY"].shape == (len(AF_SEQ), 1)
