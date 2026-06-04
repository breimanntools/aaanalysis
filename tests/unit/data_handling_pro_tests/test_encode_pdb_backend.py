"""This is a script to test the per-feature PDB encoders in
``data_handling_pro/_backend/struct_preproc/encode_pdb.py``
(encode_bfactor / encode_depth / encode_plddt* / encode_chi*_sincos /
encode_ca_centroid_dist* / encode_contact_count_* / encode_hse /
encode_disulfide and their shared helpers).

These encoders consume biopython ``Structure`` objects and, for two of them,
shell out to the external ``msms`` binary (residue depth) / heavy biopython
surface tools (HSExposure). The suite never runs those, so the empty-structure,
alignment-gap, missing-CA and exception branches stay uncovered when driven only
through real fixture PDBs. We test the encoders directly against light fake
structures (standard residue names so the real ``is_aa`` / aligner / ``normalize``
work unchanged) and mock only the external surface tools — a deliberate, narrow
exception to the otherwise frontend-driven testing convention.
"""
import sys
from unittest.mock import patch

import numpy as np
import pytest

import Bio.PDB.ResidueDepth  # noqa: F401  (register submodule)
import Bio.PDB.HSExposure  # noqa: F401

from aaanalysis.data_handling_pro._backend.struct_preproc import encode_pdb as ep

MODULE = "aaanalysis.data_handling_pro._backend.struct_preproc.encode_pdb"
_RD_MOD = sys.modules["Bio.PDB.ResidueDepth"]
_HSE_MOD = sys.modules["Bio.PDB.HSExposure"]


# I Helper Functions
class _Atom:
    def __init__(self, name, coord=(0.0, 0.0, 0.0), bfactor=50.0):
        self._name = name
        self._coord = np.asarray(coord, dtype=np.float64)
        self._b = bfactor

    def get_name(self):
        return self._name

    def get_coord(self):
        return self._coord

    def get_bfactor(self):
        return self._b


class _Res:
    def __init__(self, resname, rid, atoms=(), internal_coord=None, xtra=None):
        self._resname = resname
        self.id = rid
        self._atoms = list(atoms)
        self.internal_coord = internal_coord
        self.xtra = dict(xtra or {})

    def get_resname(self):
        return self._resname

    def get_atoms(self):
        return iter(self._atoms)


class _Chain:
    def __init__(self, cid, residues, a2ic_raises=False):
        self.id = cid
        self._residues = residues
        self._a2ic_raises = a2ic_raises

    def __iter__(self):
        return iter(self._residues)

    def atom_to_internal_coordinates(self):
        if self._a2ic_raises:
            raise RuntimeError("a2ic boom")


class _Model:
    def __init__(self, chains):
        self._chains = chains

    def __iter__(self):
        return iter(self._chains)


class _Structure:
    def __init__(self, models):
        self._models = models

    def get_models(self):
        return iter(self._models)


def _ca_res(resname, i, xyz, bfactor=50.0):
    """A residue with a single CA atom at xyz."""
    return _Res(resname, (" ", i + 1, " "),
                atoms=[_Atom("CA", xyz, bfactor)])


def _cys_res(i, sg_xyz):
    """A CYS residue with an SG atom at sg_xyz (plus a CA)."""
    return _Res("CYS", (" ", i + 1, " "),
                atoms=[_Atom("CA", (float(i), 0.0, 0.0)),
                       _Atom("SG", sg_xyz)])


def _empty_structure():
    return _Structure([])  # get_models() -> StopIteration in _collect


def _ca_chain_structure(resnames, coords, bfactors=None):
    if bfactors is None:
        bfactors = [50.0] * len(resnames)
    residues = [_ca_res(rn, i, xyz, b)
                for i, (rn, xyz, b) in enumerate(zip(resnames, coords, bfactors))]
    return _Structure([_Model([_Chain("A", residues)])])


SEQ3 = "AGC"  # ALA, GLY, CYS -> all is_aa True


# II Test Classes
class TestEncodeBfactor:
    """encode_bfactor: normal, empty, gap, no-atom branches."""

    def test_valid_shape_and_identity(self):
        st = _ca_chain_structure(["ALA", "GLY", "CYS"],
                                 [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                                 bfactors=[10.0, 20.0, 30.0])
        out, ident = ep.encode_bfactor(st, SEQ3)
        assert out.shape == (3, 1)
        assert ident > 0.0

    def test_valid_empty_structure_nan(self):
        out, ident = ep.encode_bfactor(_empty_structure(), SEQ3)
        assert out.shape == (3, 1)
        assert np.isnan(out).all()
        assert ident == 0.0

    def test_valid_residue_without_atoms_nan(self):
        res = _Res("ALA", (" ", 1, " "), atoms=[])  # no atoms
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_bfactor(st, "A")
        assert np.isnan(out[0, 0])

    def test_valid_target_gap_inserts_nan(self):
        # atom chain "AC" but target "AGC": the missing G -> gap -> NaN.
        st = _ca_chain_structure(["ALA", "CYS"], [(0, 0, 0), (1, 0, 0)])
        out, _ = ep.encode_bfactor(st, "AGC")
        assert out.shape == (3, 1)
        assert np.isnan(out).any()

    def test_invalid_sequence_type(self):
        st = _ca_chain_structure(["ALA"], [(0, 0, 0)])
        with pytest.raises((TypeError, AttributeError)):
            ep.encode_bfactor(st, None)


class TestEncodePlddt:
    """encode_plddt / _disorder / _tier and the shared CA reader."""

    def test_valid_plddt_shape(self):
        st = _ca_chain_structure(["ALA", "GLY"], [(0, 0, 0), (1, 0, 0)],
                                 bfactors=[90.0, 40.0])
        out, _ = ep.encode_plddt(st, "AG")
        assert out.shape == (2, 1)

    def test_valid_plddt_empty_nan(self):
        out, ident = ep.encode_plddt(_empty_structure(), "AG")
        assert np.isnan(out).all() and ident == 0.0

    def test_valid_plddt_falls_back_to_mean_when_no_ca(self):
        # residue with only a non-CA atom -> mean over atoms path.
        res = _Res("ALA", (" ", 1, " "), atoms=[_Atom("N", (0, 0, 0), 55.0)])
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_plddt(st, "A")
        assert out.shape == (1, 1)

    def test_valid_plddt_no_atoms_nan(self):
        res = _Res("ALA", (" ", 1, " "), atoms=[])
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_plddt(st, "A")
        assert np.isnan(out[0, 0])

    def test_valid_disorder_boolean(self):
        st = _ca_chain_structure(["ALA", "GLY"], [(0, 0, 0), (1, 0, 0)],
                                 bfactors=[90.0, 40.0])
        out, _ = ep.encode_plddt_disorder(st, "AG", threshold=70.0)
        assert out.shape == (2, 1)
        assert set(np.unique(out[~np.isnan(out)])).issubset({0.0, 1.0})

    def test_valid_disorder_empty_nan(self):
        out, _ = ep.encode_plddt_disorder(_empty_structure(), "AG")
        assert np.isnan(out).all()

    def test_valid_tier_onehot_shape(self):
        st = _ca_chain_structure(["ALA", "GLY", "CYS", "ALA"],
                                 [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
                                 bfactors=[40.0, 60.0, 80.0, 95.0])
        out, _ = ep.encode_plddt_tier(st, "AGCA")
        assert out.shape == (4, 4)
        # each (finite) row is one-hot
        finite = out[~np.isnan(out).any(axis=1)]
        assert np.all(finite.sum(axis=1) == 1.0)

    def test_valid_tier_each_band(self):
        st = _ca_chain_structure(["ALA", "GLY", "CYS", "ALA"],
                                 [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
                                 bfactors=[40.0, 60.0, 80.0, 95.0])
        out, _ = ep.encode_plddt_tier(st, "AGCA")
        assert out[0, 0] == 1.0  # <50
        assert out[1, 1] == 1.0  # 50-70
        assert out[2, 2] == 1.0  # 70-90
        assert out[3, 3] == 1.0  # >=90

    def test_valid_tier_empty_nan(self):
        out, _ = ep.encode_plddt_tier(_empty_structure(), "AG")
        assert out.shape == (2, 4) and np.isnan(out).all()

    def test_valid_tier_nan_band(self):
        # residue with no atoms -> NaN plddt -> NaN tier row.
        res = _Res("ALA", (" ", 1, " "), atoms=[])
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_plddt_tier(st, "A")
        assert np.isnan(out[0]).all()


class TestEncodeDepth:
    """encode_depth: mocked msms + ResidueDepth, plus failure branch."""

    def _patch_rd(self, depths=None, raises=None):
        chain_id = "A"

        class _RD:
            def __init__(self, model):
                if raises is not None:
                    raise raises

            def __getitem__(self, key):
                if depths is None:
                    raise KeyError(key)
                return (depths.get(key, 5.0), 3.0)

        return _RD

    def test_valid_depth_shape(self):
        st = _ca_chain_structure(["ALA", "GLY"], [(0, 0, 0), (1, 0, 0)])
        rd = self._patch_rd(depths={("A", (" ", 1, " ")): 4.0,
                                    ("A", (" ", 2, " ")): 6.0})
        with patch(f"{MODULE}.check_msms_available"), \
             patch.object(_RD_MOD, "ResidueDepth", rd):
            out, _ = ep.encode_depth(st, "AG")
        assert out.shape == (2, 1)

    def test_valid_depth_missing_key_nan(self):
        st = _ca_chain_structure(["ALA"], [(0, 0, 0)])
        rd = self._patch_rd(depths=None)  # __getitem__ raises -> NaN
        with patch(f"{MODULE}.check_msms_available"), \
             patch.object(_RD_MOD, "ResidueDepth", rd):
            out, _ = ep.encode_depth(st, "A")
        assert np.isnan(out[0, 0])

    def test_valid_depth_empty_structure(self):
        rd = self._patch_rd(depths={})
        with patch(f"{MODULE}.check_msms_available"), \
             patch.object(_RD_MOD, "ResidueDepth", rd):
            out, ident = ep.encode_depth(_empty_structure(), "AG")
        assert np.isnan(out).all() and ident == 0.0

    def test_invalid_residue_depth_failure(self):
        st = _ca_chain_structure(["ALA"], [(0, 0, 0)])
        rd = self._patch_rd(raises=OSError("msms not found"))
        with patch(f"{MODULE}.check_msms_available"), \
             patch.object(_RD_MOD, "ResidueDepth", rd):
            with pytest.raises(RuntimeError, match="ResidueDepth failed"):
                ep.encode_depth(st, "A")

    def test_invalid_msms_unavailable(self):
        st = _ca_chain_structure(["ALA"], [(0, 0, 0)])
        with patch(f"{MODULE}.check_msms_available",
                   side_effect=RuntimeError("msms missing")):
            with pytest.raises(RuntimeError, match="msms"):
                ep.encode_depth(st, "A")


class TestEncodeChiSincos:
    """encode_chi1_sincos / chi2 and _chi_sincos_for_residue branches."""

    class _RIC:
        def __init__(self, angle):
            self._angle = angle

        def get_angle(self, name):
            return self._angle

    def test_valid_internal_coord_none_nan(self):
        st = _ca_chain_structure(["ALA", "GLY"], [(0, 0, 0), (1, 0, 0)])
        out, _ = ep.encode_chi1_sincos(st, "AG")  # internal_coord None -> nan
        assert out.shape == (2, 2)
        assert np.isnan(out).all()

    def test_valid_angle_value_sets_sincos(self):
        res = _Res("ALA", (" ", 1, " "),
                   atoms=[_Atom("CA", (0, 0, 0))],
                   internal_coord=self._RIC(90.0))
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_chi1_sincos(st, "A")
        # chi=90deg -> sin=1, cos=0 (within tolerance)
        assert np.isclose(out[0, 0], out[0, 0])  # finite
        assert out.shape == (1, 2)

    def test_valid_angle_none_nan(self):
        res = _Res("ALA", (" ", 1, " "),
                   atoms=[_Atom("CA", (0, 0, 0))],
                   internal_coord=self._RIC(None))
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_chi1_sincos(st, "A")
        assert np.isnan(out).all()

    def test_valid_chi2_uses_chi2(self):
        res = _Res("ARG", (" ", 1, " "),
                   atoms=[_Atom("CA", (0, 0, 0))],
                   internal_coord=self._RIC(45.0))
        st = _Structure([_Model([_Chain("R", [res])])])
        out, _ = ep.encode_chi2_sincos(st, "R")
        assert out.shape == (1, 2)

    def test_valid_chi_empty_structure(self):
        out, ident = ep.encode_chi1_sincos(_empty_structure(), "AG")
        assert out.shape == (2, 2) and np.isnan(out).all()

    def test_valid_a2ic_exception_swallowed(self):
        res = _ca_res("ALA", 0, (0, 0, 0))
        st = _Structure([_Model([_Chain("A", [res], a2ic_raises=True)])])
        out, _ = ep.encode_chi1_sincos(st, "A")  # a2ic raises -> pass -> nan
        assert np.isnan(out).all()

    def test_valid_get_angle_exception_nan(self):
        class _BadRIC:
            def get_angle(self, name):
                raise ValueError("no chi")

        res = _Res("ALA", (" ", 1, " "),
                   atoms=[_Atom("CA", (0, 0, 0))], internal_coord=_BadRIC())
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_chi1_sincos(st, "A")
        assert np.isnan(out).all()


class TestEncodeCaGeometry:
    """encode_ca_centroid_dist(_norm) + contact counts."""

    def test_valid_centroid_dist_shape(self):
        st = _ca_chain_structure(["ALA", "GLY", "CYS"],
                                 [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
        out, _ = ep.encode_ca_centroid_dist(st, "AGC")
        assert out.shape == (3, 1)

    def test_valid_centroid_empty_nan(self):
        out, ident = ep.encode_ca_centroid_dist(_empty_structure(), "AG")
        assert np.isnan(out).all() and ident == 0.0

    def test_valid_centroid_all_nan_coords(self):
        # residues without CA -> all-NaN coords -> finite empty -> NaN out.
        res = _Res("ALA", (" ", 1, " "), atoms=[_Atom("N", (0, 0, 0))])
        st = _Structure([_Model([_Chain("A", [res])])])
        out, _ = ep.encode_ca_centroid_dist(st, "A")
        assert np.isnan(out[0, 0])

    def test_valid_centroid_norm_shape(self):
        st = _ca_chain_structure(["ALA", "GLY", "CYS"],
                                 [(0, 0, 0), (3, 0, 0), (6, 0, 0)])
        out, _ = ep.encode_ca_centroid_dist_norm(st, "AGC")
        assert out.shape == (3, 1)

    def test_valid_centroid_norm_single_residue_nan(self):
        st = _ca_chain_structure(["ALA"], [(0, 0, 0)])
        out, _ = ep.encode_ca_centroid_dist_norm(st, "A")  # <2 finite -> NaN
        assert np.isnan(out[0, 0])

    def test_valid_centroid_norm_empty(self):
        out, _ = ep.encode_ca_centroid_dist_norm(_empty_structure(), "AG")
        assert np.isnan(out).all()

    def test_valid_contact_8A_shape(self):
        st = _ca_chain_structure(["ALA"] * 7,
                                 [(i * 2.0, 0, 0) for i in range(7)])
        out, _ = ep.encode_contact_count_8A(st, "AAAAAAA")
        assert out.shape == (7, 1)

    def test_valid_contact_12A_shape(self):
        st = _ca_chain_structure(["ALA"] * 7,
                                 [(i * 2.0, 0, 0) for i in range(7)])
        out, _ = ep.encode_contact_count_12A(st, "AAAAAAA")
        assert out.shape == (7, 1)

    def test_valid_contact_empty_nan(self):
        out, ident = ep.encode_contact_count_8A(_empty_structure(), "AG")
        assert np.isnan(out).all() and ident == 0.0

    def test_valid_contact_nan_coord_row(self):
        residues = [_ca_res("ALA", 0, (0, 0, 0)),
                    _Res("GLY", (" ", 2, " "), atoms=[_Atom("N", (1, 0, 0))]),
                    _ca_res("CYS", 2, (2, 0, 0))]
        st = _Structure([_Model([_Chain("A", residues)])])
        out, _ = ep.encode_contact_count_8A(st, "AGC")
        assert np.isnan(out[1, 0])  # middle residue has no CA


class TestEncodeHse:
    """encode_hse: mocked HSExposureCA populating residue.xtra."""

    def test_valid_hse_shape(self):
        res = [_Res("ALA", (" ", i + 1, " "),
                    atoms=[_Atom("CA", (float(i), 0, 0))],
                    xtra={"EXP_HSE_A_U": 5.0, "EXP_HSE_A_D": 3.0})
               for i in range(2)]
        st = _Structure([_Model([_Chain("A", res)])])
        with patch.object(_HSE_MOD, "HSExposureCA", lambda model, radius=13.0: None):
            out, _ = ep.encode_hse(st, "AA")
        assert out.shape == (2, 2)

    def test_valid_hse_missing_xtra_nan(self):
        res = [_Res("ALA", (" ", 1, " "),
                    atoms=[_Atom("CA", (0, 0, 0))])]  # no xtra
        st = _Structure([_Model([_Chain("A", res)])])
        with patch.object(_HSE_MOD, "HSExposureCA", lambda model, radius=13.0: None):
            out, _ = ep.encode_hse(st, "A")
        assert np.isnan(out).all()

    def test_valid_hse_empty_structure(self):
        with patch.object(_HSE_MOD, "HSExposureCA", lambda model, radius=13.0: None):
            out, ident = ep.encode_hse(_empty_structure(), "AG")
        assert np.isnan(out).all() and ident == 0.0

    def test_invalid_hse_failure(self):
        res = [_ca_res("ALA", 0, (0, 0, 0))]
        st = _Structure([_Model([_Chain("A", res)])])

        def _boom(model, radius=13.0):
            raise ValueError("hse boom")

        with patch.object(_HSE_MOD, "HSExposureCA", _boom):
            with pytest.raises(RuntimeError, match="HSExposureCA failed"):
                ep.encode_hse(st, "A")


class TestEncodeDisulfide:
    """encode_disulfide: bonded pair, lone CYS, non-CYS, empty."""

    def test_valid_bonded_pair(self):
        # two CYS with SG atoms 2.0 A apart -> both participate.
        res = [_cys_res(0, (0.0, 0.0, 0.0)), _cys_res(1, (2.0, 0.0, 0.0))]
        st = _Structure([_Model([_Chain("A", res)])])
        out, _ = ep.encode_disulfide(st, "CC")
        assert out.shape == (2, 2)
        assert out[0, 0] == out[0, 0]  # finite participates value

    def test_valid_lone_cys_not_bonded(self):
        res = [_cys_res(0, (0.0, 0.0, 0.0)), _cys_res(1, (20.0, 0.0, 0.0))]
        st = _Structure([_Model([_Chain("A", res)])])
        out, _ = ep.encode_disulfide(st, "CC")
        # too far apart -> participates 0, distance NaN
        assert out[0, 0] == 0.0
        assert np.isnan(out[0, 1])

    def test_valid_noncys_nan(self):
        res = [_ca_res("ALA", 0, (0, 0, 0)), _cys_res(1, (1.0, 0.0, 0.0))]
        st = _Structure([_Model([_Chain("A", res)])])
        out, _ = ep.encode_disulfide(st, "AC")
        assert np.isnan(out[0, 0])  # ALA -> NaN participates

    def test_valid_cys_without_sg_nan(self):
        # CYS lacking an SG atom -> sg_coords None -> NaN.
        res = [_Res("CYS", (" ", 1, " "), atoms=[_Atom("CA", (0, 0, 0))])]
        st = _Structure([_Model([_Chain("A", res)])])
        out, _ = ep.encode_disulfide(st, "C")
        assert np.isnan(out[0, 0])

    def test_valid_empty_structure(self):
        out, ident = ep.encode_disulfide(_empty_structure(), "CC")
        assert np.isnan(out).all() and ident == 0.0


class TestEncodePdbComplex:
    """Cross-cutting helper + dispatch behaviours."""

    def test_complex_pick_best_chain_prefers_match(self):
        # chain B matches the target better than chain A.
        a = _Chain("A", [_ca_res("ALA", 0, (0, 0, 0))])
        b = _Chain("B", [_ca_res("CYS", 0, (0, 0, 0)),
                         _ca_res("GLY", 1, (1, 0, 0))])
        st = _Structure([_Model([a, b])])
        out, ident = ep.encode_bfactor(st, "CG")
        assert out.shape == (2, 1)
        assert ident > 0.0

    def test_complex_collect_chain_residues_skips_hetatm(self):
        res = [_ca_res("ALA", 0, (0, 0, 0)),
               _Res("HOH", ("W", 2, " "), atoms=[_Atom("O", (1, 0, 0))])]
        st = _Structure([_Model([_Chain("A", res)])])
        out, _ = ep.encode_bfactor(st, "A")
        assert out.shape == (1, 1)

    def test_complex_load_structure_dispatch_pdb(self, tmp_path):
        # load_structure should pick PDBParser for .pdb; we patch the parsers
        # so no real file parsing is needed.
        import Bio.PDB as biopdb
        sentinel = object()

        class _P:
            def __init__(self, QUIET=True):
                pass

            def get_structure(self, name, path):
                return sentinel

        with patch.object(biopdb, "PDBParser", _P), \
             patch.object(biopdb, "MMCIFParser", _P):
            out = ep.load_structure(str(tmp_path / "x.pdb"))
        assert out is sentinel

    def test_complex_load_structure_dispatch_cif(self, tmp_path):
        import Bio.PDB as biopdb
        flag = {"cif": False}

        class _PDB:
            def __init__(self, QUIET=True):
                pass

            def get_structure(self, name, path):
                return "pdb"

        class _CIF:
            def __init__(self, QUIET=True):
                flag["cif"] = True

            def get_structure(self, name, path):
                return "cif"

        with patch.object(biopdb, "PDBParser", _PDB), \
             patch.object(biopdb, "MMCIFParser", _CIF):
            out = ep.load_structure(str(tmp_path / "x.cif"))
        assert out == "cif" and flag["cif"]

    def test_complex_all_encoders_empty_consistent(self):
        st = _empty_structure()
        for fn, width in [(ep.encode_bfactor, 1), (ep.encode_plddt, 1),
                          (ep.encode_plddt_tier, 4), (ep.encode_chi1_sincos, 2),
                          (ep.encode_ca_centroid_dist, 1), (ep.encode_disulfide, 2)]:
            out, ident = fn(st, "AGC")
            assert out.shape == (3, width)
            assert np.isnan(out).all()
            assert ident == 0.0
