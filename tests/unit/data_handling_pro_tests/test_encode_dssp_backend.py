"""This is a script to test the DSSP per-feature encoders in
``data_handling_pro/_backend/struct_preproc/encode_dssp.py``
(encode_ss / encode_rasa / encode_dihedrals_sincos / encode_hbond_* and the
_ss3/_ss8 index + _safe_float + _stack_offset_energy helpers).

These are pure-Python encoders that turn DSSP per-residue lists into normalized
``(L, D)`` tensors. The frontend test covers the orchestration; here we test the
encoders directly to reach the NaN/gap/length-mismatch branches deterministically.
"""
import math

import numpy as np
import pytest

import aaanalysis.utils as ut
from aaanalysis.data_handling_pro._backend.struct_preproc.encode_dssp import (
    encode_ss,
    encode_rasa,
    encode_dihedrals_sincos,
    encode_hbond_donor,
    encode_hbond_acceptor,
    _safe_float,
    _ss3_index_for_code,
    _ss8_index_for_code,
)

GAP = ut.STR_SS_GAP  # '-'


# II Test Classes
class TestSafeFloat:
    """_safe_float: numeric passthrough, None / garbage / non-finite -> NaN."""

    def test_valid_number(self):
        assert _safe_float(2.5) == 2.5

    def test_valid_int_string(self):
        assert _safe_float("4") == 4.0

    def test_invalid_none(self):
        assert math.isnan(_safe_float(None))

    def test_invalid_garbage(self):
        assert math.isnan(_safe_float("xyz"))

    def test_invalid_inf(self):
        assert math.isnan(_safe_float(float("inf")))

    def test_invalid_nan_passthrough(self):
        assert math.isnan(_safe_float(float("nan")))


class TestSsIndexHelpers:
    """_ss3_index_for_code / _ss8_index_for_code branches."""

    def test_valid_ss3_helix_family(self):
        assert _ss3_index_for_code("G") == 0  # G -> H
        assert _ss3_index_for_code("H") == 0

    def test_valid_ss3_strand_family(self):
        assert _ss3_index_for_code("B") == 1  # B -> E

    def test_valid_ss3_coil_family(self):
        assert _ss3_index_for_code("T") == 2  # T -> C

    def test_valid_ss3_gap_none(self):
        assert _ss3_index_for_code(GAP) is None

    def test_valid_ss8_blank_space(self):
        assert _ss8_index_for_code(" ") == _ss8_index_for_code(" ")  # blank col
        assert _ss8_index_for_code(" ") is not None

    def test_valid_ss8_gap_none(self):
        assert _ss8_index_for_code(GAP) is None

    def test_valid_ss8_unknown_none(self):
        assert _ss8_index_for_code("Z") is None

    def test_valid_ss8_known_code(self):
        assert _ss8_index_for_code("H") == 0


class TestEncodeSs:
    """encode_ss: ss3 / ss8 one-hot + gap NaN rows."""

    def test_valid_ss3_shape(self):
        out = encode_ss(["H", "E", "C"], "ss3")
        assert out.shape == (3, 1) or out.shape == (3, 3)

    def test_valid_ss3_gap_nan_row(self):
        out = encode_ss(["H", GAP, "C"], "ss3")
        assert np.isnan(out[1]).all()

    def test_valid_ss8_shape(self):
        out = encode_ss(["H", "B", "E", "G", "I", "T", "S", " "], "ss8")
        assert out.shape[0] == 8

    def test_valid_ss8_gap_nan_row(self):
        out = encode_ss(["H", GAP], "ss8")
        assert np.isnan(out[1]).all()

    def test_valid_ss8_unknown_nan_row(self):
        out = encode_ss(["H", "Z"], "ss8")
        assert np.isnan(out[1]).all()

    def test_valid_ss8_blank_space_onehot(self):
        out = encode_ss([" "], "ss8")
        assert not np.isnan(out[0]).all()  # blank column set


class TestEncodeRasa:
    """encode_rasa: division by max ASA, NaN propagation, length check."""

    def test_valid_shape(self):
        out = encode_rasa([60.0, 120.0, 30.0], "ARN")
        assert out.shape == (3, 1)

    def test_valid_none_value_nan(self):
        out = encode_rasa([None, 120.0], "AR")
        assert np.isnan(out[0, 0])

    def test_valid_unknown_aa_nan(self):
        out = encode_rasa([60.0, 60.0], "AX")  # X has no MAX_ASA -> NaN
        assert np.isnan(out[1, 0])

    def test_valid_within_unit_range(self):
        out = encode_rasa([60.0], "A")
        v = out[0, 0]
        assert 0.0 <= v <= 1.0

    def test_invalid_length_mismatch(self):
        with pytest.raises(RuntimeError, match="length mismatch"):
            encode_rasa([60.0, 120.0], "A")


class TestEncodeDihedrals:
    """encode_dihedrals_sincos: sin/cos pairs + NaN + length check."""

    def test_valid_shape(self):
        out = encode_dihedrals_sincos([-60.0, -45.0], [-45.0, -50.0])
        assert out.shape == (2, 4)

    def test_valid_phi_nan_propagates(self):
        out = encode_dihedrals_sincos([None, -45.0], [-45.0, -50.0])
        assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])

    def test_valid_psi_nan_propagates(self):
        out = encode_dihedrals_sincos([-60.0, -45.0], [None, -50.0])
        assert np.isnan(out[0, 2]) and np.isnan(out[0, 3])

    def test_valid_finite_values_in_range(self):
        out = encode_dihedrals_sincos([90.0], [0.0])
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_invalid_length_mismatch(self):
        with pytest.raises(RuntimeError, match="length mismatch"):
            encode_dihedrals_sincos([-60.0, -45.0], [-45.0])


class TestEncodeHbond:
    """encode_hbond_donor / acceptor + _stack_offset_energy length check."""

    def test_valid_donor_shape(self):
        out = encode_hbond_donor([-4, 0], [-2.0, 0.0])
        assert out.shape == (2, 2)

    def test_valid_acceptor_shape(self):
        out = encode_hbond_acceptor([4, 0], [-2.0, 0.0])
        assert out.shape == (2, 2)

    def test_valid_donor_nan_entries(self):
        out = encode_hbond_donor([None], [None])
        assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])

    def test_invalid_donor_length_mismatch(self):
        with pytest.raises(RuntimeError, match="length mismatch"):
            encode_hbond_donor([-4, 0], [-2.0])

    def test_invalid_acceptor_length_mismatch(self):
        with pytest.raises(RuntimeError, match="length mismatch"):
            encode_hbond_acceptor([4], [-2.0, 0.0])


class TestEncodeDsspComplex:
    """Cross-cutting combinations across the DSSP encoders."""

    def test_complex_ss3_all_families(self):
        out = encode_ss(["H", "G", "I", "E", "B", "T", "S", " ", "-"], "ss3")
        # last is a gap -> NaN row; the rest one-hot.
        assert np.isnan(out[-1]).all()
        assert not np.isnan(out[:-1]).any()

    def test_complex_rasa_mixed_nan_and_value(self):
        out = encode_rasa([None, 60.0, 120.0], "AAR")
        assert np.isnan(out[0, 0])
        assert not np.isnan(out[1, 0])

    def test_complex_dihedrals_both_nan(self):
        out = encode_dihedrals_sincos([None], [None])
        assert np.isnan(out[0]).all()

    def test_complex_hbond_donor_vs_acceptor_same_recipe_shape(self):
        d = encode_hbond_donor([-4, -3], [-2.0, -1.5])
        a = encode_hbond_acceptor([4, 3], [-2.0, -1.5])
        assert d.shape == a.shape == (2, 2)

    def test_complex_ss8_full_alphabet_plus_gap(self):
        codes = ["H", "B", "E", "G", "I", "T", "S", " ", "-", "Z"]
        out = encode_ss(codes, "ss8")
        assert np.isnan(out[-1]).all()   # 'Z' unknown -> NaN
        assert np.isnan(out[-2]).all()   # '-' gap -> NaN
