"""Branch-coverage tests for NumericalFeature public methods.

Targets the previously-uncovered guard arms of extend_alphabet (new letter
already present) and get_parts (dict_num tensor with zero feature dimension),
reached exclusively through the public ``aa.NumericalFeature`` API.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some
import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

NF = aa.NumericalFeature


# ======================================================================================
# extend_alphabet — new letter already in the alphabet
# ======================================================================================
class TestExtendAlphabetExistingLetter:
    """The duplicate-letter guard fires when new_letter is already an index entry."""

    @settings(max_examples=5, deadline=None)
    @given(letter=some.sampled_from(["A", "C", "D", "E", "G", "K", "L"]))
    def test_existing_letter_raises(self, letter):
        df_scales = aa.load_scales()
        nf = aa.NumericalFeature()
        with pytest.raises(ValueError, match="already exists"):
            nf.extend_alphabet(df_scales=df_scales, new_letter=letter)

    def test_new_letter_succeeds(self):
        df_scales = aa.load_scales()
        nf = aa.NumericalFeature()
        out = nf.extend_alphabet(df_scales=df_scales, new_letter="B")
        assert "B" in out.index
        assert len(out) == len(df_scales) + 1


# ======================================================================================
# get_parts — dict_num tensor with D == 0
# ======================================================================================
class TestGetPartsZeroDim:
    """A (L, 0) per-residue tensor is rejected with the D=0 guard."""

    def _df_seq(self):
        return pd.DataFrame({
            "entry": ["P1"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"],
            "tmd_start": [3],
            "tmd_stop": [12],
        })

    def test_zero_dim_tensor_raises(self):
        nf = aa.NumericalFeature()
        df_seq = self._df_seq()
        dict_num = {"P1": np.zeros((20, 0))}
        with pytest.raises(ValueError, match="D=0"):
            nf.get_parts(df_seq=df_seq, dict_num=dict_num)

    def test_one_dim_tensor_succeeds(self):
        nf = aa.NumericalFeature()
        df_seq = self._df_seq()
        dict_num = {"P1": np.arange(20, dtype=float).reshape(20, 1)}
        df_parts, dict_num_parts = nf.get_parts(df_seq=df_seq, dict_num=dict_num)
        assert isinstance(df_parts, pd.DataFrame)
        assert all(v.shape[-1] == 1 for v in dict_num_parts.values())
