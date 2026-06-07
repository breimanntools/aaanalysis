"""This is a script to test combine_dict_nums()."""
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

aa.options["verbose"] = False

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


# I Helper Functions
def _make_dict_num(entries=("P1", "P2"), L_map=None, D=4, fill=1.0):
    """Build a simple ``{entry: ones((L, D))}``."""
    L_map = L_map or {"P1": 10, "P2": 5}
    return {e: np.full((L_map[e], D), fill, dtype=np.float64)
            for e in entries}


# II Test Classes
class TestCombineDictNums:
    """Single-parameter normal + invalid cases for combine_dict_nums."""

    # ----- NEGATIVES (≥10) -----
    def test_invalid_none(self):
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=None)

    def test_invalid_empty_list(self):
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[])

    def test_invalid_non_list_input(self):
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums={"P1": np.zeros((5, 3))})

    def test_invalid_list_item_not_dict(self):
        with pytest.raises(ValueError):
            aa.combine_dict_nums(
                dict_nums=[_make_dict_num(), [1, 2, 3]])

    def test_invalid_entry_sets_diverge(self):
        d1 = _make_dict_num(entries=("P1", "P2"),
                            L_map={"P1": 10, "P2": 5})
        d2 = _make_dict_num(entries=("P1", "P3"),
                            L_map={"P1": 10, "P3": 5})
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    def test_invalid_extra_entry_in_second(self):
        d1 = _make_dict_num(entries=("P1",))
        d2 = _make_dict_num(entries=("P1", "P2"))
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    def test_invalid_per_entry_L_mismatch(self):
        d1 = _make_dict_num()
        d2 = _make_dict_num(L_map={"P1": 9, "P2": 5})  # P1 wrong L
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    def test_invalid_array_value_not_ndarray(self):
        d1 = _make_dict_num(entries=("P1",))
        d2 = {"P1": [[1, 2], [3, 4]]}
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    def test_invalid_array_value_not_2d(self):
        d1 = _make_dict_num(entries=("P1",))
        d2 = {"P1": np.zeros(10)}  # 1D
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    def test_invalid_array_value_3d(self):
        d1 = _make_dict_num(entries=("P1",))
        d2 = {"P1": np.zeros((10, 3, 2))}
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    # ----- POSITIVES (≥10) -----
    def test_valid_single_input_passes_through(self):
        d = _make_dict_num()
        out = aa.combine_dict_nums(dict_nums=[d])
        assert set(out.keys()) == set(d.keys())
        np.testing.assert_array_equal(out["P1"], d["P1"])

    def test_valid_two_inputs_concat_D(self):
        d1 = _make_dict_num(D=3, fill=1.0)
        d2 = _make_dict_num(D=2, fill=2.0)
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert out["P1"].shape == (10, 5)
        assert out["P2"].shape == (5, 5)

    def test_valid_three_inputs_concat_D(self):
        d1 = _make_dict_num(D=3, fill=0.5)
        d2 = _make_dict_num(D=2, fill=1.0)
        d3 = _make_dict_num(D=4, fill=-1.0)
        out = aa.combine_dict_nums(dict_nums=[d1, d2, d3])
        assert out["P1"].shape == (10, 9)

    def test_valid_concat_preserves_per_entry_L(self):
        d1 = _make_dict_num(D=3)
        d2 = _make_dict_num(D=4)
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert out["P1"].shape[0] == 10
        assert out["P2"].shape[0] == 5

    def test_valid_concat_values_ordered(self):
        d1 = _make_dict_num(D=2, fill=1.0)
        d2 = _make_dict_num(D=2, fill=2.0)
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        np.testing.assert_array_equal(out["P1"][:, :2], 1.0)
        np.testing.assert_array_equal(out["P1"][:, 2:], 2.0)

    def test_valid_entry_set_preserved(self):
        d1 = _make_dict_num(entries=("P1", "P2", "P3"),
                            L_map={"P1": 4, "P2": 7, "P3": 2})
        d2 = _make_dict_num(entries=("P1", "P2", "P3"),
                            L_map={"P1": 4, "P2": 7, "P3": 2})
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert set(out.keys()) == {"P1", "P2", "P3"}

    def test_valid_returns_dict(self):
        out = aa.combine_dict_nums(dict_nums=[_make_dict_num()])
        assert isinstance(out, dict)

    def test_valid_output_arrays_are_ndarray(self):
        out = aa.combine_dict_nums(dict_nums=[_make_dict_num()])
        for v in out.values():
            assert isinstance(v, np.ndarray)

    def test_valid_dtype_promotion(self):
        # int32 + float64 should yield float64 (numpy default for concat).
        d1 = {"P1": np.ones((4, 2), dtype=np.int32)}
        d2 = {"P1": np.ones((4, 2), dtype=np.float64)}
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert out["P1"].dtype == np.float64

    def test_valid_nan_propagates(self):
        d1 = _make_dict_num(D=2, fill=1.0)
        d2 = {"P1": np.full((10, 1), np.nan),
              "P2": np.full((5, 1), np.nan)}
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert np.isnan(out["P1"][:, 2]).all()
        assert not np.isnan(out["P1"][:, :2]).any()


class TestCombineDictNumsComplex:
    """Cross-parameter combinations for combine_dict_nums."""

    def test_complex_single_entry_three_inputs(self):
        d1 = {"P1": np.zeros((5, 3))}
        d2 = {"P1": np.ones((5, 4))}
        d3 = {"P1": np.full((5, 2), 7.0)}
        out = aa.combine_dict_nums(dict_nums=[d1, d2, d3])
        assert out["P1"].shape == (5, 9)
        np.testing.assert_array_equal(out["P1"][:, 0:3], 0.0)
        np.testing.assert_array_equal(out["P1"][:, 3:7], 1.0)
        np.testing.assert_array_equal(out["P1"][:, 7:9], 7.0)

    def test_complex_disjoint_per_entry_L(self):
        d1 = {"A": np.zeros((3, 2)), "B": np.zeros((7, 2))}
        d2 = {"A": np.ones((3, 5)), "B": np.ones((7, 5))}
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert out["A"].shape == (3, 7)
        assert out["B"].shape == (7, 7)

    def test_complex_mismatched_L_one_entry_only(self):
        d1 = {"A": np.zeros((3, 2)), "B": np.zeros((7, 2))}
        d2 = {"A": np.zeros((3, 2)), "B": np.zeros((6, 2))}  # B mismatched
        with pytest.raises(ValueError):
            aa.combine_dict_nums(dict_nums=[d1, d2])

    def test_complex_mixed_dtypes_concat(self):
        d1 = {"P1": np.ones((4, 2), dtype=np.float32)}
        d2 = {"P1": np.ones((4, 2), dtype=np.float64)}
        d3 = {"P1": np.ones((4, 2), dtype=np.int16)}
        out = aa.combine_dict_nums(dict_nums=[d1, d2, d3])
        assert out["P1"].shape == (4, 6)

    def test_complex_concat_is_deterministic(self):
        d1 = _make_dict_num(D=3, fill=1.0)
        d2 = _make_dict_num(D=2, fill=2.0)
        a = aa.combine_dict_nums(dict_nums=[d1, d2])
        b = aa.combine_dict_nums(dict_nums=[d1, d2])
        np.testing.assert_array_equal(a["P1"], b["P1"])

    def test_complex_three_proteins_two_sources(self):
        d1 = _make_dict_num(entries=("A", "B", "C"),
                            L_map={"A": 2, "B": 3, "C": 4}, D=3)
        d2 = _make_dict_num(entries=("A", "B", "C"),
                            L_map={"A": 2, "B": 3, "C": 4}, D=2)
        out = aa.combine_dict_nums(dict_nums=[d1, d2])
        assert {k: v.shape for k, v in out.items()} == {
            "A": (2, 5), "B": (3, 5), "C": (4, 5)}
