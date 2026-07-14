"""This is a script to test AAPred.score_to_group()."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
import hypothesis.strategies as some

import aaanalysis as aa

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")

STG = aa.AAPred.score_to_group


class TestScoreToGroup:
    """Normal cases, one parameter per test."""

    # scores
    @settings(max_examples=10, deadline=None)
    @given(vals=some.lists(some.floats(min_value=0, max_value=100, allow_nan=False), min_size=1, max_size=20))
    def test_scores_list(self, vals):
        g = STG(vals, thresholds=[50], labels=["lo", "hi"])
        assert isinstance(g, pd.Series)
        assert len(g) == len(vals)

    def test_scores_numpy(self):
        g = STG(np.array([10.0, 60.0]), thresholds=[50], labels=["lo", "hi"])
        assert g.tolist() == ["lo", "hi"]

    def test_scores_series_preserves_index(self):
        s = pd.Series([10.0, 60.0], index=["p1", "p2"])
        g = STG(s, thresholds=[50], labels=["lo", "hi"])
        assert list(g.index) == ["p1", "p2"]

    def test_scores_nan_is_missing(self):
        g = STG([np.nan, 60.0], thresholds=[50], labels=["lo", "hi"])
        assert pd.isna(g.iloc[0]) and g.iloc[1] == "hi"

    def test_scores_none_raises(self):
        with pytest.raises(ValueError):
            STG(None, thresholds=[50], labels=["lo", "hi"])

    # thresholds
    @settings(max_examples=10, deadline=None)
    @given(t=some.floats(min_value=1, max_value=99, allow_nan=False))
    def test_thresholds_single(self, t):
        g = STG([0.0, 100.0], thresholds=[t], labels=["lo", "hi"])
        assert g.tolist() == ["lo", "hi"]

    def test_thresholds_multiple(self):
        g = STG([10.0, 60.0, 90.0], thresholds=[50, 80], labels=["a", "b", "c"])
        assert g.tolist() == ["a", "b", "c"]

    @pytest.mark.parametrize("bad", [[80, 50], [50, 50], [10, 5, 20]])
    def test_thresholds_not_increasing_raises(self, bad):
        labels = [f"b{i}" for i in range(len(bad) + 1)]
        with pytest.raises(ValueError, match="increasing"):
            STG([10.0, 20.0], thresholds=bad, labels=labels)

    def test_thresholds_none_raises(self):
        with pytest.raises(ValueError):
            STG([10.0], thresholds=None, labels=["lo", "hi"])

    def test_thresholds_non_numeric_raises(self):
        with pytest.raises(ValueError):
            STG([10.0], thresholds=["x"], labels=["lo", "hi"])

    # labels
    def test_labels_three_bands(self):
        g = STG([10.0, 60.0, 90.0], thresholds=[50, 80], labels=["low", "mid", "high"])
        assert list(g.cat.categories) == ["low", "mid", "high"]

    @pytest.mark.parametrize("thresholds,labels", [([50], ["a", "b", "c"]), ([50, 80], ["a", "b"])])
    def test_labels_length_mismatch_raises(self, thresholds, labels):
        with pytest.raises(ValueError, match="one more element"):
            STG([10.0], thresholds=thresholds, labels=labels)

    def test_labels_not_unique_raises(self):
        with pytest.raises(ValueError, match="unique"):
            STG([10.0], thresholds=[50, 80], labels=["a", "a", "b"])

    def test_labels_none_raises(self):
        with pytest.raises(ValueError):
            STG([10.0], thresholds=[50], labels=None)

    # score_range
    def test_score_range_percent(self):
        g = STG([0.2, 60.0], thresholds=[50], labels=["lo", "hi"], score_range="percent")
        assert g.tolist() == ["lo", "hi"]

    def test_score_range_proba(self):
        g = STG([0.2, 0.6], thresholds=[0.5], labels=["lo", "hi"], score_range="proba")
        assert g.tolist() == ["lo", "hi"]

    def test_score_range_invalid_option_raises(self):
        with pytest.raises(ValueError):
            STG([0.2], thresholds=[0.5], labels=["lo", "hi"], score_range="fraction")

    def test_score_range_proba_threshold_out_of_range_raises(self):
        # A percent-scale threshold under proba bounds is rejected (no silent mix).
        with pytest.raises(ValueError, match="proba"):
            STG([0.2], thresholds=[80], labels=["lo", "hi"], score_range="proba")

    def test_score_range_percent_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="percent"):
            STG([10.0], thresholds=[150], labels=["lo", "hi"], score_range="percent")


class TestScoreToGroupComplex:
    """Combinations and edge interactions."""

    def test_ordered_categorical_supports_comparison(self):
        g = STG([10.0, 60.0, 90.0], thresholds=[50, 80], labels=["low", "mid", "high"])
        # Ordered categoricals support < / > against a category
        assert (g > "low").tolist() == [False, True, True]

    def test_all_below_first_threshold(self):
        g = STG([1.0, 2.0, 3.0], thresholds=[50, 80], labels=["a", "b", "c"])
        assert g.tolist() == ["a", "a", "a"]

    def test_all_above_last_threshold(self):
        g = STG([90.0, 95.0], thresholds=[50, 80], labels=["a", "b", "c"])
        assert g.tolist() == ["c", "c"]

    def test_mixed_nan_and_values_keep_index(self):
        s = pd.Series([np.nan, 10.0, 90.0], index=["x", "y", "z"])
        g = STG(s, thresholds=[50], labels=["lo", "hi"])
        assert list(g.index) == ["x", "y", "z"]
        assert pd.isna(g.loc["x"]) and g.loc["y"] == "lo" and g.loc["z"] == "hi"

    def test_negative_scores_allowed_when_in_range(self):
        # score_range guards thresholds, not scores; a below-range score just maps to band 0
        g = STG([-5.0, 60.0], thresholds=[50], labels=["lo", "hi"])
        assert g.tolist() == ["lo", "hi"]

    def test_empty_scores_raises(self):
        # An empty score vector has nothing to classify; the array check rejects it.
        with pytest.raises(ValueError):
            STG(np.array([], dtype=float), thresholds=[50], labels=["lo", "hi"])

    def test_series_categorical_name(self):
        g = STG([10.0], thresholds=[50], labels=["lo", "hi"])
        assert g.name == "group"

    def test_labels_length_and_threshold_order_both_wrong(self):
        # length mismatch is reported (validated before order)
        with pytest.raises(ValueError, match="one more element"):
            STG([10.0], thresholds=[80, 50], labels=["a", "b", "c", "d"])

    def test_thresholds_tuple_accepted(self):
        g = STG([10.0, 90.0], thresholds=(50,), labels=["lo", "hi"])
        assert g.tolist() == ["lo", "hi"]


class TestScoreToGroupGoldenValues:
    """Hand-computed band assignments, boundary inclusion, and plot-colouring parity."""

    def test_boundary_inclusion_right_open(self):
        # Threshold is an inclusive lower bound: exactly-at-threshold -> higher band.
        g = STG([49.999, 50.0, 79.999, 80.0], thresholds=[50, 80], labels=["low", "mid", "high"])
        assert g.tolist() == ["low", "mid", "mid", "high"]

    def test_golden_full_assignment(self):
        scores = [0, 10, 49, 50, 65, 80, 100]
        g = STG(scores, thresholds=[50, 80], labels=["low", "mid", "high"])
        assert g.tolist() == ["low", "low", "low", "mid", "mid", "high", "high"]

    def test_codes_match_band_index(self):
        # The categorical codes equal the shared backend band-index kernel.
        from aaanalysis.prediction._backend.aa_pred.aa_pred_group import assign_band_index
        scores = np.array([5.0, 55.0, 85.0, 50.0, 80.0])
        g = STG(scores, thresholds=[50, 80], labels=["a", "b", "c"])
        expected = assign_band_index(scores, [50.0, 80.0])
        assert g.cat.codes.tolist() == list(expected)

    def test_plot_band_parity(self):
        # predict_group(band=True) colours each bar by score_to_group's boundary rule:
        # the per-bin band index equals score_to_group's code for that bin's left edge.
        from aaanalysis.prediction._aa_pred_plot import _band_index
        thresholds = [50.0, 80.0]
        for edge in [0.0, 49.9, 50.0, 65.0, 80.0, 99.0]:
            code = STG([edge], thresholds=thresholds, labels=["a", "b", "c"]).cat.codes.iloc[0]
            assert _band_index(edge, thresholds) == int(code)
