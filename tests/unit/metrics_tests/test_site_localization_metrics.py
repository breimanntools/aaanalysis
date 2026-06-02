"""This is a script to test the site-localization metrics and score smoothing.

Covers Stage-3 decisions D8 and D9b — the new public metrics for windowed
protease / PTM prediction:

* ``comp_per_protein_ap`` — per-protein average precision (with optional ``+/-k``
  positional tolerance), NaN-aware.
* ``comp_detection_metrics`` — pooled TP/FP/FN/TN -> recall / precision / F1 /
  MCC at a fixed threshold.
* ``comp_bootstrap_ci`` — percentile bootstrap CI of the mean (seeded).
* ``comp_smooth_scores`` — peak-preserving triangular / Gaussian smoothing, NaN-aware.
"""
import numpy as np
import pytest

import aaanalysis as aa

aa.options["verbose"] = False


# Helper functions
def _perfect():
    """Two proteins where the top-ranked residues are exactly the positives."""
    list_scores = [np.array([0.9, 0.1, 0.8, 0.2]), np.array([0.1, 0.9, 0.2, 0.7])]
    list_positions = [[0, 2], [1, 3]]
    return list_scores, list_positions


class TestCompPerProteinAP:
    """Normal + negative cases for comp_per_protein_ap."""

    def test_perfect_ranking_is_one(self):
        s, p = _perfect()
        ap = aa.comp_per_protein_ap(list_scores=s, list_positions=p)
        assert np.allclose(ap, [1.0, 1.0])

    def test_returns_one_value_per_protein(self):
        s, p = _perfect()
        ap = aa.comp_per_protein_ap(list_scores=s, list_positions=p)
        assert ap.shape == (2,)

    def test_no_positives_is_nan(self):
        ap = aa.comp_per_protein_ap(list_scores=[np.array([0.1, 0.2])], list_positions=[[]])
        assert np.isnan(ap[0])

    def test_all_nan_scores_is_nan(self):
        ap = aa.comp_per_protein_ap(list_scores=[np.array([np.nan, np.nan])], list_positions=[[0]])
        assert np.isnan(ap[0])

    def test_tolerance_zero_misses_offby_one(self):
        # Peak at index 1, true site at index 2 -> AP < 1 without tolerance.
        ap = aa.comp_per_protein_ap(list_scores=[np.array([0.1, 0.9, 0.2, 0.05])],
                                    list_positions=[[2]], tolerance=0)
        assert ap[0] < 1.0

    def test_tolerance_one_rescues_offby_one(self):
        ap = aa.comp_per_protein_ap(list_scores=[np.array([0.1, 0.9, 0.2, 0.05])],
                                    list_positions=[[2]], tolerance=1)
        assert ap[0] == 1.0

    def test_nanmean_aggregates(self):
        s, p = _perfect()
        ap = aa.comp_per_protein_ap(list_scores=s, list_positions=p)
        assert np.isclose(np.nanmean(ap), 1.0)

    # Negative tests
    def test_invalid_length_mismatch(self):
        s, p = _perfect()
        with pytest.raises(ValueError):
            aa.comp_per_protein_ap(list_scores=s, list_positions=p[:1])

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            aa.comp_per_protein_ap(list_scores=[], list_positions=[])

    def test_invalid_tolerance_negative(self):
        s, p = _perfect()
        with pytest.raises(ValueError):
            aa.comp_per_protein_ap(list_scores=s, list_positions=p, tolerance=-1)

    def test_invalid_tolerance_float(self):
        s, p = _perfect()
        with pytest.raises(ValueError):
            aa.comp_per_protein_ap(list_scores=s, list_positions=p, tolerance=1.5)


class TestCompDetectionMetrics:
    """Normal + negative cases for comp_detection_metrics."""

    def test_keys(self):
        s, p = _perfect()
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=0.5)
        assert set(dm) == {"recall", "precision", "f1", "mcc", "tp", "fp", "fn", "tn"}

    def test_perfect_recall_precision(self):
        s, p = _perfect()
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=0.5)
        assert dm["tp"] == 4 and dm["fp"] == 0 and dm["fn"] == 0
        assert dm["recall"] == 1.0 and dm["precision"] == 1.0 and dm["f1"] == 1.0

    def test_high_threshold_drops_calls(self):
        s, p = _perfect()
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=0.95)
        assert dm["tp"] == 0 and dm["fn"] == 4

    def test_counts_are_ints(self):
        s, p = _perfect()
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p)
        assert all(isinstance(dm[k], int) for k in ["tp", "fp", "fn", "tn"])

    def test_tolerance_credits_neighbour(self):
        # Call at index 1, true site at index 2; tolerance=1 makes it a TP.
        dm = aa.comp_detection_metrics(list_scores=[np.array([0.0, 0.9, 0.0])],
                                       list_positions=[[2]], threshold=0.5, tolerance=1)
        assert dm["tp"] == 1 and dm["fp"] == 0

    def test_nan_scores_not_counted(self):
        dm = aa.comp_detection_metrics(list_scores=[np.array([np.nan, 0.9])],
                                       list_positions=[[1]], threshold=0.5)
        assert dm["tp"] == 1 and dm["tn"] == 0

    # Negative tests
    def test_invalid_length_mismatch(self):
        s, p = _perfect()
        with pytest.raises(ValueError):
            aa.comp_detection_metrics(list_scores=s, list_positions=p[:1])

    def test_invalid_threshold_none(self):
        s, p = _perfect()
        with pytest.raises(ValueError):
            aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=None)

    def test_invalid_tolerance_negative(self):
        s, p = _perfect()
        with pytest.raises(ValueError):
            aa.comp_detection_metrics(list_scores=s, list_positions=p, tolerance=-2)


class TestCompBootstrapCI:
    """Normal + negative cases for comp_bootstrap_ci."""

    def test_returns_triple(self):
        out = aa.comp_bootstrap_ci(values=np.array([0.2, 0.3, 0.25, 0.28, 0.22]), n_rounds=100, seed=0)
        assert isinstance(out, dict) and set(out) == {"mean", "ci_low", "ci_high"}

    def test_ordered(self):
        out = aa.comp_bootstrap_ci(values=np.array([0.2, 0.3, 0.25, 0.28, 0.22]), n_rounds=200, seed=0)
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]

    def test_deterministic_with_seed(self):
        v = np.array([0.2, 0.3, 0.25, 0.28, 0.22])
        a = aa.comp_bootstrap_ci(values=v, n_rounds=200, seed=1)
        b = aa.comp_bootstrap_ci(values=v, n_rounds=200, seed=1)
        assert a == b

    def test_nan_values_dropped(self):
        out = aa.comp_bootstrap_ci(values=np.array([0.2, np.nan, 0.3]), n_rounds=50, seed=0)
        assert not np.isnan(out["mean"])

    def test_all_nan_is_nan(self):
        out = aa.comp_bootstrap_ci(values=np.array([np.nan, np.nan]), n_rounds=50, seed=0)
        assert all(np.isnan(x) for x in out.values())

    # Negative tests
    def test_invalid_n_rounds_zero(self):
        with pytest.raises(ValueError):
            aa.comp_bootstrap_ci(values=np.array([0.2, 0.3]), n_rounds=0, seed=0)

    def test_invalid_ci_out_of_range(self):
        with pytest.raises(ValueError):
            aa.comp_bootstrap_ci(values=np.array([0.2, 0.3]), ci=1.5, seed=0)

    def test_invalid_seed_negative(self):
        with pytest.raises(ValueError):
            aa.comp_bootstrap_ci(values=np.array([0.2, 0.3]), seed=-1)


class TestSmoothScores:
    """Normal + negative cases for comp_smooth_scores."""

    def test_length_preserved(self):
        raw = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        assert aa.comp_smooth_scores(scores=raw, window=1).shape == raw.shape

    def test_peak_preserved(self):
        raw = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        sm = aa.comp_smooth_scores(scores=raw, method="triangular", window=1, peak_preserving=True)
        assert sm[2] >= 1.0

    def test_non_peak_preserving_attenuates(self):
        raw = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        sm = aa.comp_smooth_scores(scores=raw, method="triangular", window=1, peak_preserving=False)
        assert sm[2] < 1.0

    def test_gaussian_method(self):
        raw = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        sm = aa.comp_smooth_scores(scores=raw, method="gaussian", window=2)
        assert sm.shape == raw.shape

    def test_nan_aware(self):
        raw = np.array([np.nan, 1.0, 0.0, np.nan])
        sm = aa.comp_smooth_scores(scores=raw, window=1)
        # Finite neighbours fill index 0; an all-NaN neighbourhood stays NaN.
        assert sm.shape == raw.shape

    def test_spreads_signal_to_neighbours(self):
        raw = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        sm = aa.comp_smooth_scores(scores=raw, method="triangular", window=1, peak_preserving=False)
        assert sm[1] > 0.0 and sm[3] > 0.0

    # Negative tests
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            aa.comp_smooth_scores(scores=np.array([1.0, 2.0]), method="boxcar")

    def test_invalid_window_zero(self):
        with pytest.raises(ValueError):
            aa.comp_smooth_scores(scores=np.array([1.0, 2.0]), window=0)

    def test_invalid_peak_preserving(self):
        with pytest.raises(ValueError):
            aa.comp_smooth_scores(scores=np.array([1.0, 2.0]), peak_preserving="yes")

    def test_invalid_sigma_negative(self):
        with pytest.raises(ValueError):
            aa.comp_smooth_scores(scores=np.array([1.0, 2.0]), method="gaussian", sigma=-1.0)


class TestSiteLocalizationComplex:
    """Cross-metric interactions and realistic pipelines."""

    def test_ap_then_bootstrap(self):
        s, p = _perfect()
        ap = aa.comp_per_protein_ap(list_scores=s, list_positions=p)
        out = aa.comp_bootstrap_ci(values=ap, n_rounds=100, seed=0)
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]

    def test_smoothing_lifts_true_site_score(self):
        # A true site adjacent to a high-scoring neighbour: smoothing raises the
        # site's own score (signal bleeds in from the neighbour).
        raw = np.array([0.0, 0.9, 0.1, 0.0, 0.0])  # true site at index 2, peak at 1
        sm = aa.comp_smooth_scores(scores=raw, method="triangular", window=1, peak_preserving=False)
        assert sm[2] > raw[2]

    def test_detection_vs_ranking_distinct(self):
        # All scores below threshold -> 0 detection, but ranking AP can still be 1.
        s = [np.array([0.4, 0.1, 0.3])]
        p = [[0]]
        ap = aa.comp_per_protein_ap(list_scores=s, list_positions=p)[0]
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=0.5)
        assert ap == 1.0 and dm["tp"] == 0

    def test_tolerance_consistent_between_ap_and_detection(self):
        s = [np.array([0.0, 0.9, 0.0])]
        p = [[2]]
        ap = aa.comp_per_protein_ap(list_scores=s, list_positions=p, tolerance=1)[0]
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=0.5, tolerance=1)
        assert ap == 1.0 and dm["tp"] == 1

    def test_multi_protein_pooling(self):
        s = [np.array([0.9, 0.1]), np.array([0.2, 0.8]), np.array([0.1, 0.1])]
        p = [[0], [1], []]
        dm = aa.comp_detection_metrics(list_scores=s, list_positions=p, threshold=0.5)
        # Two true sites, both called; protein 3 has no positives and no calls.
        assert dm["tp"] == 2 and dm["fn"] == 0
